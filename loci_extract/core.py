"""Core extraction orchestration.

``extract_document(pdf_path, opts)`` is the single entry point used by
the CLI, the API, and the webapp. It:

  1. detects per-page type (text vs image)
  2. extracts page text via pdfminer (text pages), OCR (image pages, default),
     or the VLM (all pages when ``opts.vision=True``)
  3. runs boundary detection to split multi-section PDFs
  4. for each section: detects family, augments the prompt, calls the LLM
  5. validates with the pydantic schema and redacts SSN
  6. applies post-hoc dedup for W-2 copies (belt-and-suspenders)
  7. merges section results into one Extraction

``extract_batch(paths, opts)`` is a loop that returns one Extraction per
input PDF. It does not merge across PDFs — callers typically want a
per-file mapping. The CLI batch path merges manually for CSV/Lacerte.

Pure library: no prints, no ``sys.exit``. Callers wire verbose output via
``progress_callback``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loci_extract.detector import detect_page_types, identify_doc_types
from loci_extract.extractor import extract_text_pages
from loci_extract.llm import make_client, parse_extraction
from loci_extract.ocr import extract_pages as ocr_extract_pages
from loci_extract.prompts import PER_DOC_HINTS, SYSTEM_PROMPT
from loci_extract.schema import Extraction
from loci_extract.vision import vision_extract_pages

logger = logging.getLogger("loci_extract.core")

ProgressCallback = Callable[[str], None]


@dataclass
class ExtractionOptions:
    model_url: str
    model_name: str = "local"
    api_key: str = "local"
    ocr_engine: Literal["auto", "tesseract", "easyocr", "paddleocr"] = "auto"
    gpu: Literal["auto", "true", "false"] = "auto"
    dpi: int = 300
    vision: bool = False
    vision_model: str | None = None
    redact: bool = True
    temperature: float = 0.0
    max_tokens: int = 4096
    retry: int = 2
    # Phase 1 additions for financial documents:
    chunk_size_tokens: int = 6000  # cap input tokens per LLM chunk
    verify_totals: bool = True     # run Python-side totals verifier on financial docs
    fix_orientation: bool = True   # auto-rotate scanned-sideways pages via Tesseract OSD
    force_family: str | None = None  # override family detection: "tax" | "financial_simple" | ...
    max_parallel: int = 4          # concurrent chunk LLM calls; 1 = sequential


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def _dedup_documents(extraction: Extraction) -> Extraction:
    """Remove duplicate W-2 records by (employer_ein, employee_ssn_last4, tax_year).

    Belt-and-suspenders — the system prompt also tells the model to dedup,
    but 7B/14B-tier models sometimes emit multiple copies.
    """
    seen: set[tuple] = set()
    deduped = []
    for doc in extraction.documents:
        if doc.document_type == "W2":
            data = doc.data or {}
            employer = data.get("employer", {}) or {}
            employee = data.get("employee", {}) or {}
            key = (
                "W2",
                employer.get("ein") or employer.get("name"),
                employee.get("ssn_last4") or employee.get("name"),
                doc.tax_year,
            )
            if key in seen:
                continue
            seen.add(key)
        deduped.append(doc)
    if len(deduped) != len(extraction.documents):
        logger.info("Deduplicated %d → %d documents", len(extraction.documents), len(deduped))
    return Extraction(documents=deduped)


# ---------------------------------------------------------------------------
# Text gathering (per page → dict)
# ---------------------------------------------------------------------------


def _gather_pages(
    pdf_path: Path,
    opts: ExtractionOptions,
    client,
    progress: ProgressCallback | None,
) -> dict[int, str]:
    """Extract per-page text. Returns ``{page_num: text}``.

    When ``opts.vision=True``, ALL pages are rendered as images and sent to
    the VLM — the text layer is ignored entirely. This is the reliable path
    for scans, image-only PDFs, and encoding-broken PDFs.

    When ``opts.vision=False`` (default), text-layer pages go through pdfminer
    and image pages go through the configured OCR engine.
    """
    from loci_extract.detector import get_extraction_strategy
    strategy_info = get_extraction_strategy(pdf_path)
    encoding_broken = strategy_info.get("encoding_broken", False)
    if encoding_broken and progress:
        progress(f"strategy: {strategy_info['reason'][:100]}")

    page_types = detect_page_types(pdf_path)
    if not page_types:
        raise RuntimeError(f"Could not detect pages in {pdf_path}")
    if progress:
        progress(f"pages detected: {len(page_types)} ({sum(1 for v in page_types.values() if v == 'text')} text, "
                 f"{sum(1 for v in page_types.values() if v == 'image')} image)")

    page_text: dict[int, str] = {}

    if opts.vision:
        # Vision mode: render ALL pages as images → VLM.
        # Ignores text layer entirely — best for scans and image-only PDFs.
        all_pages = sorted(page_types.keys())
        vision_model = opts.vision_model or opts.model_name
        if progress:
            progress(f"rendering {len(all_pages)} page(s) for VLM ({vision_model}) at {opts.dpi} DPI")
        page_text = vision_extract_pages(
            client,
            pdf_path,
            all_pages,
            vision_model=vision_model,
            system_prompt="You are a careful OCR system. Transcribe visible text exactly.",
            dpi=opts.dpi,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
        )
    else:
        # Standard path: text pages → pdfminer, image pages → OCR
        if encoding_broken:
            text_pages: list[int] = []
            image_pages = list(page_types.keys())
        else:
            text_pages = [p for p, t in page_types.items() if t == "text"]
            image_pages = [p for p, t in page_types.items() if t == "image"]

        if text_pages:
            if progress:
                progress(f"extracting text layer for {len(text_pages)} page(s) via pdfminer")
            page_text.update(extract_text_pages(pdf_path, text_pages))

        if image_pages:
            if progress:
                progress(f"OCRing {len(image_pages)} image page(s) via {opts.ocr_engine}")
            ocr_text = ocr_extract_pages(
                pdf_path,
                image_pages,
                engine=opts.ocr_engine,
                gpu=opts.gpu,
                dpi=opts.dpi,
                fix_orientation=opts.fix_orientation,
            )
            page_text.update(ocr_text)

    return page_text


def _concat_pages(page_text: dict[int, str]) -> str:
    """Join per-page text with ``--- PAGE N ---`` markers."""
    ordered = sorted(page_text.items())
    parts: list[str] = []
    for page_num, text in ordered:
        text = (text or "").strip()
        if not text:
            continue
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(text)
    return "\n".join(parts)


# Keep the old name as an alias for callers that use it directly
# (e.g. detect_document, API endpoints).
def _gather_page_text(
    pdf_path: Path,
    opts: ExtractionOptions,
    client,
    progress: ProgressCallback | None,
) -> str:
    """Legacy wrapper: gather pages and concatenate. Used by detect_document()."""
    return _concat_pages(_gather_pages(pdf_path, opts, client, progress))


# ---------------------------------------------------------------------------
# Section extraction (one logical document section → Extraction)
# ---------------------------------------------------------------------------


def _extract_section(
    client,
    page_text: dict[int, str],
    opts: ExtractionOptions,
    progress: ProgressCallback | None,
) -> Extraction:
    """Extract one logical section: concat → family detect → LLM dispatch."""
    raw_text = _concat_pages(page_text)
    if not raw_text.strip():
        return Extraction(documents=[])

    # Detector hints
    hint_types = identify_doc_types(raw_text)
    hint_text = ""
    if hint_types:
        sections = [f"[{t}] {PER_DOC_HINTS[t]}" for t in hint_types if t in PER_DOC_HINTS]
        if sections:
            hint_text = (
                "The document text below is pre-identified as likely containing: "
                + ", ".join(hint_types)
                + ". Hints per type:\n"
                + "\n".join(sections)
                + "\n\n"
            )
    user_text = f"{hint_text}Document text (multiple pages separated by --- PAGE N --- markers):\n\n{raw_text}"

    # Family routing
    family = _resolve_family(raw_text, opts.force_family)
    if progress:
        progress(f"detected family: {family}")

    if family == "tax":
        extraction = parse_extraction(
            client,
            user_text,
            system_prompt=SYSTEM_PROMPT,
            model_name=opts.model_name,
            temperature=opts.temperature,
            max_tokens=opts.max_tokens,
            retry=opts.retry,
            redact=opts.redact,
        )
        extraction = _dedup_documents(extraction)
    else:
        from loci_extract.core_chunked import extract_financial_document

        extraction = extract_financial_document(
            client=client,
            raw_text=raw_text,
            opts=opts,
            family=family,
            progress=progress,
        )

    return extraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_XLSX_EXTS = {".xlsx", ".xlsm", ".xltx", ".xltm"}


def extract_document(
    pdf_path: str | Path,
    opts: ExtractionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Extraction:
    """Extract one document (PDF or XLSX) → validated ``Extraction``.

    For PDFs: runs boundary detection to split multi-section documents
    (e.g. BS + P&L + GL in one file), then extracts each section
    independently with its own family detection and LLM call.

    For XLSX: single-section extraction (XLSX files don't have page
    boundaries).

    Raises ``json.JSONDecodeError`` / ``pydantic.ValidationError`` if the
    LLM fails all retries, ``RuntimeError`` on empty/invalid input.
    """
    pdf_path = Path(pdf_path)
    client = make_client(opts.model_url, api_key=opts.api_key)

    if progress_callback:
        progress_callback(f"opening {pdf_path}")

    # XLSX path: no OCR, no vision, no encoding checks, no boundaries.
    if pdf_path.suffix.lower() in _XLSX_EXTS:
        from loci_extract.xlsx import extract_xlsx_text
        if progress_callback:
            progress_callback(f"reading XLSX via openpyxl ({pdf_path.name})")
        raw_text = extract_xlsx_text(pdf_path)
        if not raw_text.strip():
            raise RuntimeError(f"No cells could be read from {pdf_path}")
        # XLSX is always single-section — extract directly
        page_text = {1: raw_text}
        extraction = _extract_section(client, page_text, opts, progress_callback)
        if progress_callback:
            progress_callback(f"extracted {len(extraction.documents)} document(s)")
        return extraction

    # PDF path: gather per-page text, then boundary-detect + per-section extract.
    page_text = _gather_pages(pdf_path, opts, client, progress_callback)
    if not any((t or "").strip() for t in page_text.values()):
        raise RuntimeError(f"No text could be recovered from {pdf_path} (all pages empty after OCR/vision)")

    # Boundary detection — split multi-section PDFs
    from loci_extract.boundary_detector import detect_boundaries
    pages_list = [{"page": p, "text": t} for p, t in sorted(page_text.items())]
    sections = detect_boundaries(pages_list)

    if len(sections) <= 1:
        # Single document — fast path, no splitting overhead
        if progress_callback:
            progress_callback(f"calling LLM {opts.model_name} at {opts.model_url}")
        extraction = _extract_section(client, page_text, opts, progress_callback)
    else:
        # Multi-section: extract each independently and merge
        if progress_callback:
            progress_callback(
                f"boundary detection: {len(sections)} section(s) — "
                + ", ".join(f"{s.document_type} (pp {s.start_page}-{s.end_page})" for s in sections)
            )
        all_docs = []
        for section in sections:
            subset = {p: t for p, t in page_text.items()
                      if section.start_page <= p <= section.end_page}
            if not any((t or "").strip() for t in subset.values()):
                continue
            if progress_callback:
                progress_callback(
                    f"extracting section: {section.document_type} "
                    f"(pages {section.start_page}-{section.end_page})"
                )
            section_extraction = _extract_section(client, subset, opts, progress_callback)
            all_docs.extend(section_extraction.documents)
        extraction = Extraction(documents=all_docs)

    if progress_callback:
        progress_callback(f"extracted {len(extraction.documents)} document(s)")
    return extraction


def _resolve_family(raw_text: str, force: str | None) -> str:
    """Pick document family. ``force`` overrides; otherwise the master detector
    picks tax-first, financial-second, falling back to ``tax`` (preserves the
    pre-refactor behavior on documents that are neither identifiable)."""
    if force:
        return force
    from loci_extract.detector import detect_financial_document_type, detect_tax_document_type

    tax = detect_tax_document_type(raw_text)
    if tax.document_type != "UNKNOWN" and tax.confidence >= 0.5:
        return "tax"
    fin_type = detect_financial_document_type(raw_text)
    if fin_type != "FINANCIAL_UNKNOWN":
        from loci_extract.prompts import DOCUMENT_FAMILY_MAP, DocumentFamily
        family = DOCUMENT_FAMILY_MAP.get(fin_type, DocumentFamily.FINANCIAL_SIMPLE)
        return family.value if hasattr(family, "value") else family
    return "tax"  # safe fallback: original tax-prompt behavior


def extract_batch(
    pdf_paths: list[str | Path],
    opts: ExtractionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[tuple[Path, Extraction]]:
    """Run ``extract_document`` over each path. Returns ``[(path, extraction), ...]``.

    Errors on individual files are logged and the path is paired with an
    empty Extraction so callers can report per-file failures without
    aborting the whole batch.
    """
    results: list[tuple[Path, Extraction]] = []
    for raw in pdf_paths:
        p = Path(raw)
        if progress_callback:
            progress_callback(f"--- {p} ---")
        try:
            results.append((p, extract_document(p, opts, progress_callback=progress_callback)))
        except Exception as exc:
            logger.error("extraction failed for %s: %s", p, exc)
            if progress_callback:
                progress_callback(f"ERROR: {p}: {exc}")
            results.append((p, Extraction(documents=[])))
    return results


def detect_document(
    pdf_path: str | Path,
    opts: ExtractionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Detect document type without calling the LLM.

    Returns a dict with detection results: document_type, document_family,
    confidence, strategy, encoding_broken, and optional tax-specific fields.
    """
    from dataclasses import asdict

    from loci_extract.detector import detect

    pdf_path = Path(pdf_path)

    if pdf_path.suffix.lower() in _XLSX_EXTS:
        from loci_extract.xlsx import extract_xlsx_text
        if progress_callback:
            progress_callback(f"reading XLSX via openpyxl ({pdf_path.name})")
        raw_text = extract_xlsx_text(pdf_path)
        if not raw_text.strip():
            raise RuntimeError(f"No cells could be read from {pdf_path}")
    else:
        client = make_client(opts.model_url, api_key=opts.api_key)
        raw_text = _gather_page_text(pdf_path, opts, client, progress_callback)
        if not raw_text.strip():
            raise RuntimeError(f"No text could be recovered from {pdf_path}")

    result = detect(pdf_path, raw_text)
    return asdict(result)


__all__ = ["ExtractionOptions", "extract_document", "extract_batch", "detect_document"]
