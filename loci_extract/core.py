"""Core extraction orchestration.

``extract_document(pdf_path, opts)`` is the single entry point used by
the CLI, the API, and the webapp. It:

  1. detects per-page type (text vs image)
  2. extracts page text via pdfminer (text pages), OCR (image pages, default),
     or the VLM (image pages when ``opts.vision=True``, or all pages when
     ``opts.vision=True`` is meant as "use vision everywhere")
  3. concatenates pages with ``\n---PAGE BREAK---\n``
  4. augments the prompt with per-doc-type hints from the detector
  5. calls the LLM to produce structured JSON
  6. validates with the pydantic schema and redacts SSN
  7. applies post-hoc dedup for W-2 copies (belt-and-suspenders)

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
# Text gathering (per page → one big blob)
# ---------------------------------------------------------------------------


def _gather_page_text(
    pdf_path: Path,
    opts: ExtractionOptions,
    client,
    progress: ProgressCallback | None,
) -> str:
    page_types = detect_page_types(pdf_path)
    if not page_types:
        raise RuntimeError(f"Could not detect pages in {pdf_path}")
    if progress:
        progress(f"pages detected: {len(page_types)} ({sum(1 for v in page_types.values() if v == 'text')} text, "
                 f"{sum(1 for v in page_types.values() if v == 'image')} image)")

    text_pages = [p for p, t in page_types.items() if t == "text"]
    image_pages = [p for p, t in page_types.items() if t == "image"]

    page_text: dict[int, str] = {}

    # Text pages → pdfminer
    if text_pages:
        if progress:
            progress(f"extracting text layer for {len(text_pages)} page(s) via pdfminer")
        page_text.update(extract_text_pages(pdf_path, text_pages))

    # Image pages — OCR or vision
    if image_pages:
        if opts.vision:
            vision_model = opts.vision_model or opts.model_name
            if progress:
                progress(f"rendering {len(image_pages)} image page(s) for VLM ({vision_model})")
            vision_text = vision_extract_pages(
                client,
                pdf_path,
                image_pages,
                vision_model=vision_model,
                system_prompt="You are a careful OCR system. Transcribe visible text exactly.",
                dpi=opts.dpi,
                max_tokens=opts.max_tokens,
                temperature=opts.temperature,
            )
            page_text.update(vision_text)
        else:
            if progress:
                progress(f"OCRing {len(image_pages)} image page(s) via {opts.ocr_engine}")
            ocr_text = ocr_extract_pages(
                pdf_path,
                image_pages,
                engine=opts.ocr_engine,
                gpu=opts.gpu,
                dpi=opts.dpi,
            )
            page_text.update(ocr_text)

    # Concatenate in page order with explicit breaks so the LLM can see
    # the original pagination.
    ordered = sorted(page_text.items())
    parts: list[str] = []
    for page_num, text in ordered:
        text = (text or "").strip()
        if not text:
            continue
        parts.append(f"--- PAGE {page_num} ---")
        parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_document(
    pdf_path: str | Path,
    opts: ExtractionOptions,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Extraction:
    """Extract one PDF → validated ``Extraction``.

    Raises ``json.JSONDecodeError`` / ``pydantic.ValidationError`` if the
    LLM fails all retries, ``RuntimeError`` on empty/invalid PDFs.
    """
    pdf_path = Path(pdf_path)
    client = make_client(opts.model_url, api_key=opts.api_key)

    if progress_callback:
        progress_callback(f"opening {pdf_path}")

    raw_text = _gather_page_text(pdf_path, opts, client, progress_callback)
    if not raw_text.strip():
        raise RuntimeError(f"No text could be recovered from {pdf_path} (all pages empty after OCR/vision)")

    # Detector hints (keyword-based). We prepend per-doc nudges for every
    # doc type the detector found — the model sees a short "your document(s)
    # appear to include X, Y" preamble to bias layout interpretation.
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

    if progress_callback:
        progress_callback(f"calling LLM {opts.model_name} at {opts.model_url}")

    # ── Family routing: tax docs use the original parse_extraction path
    # (preserves byte-exact tax behavior — the golden file regression depends
    # on this). Financial docs go through the new chunked + verified pipeline.
    family = _resolve_family(raw_text, opts.force_family)
    if progress_callback:
        progress_callback(f"detected family: {family}")

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
        # Financial branch — chunk → multi-LLM → merge → verify → derived
        from loci_extract.core_chunked import extract_financial_document

        extraction = extract_financial_document(
            client=client,
            raw_text=raw_text,
            opts=opts,
            family=family,
            progress=progress_callback,
        )

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


__all__ = ["ExtractionOptions", "extract_document", "extract_batch"]
