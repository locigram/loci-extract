from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path

import time
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.capabilities import cuda_available, detect_capabilities, detect_compare_pipelines
from app.classification.routing import classify_with_profile
from app.profiles import get_profile
from app.profiles.schema import ExtractionProfile
from app.router import UnsupportedDocumentError, extract_file
from app.schemas import ExtractionPayload, StructuredExtractionResponse
from app.structured.router import build_structured_document

app = FastAPI(title="loci-extract", version="0.1.0")

_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def ui_root():
    return FileResponse(_STATIC_DIR / "index.html")
VALID_OCR_STRATEGIES = {"auto", "always", "never", "force_image", "vlm", "vlm_hybrid"}
VALID_OCR_BACKENDS = {"auto", "tesseract", "paddleocr"}
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _max_upload_bytes() -> int:
    raw = os.getenv("LOCI_EXTRACT_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = DEFAULT_MAX_UPLOAD_BYTES
    return value if value > 0 else DEFAULT_MAX_UPLOAD_BYTES


def _snippet(text: str, *, max_len: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"



def _build_ocr_enrichment(payload: ExtractionPayload) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    page_provenance = payload.extra.get("page_provenance") if isinstance(payload.extra.get("page_provenance"), list) else []
    ocr_attempted = bool(payload.extra.get("ocr_attempted"))
    selected_pass = payload.extra.get("selected_ocr_pass")
    result_source = payload.extra.get("result_source")
    has_ocr_signals = ocr_attempted or selected_pass is not None or any(
        isinstance(entry, dict) and entry.get("source") in {"ocr", "parser_fallback"} for entry in page_provenance
    )
    if not has_ocr_signals:
        return None, []

    ocr_scores = [
        float(entry.get("ocr_score") or 0.0)
        for entry in page_provenance
        if isinstance(entry, dict) and isinstance(entry.get("ocr_score"), (int, float))
    ]
    average_score = payload.extra.get("ocr_average_score")
    if not isinstance(average_score, (int, float)):
        average_score = payload.extra.get("ocr_score")
    if not isinstance(average_score, (int, float)):
        average_score = round(sum(ocr_scores) / len(ocr_scores), 2) if ocr_scores else 0.0

    weak_pages: list[int] = []
    for entry in page_provenance:
        if not isinstance(entry, dict):
            continue
        if entry.get("source") not in {"ocr", "parser_fallback"}:
            continue
        page_number = entry.get("page_number")
        score = float(entry.get("ocr_score", 0.0) or 0.0)
        text_length = int(entry.get("text_length", 0) or 0)
        if isinstance(page_number, int) and (score < 10 or text_length < 20):
            weak_pages.append(page_number)

    evidence_snippets: list[dict[str, object]] = []
    for segment in payload.segments:
        if segment.type != "page":
            continue
        source = segment.metadata.get("source")
        if source not in {"ocr", "parser_fallback"}:
            continue
        page_number = segment.metadata.get("page_number")
        evidence_snippets.append(
            {
                "page_number": page_number,
                "source": source,
                "ocr_score": segment.metadata.get("ocr_score"),
                "selected_ocr_pass": segment.metadata.get("selected_ocr_pass"),
                "snippet": _snippet(segment.text),
            }
        )

    if not evidence_snippets and payload.raw_text.strip():
        evidence_snippets.append(
            {
                "page_number": 1,
                "source": result_source,
                "ocr_score": payload.extra.get("ocr_score") or average_score,
                "selected_ocr_pass": selected_pass,
                "snippet": _snippet(payload.raw_text),
            }
        )

    summary = {
        "attempted": ocr_attempted,
        "used": payload.extraction.ocr_used,
        "result_source": result_source,
        "average_score": round(float(average_score or 0.0), 2),
        "min_score": round(min(ocr_scores), 2) if ocr_scores else round(float(average_score or 0.0), 2),
        "max_score": round(max(ocr_scores), 2) if ocr_scores else round(float(average_score or 0.0), 2),
        "selected_passes": [
            entry.get("selected_ocr_pass")
            for entry in page_provenance
            if isinstance(entry, dict) and entry.get("selected_ocr_pass")
        ]
        or ([selected_pass] if selected_pass else []),
        "weak_pages": weak_pages,
        "low_quality": bool((average_score or 0.0) < 10 or weak_pages),
    }
    return summary, evidence_snippets[:3]



def _enrich_payload_extra(payload: ExtractionPayload) -> None:
    """Add stable extraction stats and provenance hints to payload.extra."""
    warning_codes = [warning.code for warning in payload.extraction.warnings]
    content_detected = bool(payload.raw_text.strip()) or bool(payload.segments)

    payload.extra.setdefault("segment_count", len(payload.segments))
    payload.extra.setdefault("chunk_count", len(payload.chunks))
    payload.extra.setdefault("warning_codes", warning_codes)
    payload.extra.setdefault("has_warnings", bool(warning_codes))
    payload.extra.setdefault("content_detected", content_detected)
    payload.extra.setdefault("empty_content", not content_detected)

    if payload.extraction.status == "partial" and not content_detected:
        payload.extra.setdefault("partial_reason", "empty_content")

    page_segments = [segment for segment in payload.segments if segment.type == "page"]
    payload.extra.setdefault("non_empty_page_count", len(page_segments))

    sheet_segments = [segment for segment in payload.segments if segment.type == "sheet"]
    payload.extra.setdefault("non_empty_sheet_count", len(sheet_segments))

    table_segments = [segment for segment in payload.segments if segment.type == "table"]
    payload.extra.setdefault("table_segment_count", len(table_segments))

    ocr_quality_summary, ocr_evidence_snippets = _build_ocr_enrichment(payload)
    if ocr_quality_summary is not None:
        payload.extra.setdefault("ocr_quality_summary", ocr_quality_summary)
        payload.extra.setdefault("ocr_evidence_snippets", ocr_evidence_snippets)


def _render_first_page_image(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
):
    """Render the first page of a PDF or return the image directly for layout classification."""
    try:
        from PIL import Image
        from io import BytesIO

        if filename.lower().endswith(".pdf") or mime_type == "application/pdf":
            import fitz

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) == 0:
                return None
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            return Image.open(BytesIO(pix.tobytes("png")))

        if mime_type.startswith("image/"):
            return Image.open(BytesIO(file_bytes))
    except Exception as exc:
        import logging
        logging.getLogger("loci.main").debug("Failed to render page image for %s: %s", filename, exc)
    return None


def _resolve_profile_params(
    profile_name: str | None,
    *,
    ocr_strategy: str | None,
    ocr_backend: str | None,
    mask_pii: bool | None,
    enable_llm_enrichment: bool | None,
) -> tuple[ExtractionProfile | None, str, str, bool, bool]:
    """Resolve profile defaults merged with explicit Form overrides.

    Returns (profile, ocr_strategy, ocr_backend, mask_pii, enable_llm_enrichment).
    """
    profile = None
    if profile_name:
        profile = get_profile(profile_name)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction profile '{profile_name}' not found.",
            )

    # Merge: explicit value wins over profile default, profile wins over system default
    resolved_ocr_strategy = ocr_strategy if ocr_strategy is not None else (
        profile.ocr_strategy if profile else "auto"
    )
    resolved_ocr_backend = ocr_backend if ocr_backend is not None else (
        profile.ocr_backend if profile else "auto"
    )
    resolved_mask_pii = mask_pii if mask_pii is not None else (
        profile.mask_pii if profile else True
    )
    resolved_llm = enable_llm_enrichment if enable_llm_enrichment is not None else (
        profile.enable_llm_enrichment if profile else False
    )
    return profile, resolved_ocr_strategy, resolved_ocr_backend, resolved_mask_pii, resolved_llm


def _validate_ocr_strategy(ocr_strategy: str) -> None:
    if ocr_strategy not in VALID_OCR_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ocr_strategy '{ocr_strategy}'. Expected one of: auto, always, never.",
        )


def _validate_ocr_backend(ocr_backend: str) -> None:
    if ocr_backend not in VALID_OCR_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ocr_backend '{ocr_backend}'. Expected one of: auto, tesseract, paddleocr.",
        )


def _save_upload_tempfile(file_bytes: bytes, filename: str) -> Path:
    max_upload_bytes = _max_upload_bytes()
    if len(file_bytes) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Uploaded file is too large ({len(file_bytes)} bytes). "
                f"Maximum allowed size is {max_upload_bytes} bytes."
            ),
        )
    suffix = Path(filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        return Path(tmp.name)


def _extract_from_path(
    *,
    tmp_path: Path,
    filename: str,
    mime_type: str,
    include_chunks: bool,
    ocr_strategy: str,
    ocr_backend: str = "auto",
) -> ExtractionPayload:
    try:
        payload = extract_file(
            tmp_path,
            filename or tmp_path.name,
            mime_type,
            ocr_strategy=ocr_strategy,
            ocr_backend=ocr_backend,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    if not include_chunks:
        payload.chunks = []
    payload.extra.setdefault("ocr_strategy", ocr_strategy)
    payload.extra.setdefault("ocr_backend_requested", ocr_backend)
    _enrich_payload_extra(payload)
    return payload


def _extract_payload_from_upload(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    include_chunks: bool,
    ocr_strategy: str,
    ocr_backend: str = "auto",
) -> ExtractionPayload:
    tmp_path = _save_upload_tempfile(file_bytes, filename)
    try:
        return _extract_from_path(
            tmp_path=tmp_path,
            filename=filename,
            mime_type=mime_type,
            include_chunks=include_chunks,
            ocr_strategy=ocr_strategy,
            ocr_backend=ocr_backend,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
):
    """Fast document identification — returns document type and suggested filename.

    Renders page 1 as an image and asks the VLM to identify the document type.
    Falls back to rule-based classification if VLM is unavailable.
    No full extraction is performed — this is designed to be fast for sorting/renaming.
    """
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()

    max_upload_bytes = _max_upload_bytes()
    if len(file_bytes) > max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large ({len(file_bytes)} bytes)")

    # Try to get a page image for VLM identification
    page_image = _render_first_page_image(file_bytes, filename, mime_type)

    # Try VLM identification first (most accurate)
    vlm_result = None
    if page_image is not None:
        try:
            from app.llm.config import get_vlm_client
            vlm_client = get_vlm_client()
            if vlm_client:
                vlm_result = vlm_client.vision_extract_json(
                    "You are a document identification expert. Identify the document type from this image.\n"
                    "Return ONLY a JSON object with:\n"
                    '- "doc_type": specific document type (e.g. "W-2", "1099-NEC", "1040", "1040-SR", '
                    '"Schedule A", "Schedule C", "Balance Sheet", "Invoice", "Receipt", "Contract", '
                    '"Bank Statement", "Pay Stub", "Letter", etc.)\n'
                    '- "tax_year": tax year if visible (integer or null)\n'
                    '- "entity_name": primary person/company name if visible (string or null)\n'
                    '- "confidence": float 0-1\n'
                    '- "suggested_filename": a clean descriptive filename like '
                    '"2024-W2-Acme-Corp.pdf" or "2024-1040-Schedule-C.pdf" or "Receipt-2024-03-14.pdf"',
                    "Identify this document.",
                    page_image,
                )
        except Exception:
            pass

    # Parse VLM result
    if vlm_result and vlm_result.get("doc_type"):
        doc_type_raw = str(vlm_result.get("doc_type", "unknown"))
        tax_year = vlm_result.get("tax_year")
        entity_name = vlm_result.get("entity_name")
        confidence = float(vlm_result.get("confidence", 0.0))
        suggested_filename = vlm_result.get("suggested_filename")
        strategy = "vlm"
    else:
        # Fallback: quick extraction + rule-based classification
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            payload = extract_file(tmp_path, filename, mime_type, ocr_strategy="auto")
            from app.classification.rules import classify_document as _classify
            result = _classify(filename=filename, mime_type=mime_type, raw_text=payload.raw_text)
            doc_type_raw = result.doc_type
            confidence = result.confidence
            strategy = result.strategy
            tax_year = None
            entity_name = None
            suggested_filename = None
        except Exception:
            doc_type_raw = "unknown"
            confidence = 0.0
            strategy = "error"
            tax_year = None
            entity_name = None
            suggested_filename = None
        finally:
            tmp_path.unlink(missing_ok=True)

    # Generate suggested filename if VLM didn't provide one
    if not suggested_filename:
        ext = Path(filename).suffix or ".pdf"
        parts = []
        if tax_year:
            parts.append(str(tax_year))
        parts.append(str(doc_type_raw).replace(" ", "-"))
        if entity_name:
            clean_name = "".join(c if c.isalnum() or c in " -" else "" for c in str(entity_name))
            parts.append(clean_name.strip().replace(" ", "-")[:40])
        suggested_filename = "-".join(parts) + ext if parts else filename

    return {
        "original_filename": filename,
        "doc_type": doc_type_raw,
        "tax_year": tax_year,
        "entity_name": entity_name,
        "confidence": round(confidence, 4),
        "strategy": strategy,
        "suggested_filename": suggested_filename,
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "loci-extract"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return detect_capabilities()


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    include_chunks: bool = Form(True),
    ocr_strategy: str | None = Form(None),
    ocr_backend: str | None = Form(None),
    extraction_profile: str | None = Form(None),
):
    profile, resolved_ocr_strategy, resolved_ocr_backend, _, _ = _resolve_profile_params(
        extraction_profile,
        ocr_strategy=ocr_strategy,
        ocr_backend=ocr_backend,
        mask_pii=None,
        enable_llm_enrichment=None,
    )
    _validate_ocr_strategy(resolved_ocr_strategy)
    _validate_ocr_backend(resolved_ocr_backend)
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()
    payload = _extract_payload_from_upload(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        include_chunks=include_chunks,
        ocr_strategy=resolved_ocr_strategy,
        ocr_backend=resolved_ocr_backend,
    )
    if extraction_profile:
        payload.extra.setdefault("extraction_profile", extraction_profile)
    return payload.model_dump()


@app.post("/extract/structured")
async def extract_structured(
    file: UploadFile = File(...),
    include_chunks: bool = Form(True),
    ocr_strategy: str | None = Form(None),
    ocr_backend: str | None = Form(None),
    doc_type_hint: str | None = Form(None),
    mask_pii: bool | None = Form(None),
    enable_llm_enrichment: bool | None = Form(None),
    extraction_profile: str | None = Form(None),
):
    profile, resolved_ocr_strategy, resolved_ocr_backend, resolved_mask_pii, resolved_llm = (
        _resolve_profile_params(
            extraction_profile,
            ocr_strategy=ocr_strategy,
            ocr_backend=ocr_backend,
            mask_pii=mask_pii,
            enable_llm_enrichment=enable_llm_enrichment,
        )
    )
    _validate_ocr_strategy(resolved_ocr_strategy)
    _validate_ocr_backend(resolved_ocr_backend)
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()
    raw_payload = _extract_payload_from_upload(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        include_chunks=include_chunks,
        ocr_strategy=resolved_ocr_strategy,
        ocr_backend=resolved_ocr_backend,
    )
    # Render page image for GPU-based classification (layout or donut)
    page_image = None
    needs_page_image = profile is not None and profile.classifier.strategy in ("layout", "donut-irs", "auto")
    if needs_page_image or cuda_available()["available"]:
        page_image = _render_first_page_image(file_bytes, filename, mime_type)

    classification = classify_with_profile(
        profile=profile,
        filename=filename,
        mime_type=mime_type,
        raw_text=raw_payload.raw_text,
        doc_type_hint=doc_type_hint,
        page_image=page_image,
    )
    structured = build_structured_document(
        classification, raw_payload, mask_pii=resolved_mask_pii, enable_llm_enrichment=resolved_llm
    )
    response = StructuredExtractionResponse(
        document_id=raw_payload.document_id,
        classification=classification,
        raw_extraction=raw_payload,
        structured=structured,
        extra={
            "mask_pii": resolved_mask_pii,
            "enable_llm_enrichment": resolved_llm,
            "extraction_profile": extraction_profile,
        },
    )
    return response.model_dump()


COMPARE_PIPELINE_SPECS: dict[str, tuple[str, str]] = {
    "parser": ("never", "auto"),
    "ocr_tesseract": ("always", "tesseract"),
    "ocr_paddle": ("always", "paddleocr"),
    "force_image_tesseract": ("force_image", "tesseract"),
    "force_image_paddle": ("force_image", "paddleocr"),
    "vlm_pure": ("vlm", "auto"),
    "vlm_hybrid": ("vlm_hybrid", "auto"),
}

_COMPARE_PREVIEW_BYTES = 8192


@app.post("/extract/compare")
async def extract_compare(
    file: UploadFile = File(...),
    pipelines: List[str] = Form(...),
    doc_type_hint: str | None = Form(None),
):
    """Run a single upload through multiple extraction pipelines for side-by-side comparison.

    Each pipeline name maps to a preset (ocr_strategy, ocr_backend) pair. Pipelines not
    available on this host are marked unavailable in the response but never raise.
    """
    del doc_type_hint  # reserved for future per-pipeline classification hinting
    if not pipelines:
        raise HTTPException(status_code=400, detail="At least one pipeline name is required.")

    unknown = [p for p in pipelines if p not in COMPARE_PIPELINE_SPECS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pipeline(s): {unknown}. Valid: {sorted(COMPARE_PIPELINE_SPECS)}",
        )

    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()

    available = set(detect_compare_pipelines()["available_pipelines"])  # type: ignore[arg-type]

    tmp_path = _save_upload_tempfile(file_bytes, filename)
    results: dict[str, dict[str, object]] = {}
    try:
        # De-duplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for name in pipelines:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        for name in ordered:
            if name not in available:
                results[name] = {
                    "ok": False,
                    "error": f"pipeline '{name}' is not available on this host",
                    "available": False,
                }
                continue

            strategy, backend = COMPARE_PIPELINE_SPECS[name]
            start = time.perf_counter()
            try:
                payload = _extract_from_path(
                    tmp_path=tmp_path,
                    filename=filename,
                    mime_type=mime_type,
                    include_chunks=False,
                    ocr_strategy=strategy,
                    ocr_backend=backend,
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                raw_text = payload.raw_text or ""
                preview = raw_text[:_COMPARE_PREVIEW_BYTES]
                truncated = len(raw_text) > _COMPARE_PREVIEW_BYTES
                result: dict[str, object] = {
                    "ok": True,
                    "available": True,
                    "pipeline": name,
                    "ocr_strategy": strategy,
                    "ocr_backend": backend,
                    "elapsed_ms": elapsed_ms,
                    "raw_text_len": len(raw_text),
                    "raw_text_preview": preview,
                    "raw_text_truncated": truncated,
                    "extra": payload.extra,
                    "warnings": [w.model_dump() for w in payload.extraction.warnings],
                    "status": payload.extraction.status,
                    "segment_count": len(payload.segments),
                }
                try:
                    from app.classification.rules import classify_document as _classify
                    classification = _classify(
                        filename=filename,
                        mime_type=mime_type,
                        raw_text=raw_text,
                    )
                    result["classification"] = classification.model_dump()
                except Exception:
                    pass
                results[name] = result
            except HTTPException:
                raise
            except Exception as exc:
                results[name] = {
                    "ok": False,
                    "available": True,
                    "elapsed_ms": int((time.perf_counter() - start) * 1000),
                    "error": f"{type(exc).__name__}: {exc}",
                }
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": len(file_bytes),
        "pipelines": ordered,
        "results": results,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }
