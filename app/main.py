from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.capabilities import detect_capabilities
from app.classification.rules import classify_document
from app.router import UnsupportedDocumentError, extract_file
from app.schemas import ExtractionPayload, StructuredExtractionResponse
from app.structured.router import build_structured_document

app = FastAPI(title="loci-extract", version="0.1.0")
VALID_OCR_STRATEGIES = {"auto", "always", "never"}
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


def _validate_ocr_strategy(ocr_strategy: str) -> None:
    if ocr_strategy not in VALID_OCR_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ocr_strategy '{ocr_strategy}'. Expected one of: auto, always, never.",
        )


def _extract_payload_from_upload(
    *,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    include_chunks: bool,
    ocr_strategy: str,
) -> ExtractionPayload:
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
        tmp_path = Path(tmp.name)

    try:
        payload = extract_file(tmp_path, filename or tmp_path.name, mime_type, ocr_strategy=ocr_strategy)
        if not include_chunks:
            payload.chunks = []
        payload.extra.setdefault("ocr_strategy", ocr_strategy)
        _enrich_payload_extra(payload)
        return payload
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


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
    ocr_strategy: str = Form("auto"),
):
    _validate_ocr_strategy(ocr_strategy)
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()
    payload = _extract_payload_from_upload(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        include_chunks=include_chunks,
        ocr_strategy=ocr_strategy,
    )
    return payload.model_dump()


@app.post("/extract/structured")
async def extract_structured(
    file: UploadFile = File(...),
    include_chunks: bool = Form(True),
    ocr_strategy: str = Form("auto"),
    doc_type_hint: str | None = Form(None),
    mask_pii: bool = Form(True),
):
    _validate_ocr_strategy(ocr_strategy)
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = await file.read()
    raw_payload = _extract_payload_from_upload(
        file_bytes=file_bytes,
        filename=filename,
        mime_type=mime_type,
        include_chunks=include_chunks,
        ocr_strategy=ocr_strategy,
    )
    classification = classify_document(
        filename=filename,
        mime_type=mime_type,
        raw_text=raw_payload.raw_text,
        doc_type_hint=doc_type_hint,
    )
    structured = build_structured_document(classification, raw_payload, mask_pii=mask_pii)
    response = StructuredExtractionResponse(
        document_id=raw_payload.document_id,
        classification=classification,
        raw_extraction=raw_payload,
        structured=structured,
        extra={"mask_pii": mask_pii},
    )
    return response.model_dump()
