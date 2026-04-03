from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.capabilities import detect_capabilities
from app.router import UnsupportedDocumentError, extract_file
from app.schemas import ExtractionPayload

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
    if ocr_strategy not in VALID_OCR_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ocr_strategy '{ocr_strategy}'. Expected one of: auto, always, never.",
        )

    suffix = Path(file.filename or "upload.bin").suffix
    file_bytes = await file.read()
    max_upload_bytes = _max_upload_bytes()
    if len(file_bytes) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Uploaded file is too large ({len(file_bytes)} bytes). "
                f"Maximum allowed size is {max_upload_bytes} bytes."
            ),
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"

    try:
        payload = extract_file(
            tmp_path,
            file.filename or tmp_path.name,
            mime_type,
            ocr_strategy=ocr_strategy,
        )
        if not include_chunks:
            payload.chunks = []
        payload.extra.setdefault("ocr_strategy", ocr_strategy)
        _enrich_payload_extra(payload)
        return payload.model_dump()
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
