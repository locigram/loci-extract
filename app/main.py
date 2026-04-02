from __future__ import annotations

import mimetypes
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.router import UnsupportedDocumentError, extract_file

app = FastAPI(title="loci-extract", version="0.1.0")
VALID_OCR_STRATEGIES = {"auto", "always", "never"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "loci-extract"}


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
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
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
        return payload.model_dump()
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
