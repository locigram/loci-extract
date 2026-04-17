"""FastAPI server — HTTP API that wraps ``loci_extract.core``.

Endpoints:
    GET  /healthz
    GET  /capabilities            → OCR engines, GPU, LLM reachability
    POST /detect                  → multipart file → document type detection (no LLM)
    POST /extract                 → multipart file → Extraction JSON (or any format)
    POST /extract/batch           → multipart multiple files → per-file results
    GET  /                        → web UI (served only if [web] extra installed)

Auth: if ``LOCI_EXTRACT_API_KEY`` is set, every non-/healthz endpoint
requires ``Authorization: Bearer <key>``. Otherwise open (local-first).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

try:
    from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, PlainTextResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The API server requires the [api] extra. Install with `pip install loci-extract[api]`."
    ) from exc

from loci_extract import __version__
from loci_extract.core import ExtractionOptions, detect_document, extract_batch, extract_document
from loci_extract.formatters import format_extraction
from loci_extract.ocr import available_engines
from loci_extract.schema import Extraction

logger = logging.getLogger("loci_extract.api")

_DEFAULT_MODEL_URL = os.getenv("LOCI_EXTRACT_MODEL_URL", "http://10.10.100.20:9020/v1")
_DEFAULT_MODEL_NAME = os.getenv("LOCI_EXTRACT_MODEL_NAME", "qwen3-vl-32b")
_DEFAULT_VISION_MODEL = os.getenv("LOCI_EXTRACT_VISION_MODEL", _DEFAULT_MODEL_NAME)
_DEFAULT_LLM_API_KEY = os.getenv("LOCI_EXTRACT_API_KEY_LLM", "local")
_API_KEY = os.getenv("LOCI_EXTRACT_API_KEY", "").strip()
_MAX_UPLOAD_BYTES = int(os.getenv("LOCI_EXTRACT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))


app = FastAPI(
    title="loci-extract",
    version=__version__,
    description="Local-first tax document extraction API.",
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not _API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    if authorization.split(" ", 1)[1].strip() != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_upload(file: UploadFile, tmp_dir: Path) -> Path:
    data = file.file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large ({len(data)} bytes, limit {_MAX_UPLOAD_BYTES})",
        )
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    out = tmp_dir / f"{os.urandom(6).hex()}{suffix}"
    out.write_bytes(data)
    return out


def _options(
    model_url: str | None,
    model_name: str | None,
    ocr_engine: str | None,
    gpu: str | None,
    dpi: int | None,
    vision: bool,
    vision_model: str | None,
    redact: bool,
    temperature: float | None,
    max_tokens: int | None,
    retry: int | None,
    api_key: str | None = None,
) -> ExtractionOptions:
    return ExtractionOptions(
        model_url=model_url or _DEFAULT_MODEL_URL,
        model_name=model_name or _DEFAULT_MODEL_NAME,
        api_key=api_key or _DEFAULT_LLM_API_KEY,
        ocr_engine=ocr_engine or "auto",  # type: ignore[arg-type]
        gpu=gpu or "auto",  # type: ignore[arg-type]
        dpi=dpi or 300,
        vision=vision,
        vision_model=vision_model or _DEFAULT_VISION_MODEL,
        redact=redact,
        temperature=temperature if temperature is not None else 0.0,
        max_tokens=max_tokens or 4096,
        retry=retry if retry is not None else 2,
    )


def _format_response(extraction: Extraction, fmt: str) -> Response:
    try:
        body = format_extraction(extraction, fmt)
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if fmt == "json":
        return Response(content=body, media_type="application/json")
    if fmt == "csv":
        return PlainTextResponse(content=body, media_type="text/csv")
    return PlainTextResponse(content=body, media_type="text/plain")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "loci-extract", "version": __version__}


@app.get("/capabilities", dependencies=[Depends(require_api_key)])
def capabilities() -> dict[str, object]:
    engines = available_engines()
    # Probe the LLM without actually calling it; just whether the client
    # can be constructed against the configured URL.
    llm_info = {
        "model_url": _DEFAULT_MODEL_URL,
        "model_name": _DEFAULT_MODEL_NAME,
        "vision_model": _DEFAULT_VISION_MODEL,
    }
    return {
        "ocr_engines": engines,
        "llm": llm_info,
        "max_upload_bytes": _MAX_UPLOAD_BYTES,
        "auth_required": bool(_API_KEY),
        "version": __version__,
    }


@app.post("/detect", dependencies=[Depends(require_api_key)])
def detect_endpoint(
    file: UploadFile = File(...),
    ocr_engine: str | None = Form(None),
    gpu: str | None = Form(None),
    dpi: int | None = Form(None),
    vision: bool = Form(False),
    vision_model: str | None = Form(None),
) -> dict:
    """Detect document type without calling the LLM. Fast — regex/heuristics only."""
    opts = _options(
        model_url=None, model_name=None, ocr_engine=ocr_engine, gpu=gpu, dpi=dpi,
        vision=vision, vision_model=vision_model, redact=True, temperature=None,
        max_tokens=None, retry=None,
    )
    with tempfile.TemporaryDirectory(prefix="loci-extract-api-") as tmp:
        tmp_path = Path(tmp)
        pdf_path = _save_upload(file, tmp_path)
        try:
            result = detect_document(pdf_path, opts)
        except Exception as exc:
            logger.exception("detection failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return result


@app.post("/extract", dependencies=[Depends(require_api_key)])
def extract(    file: UploadFile = File(...),
    model_url: str | None = Form(None),
    model_name: str | None = Form(None),
    api_key: str | None = Form(None),
    ocr_engine: str | None = Form(None),
    gpu: str | None = Form(None),
    dpi: int | None = Form(None),
    vision: bool = Form(False),
    vision_model: str | None = Form(None),
    redact: bool = Form(True),
    temperature: float | None = Form(None),
    max_tokens: int | None = Form(None),
    retry: int | None = Form(None),
    format: str = Form("json"),
) -> Response:
    opts = _options(
        model_url, model_name, ocr_engine, gpu, dpi, vision, vision_model,
        redact, temperature, max_tokens, retry, api_key=api_key,
    )
    with tempfile.TemporaryDirectory(prefix="loci-extract-api-") as tmp:
        tmp_path = Path(tmp)
        pdf_path = _save_upload(file, tmp_path)
        try:
            extraction = extract_document(pdf_path, opts)
        except Exception as exc:
            logger.exception("extraction failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return _format_response(extraction, format)


@app.post("/extract/batch", dependencies=[Depends(require_api_key)])
def extract_batch_endpoint(    files: list[UploadFile] = File(...),
    model_url: str | None = Form(None),
    model_name: str | None = Form(None),
    api_key: str | None = Form(None),
    ocr_engine: str | None = Form(None),
    gpu: str | None = Form(None),
    dpi: int | None = Form(None),
    vision: bool = Form(False),
    vision_model: str | None = Form(None),
    redact: bool = Form(True),
    temperature: float | None = Form(None),
    max_tokens: int | None = Form(None),
    retry: int | None = Form(None),
) -> dict:
    opts = _options(
        model_url, model_name, ocr_engine, gpu, dpi, vision, vision_model,
        redact, temperature, max_tokens, retry, api_key=api_key,
    )
    with tempfile.TemporaryDirectory(prefix="loci-extract-api-") as tmp:
        tmp_path = Path(tmp)
        saved = [(f.filename or "upload.pdf", _save_upload(f, tmp_path)) for f in files]
        results = extract_batch([p for _, p in saved], opts)
    return {
        "results": [
            {
                "filename": saved[i][0],
                "documents": results[i][1].model_dump().get("documents", []),
            }
            for i in range(len(results))
        ]
    }


# ---------------------------------------------------------------------------
# Web UI (optional)
# ---------------------------------------------------------------------------


_STATIC_DIR = Path(__file__).resolve().parent.parent / "webapp" / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = _STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>loci-extract</h1><p>API running. UI not installed.</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# uvicorn entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="loci-extract-api")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WEB_CONCURRENCY", "1")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    uvicorn.run(
        "loci_extract.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
