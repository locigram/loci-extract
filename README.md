# loci-extract

Standalone document extraction and OCR service for ephemeral ingestion pipelines.

`loci-extract` accepts files like PDF, DOCX, XLSX, JPG, and PNG, routes them through the best available extractor, and returns a normalized payload containing:

- raw extracted text
- page/sheet/section segments
- table content normalized to text
- optional RAG-ready chunks
- extraction provenance and warnings

## Goals

- Reusable across multiple systems, including but not limited to Locigram
- Preserve both canonical raw extraction output and derived chunk output
- Use prebuilt OCR/document tools behind one stable API
- Support ephemeral processing first, with optional async jobs later

## Initial stack

- Python 3.11+
- FastAPI
- Pydantic
- PyMuPDF
- python-docx
- openpyxl
- pandas
- pytesseract
- Pillow
- pdfplumber
- OCRmyPDF (optional system dependency)

## Planned API

### `GET /healthz`
Health check.

### `POST /extract`
Multipart upload endpoint that returns a normalized extraction payload.

Request:
- `file`: attachment to process
- `include_chunks`: boolean, default true
- `ocr_strategy`: `auto | always | never`

Response:
- document metadata
- extraction metadata
- raw text
- structured segments
- chunks

## Development

```bash
cd ~/projects/loci-extract
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
pytest -q
```

## Status

Bootstrap scaffold in progress.
