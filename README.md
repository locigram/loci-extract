# loci-extract

Standalone document extraction and OCR-oriented ingestion service for ephemeral processing pipelines.

`loci-extract` accepts common business attachments such as PDFs, Word docs, spreadsheets, and images, routes them through format-specific extractors, and returns a normalized JSON payload with:

- canonical raw extracted text
- page / sheet / paragraph segments
- extraction warnings and provenance
- optional chunked text for RAG pipelines

The service is designed to be reusable across systems, including but not limited to Locigram.

## Current capabilities

Supported file types today:

- PDF via **PyMuPDF**
- DOCX via **python-docx**
- XLSX via **openpyxl**
- PNG / JPG / JPEG / TIFF / WEBP via **Tesseract**
- TXT / MD / CSV / JSON as plain text

Current status:

- parser-first extraction is implemented
- image OCR is implemented through Tesseract when the `tesseract` binary is available
- PDF text extraction is implemented through PyMuPDF
- scanned-PDF OCR fallback is implemented through a conditional Tesseract path when no PDF text layer is present
- when OCR dependencies are missing, the API returns structured warnings instead of crashing
- `/capabilities` reports which OCR/PDF backends are currently available on the host

## Why this exists

`loci-extract` separates **document extraction** from downstream systems.

That means one service can:

1. ingest a file
2. extract canonical text and segments
3. optionally derive chunks
4. hand the result to any caller

This keeps downstream systems focused on search, storage, classification, or knowledge ingestion instead of file parsing.

## Output model

Every extractor returns the same top-level shape:

```json
{
  "document_id": "uuid",
  "metadata": {
    "filename": "contract.pdf",
    "mime_type": "application/pdf",
    "source_type": "pdf",
    "page_count": 4,
    "sheet_names": [],
    "language": null
  },
  "extraction": {
    "extractor": "pymupdf",
    "ocr_used": false,
    "status": "success",
    "warnings": []
  },
  "raw_text": "full extracted text",
  "segments": [],
  "chunks": [],
  "extra": {}
}
```

Key idea:

- `raw_text` is the canonical extraction output
- `segments` preserve local structure
- `chunks` are derived artifacts for retrieval pipelines

## Installation

### 1. Create a virtualenv

```bash
cd ~/projects/loci-extract
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
pip install -e .[dev]
```

### 3. Install OCR dependencies for image extraction

`pytesseract` requires the `tesseract` binary to be available on the system.

Ubuntu/Debian example:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Optional future PDF OCR path may also use tools like:

- `ocrmypdf`
- `tesseract-ocr`
- `ghostscript`

Those are not required for the current scaffold.

## Running the service

### Development server

```bash
cd ~/projects/loci-extract
source .venv/bin/activate
uvicorn app.main:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## API usage

### Health check

```bash
curl http://127.0.0.1:8000/healthz
```

Expected response:

```json
{"status":"ok","service":"loci-extract"}
```

### Capability check

```bash
curl http://127.0.0.1:8000/capabilities
```

Use this endpoint to see whether the current machine has OCR/PDF helper binaries available.

### Extract a text file

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@./sample.txt" \
  -F "include_chunks=true" \
  -F "ocr_strategy=auto"
```

### Extract a PDF

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@./contract.pdf" \
  -F "include_chunks=true" \
  -F "ocr_strategy=auto"
```

### Extract an image with OCR

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@./scan.jpg" \
  -F "include_chunks=false" \
  -F "ocr_strategy=always"
```

## Request fields

### `file`
Multipart uploaded document.

### `include_chunks`
- `true` (default): include derived chunks
- `false`: return only canonical extraction outputs

### `ocr_strategy`
Allowed values:

- `auto` — use OCR when appropriate
- `always` — force OCR intent where supported
- `never` — disable OCR intent

This setting is recorded in the response payload. For PDFs and images, behavior depends on whether OCR backends are available on the host. Check `/capabilities` to see what is currently installed.

## Running tests

```bash
cd ~/projects/loci-extract
source .venv/bin/activate
pytest -q
```

## Local model workflow

Core extraction should stay deterministic and parser-first.

Local OpenAI-compatible models are intended for later optional enrichment tasks such as:

- section labeling
- metadata extraction
- document classification
- chunk enrichment
- review and audit passes

See:

- `docs/local-models.md`
- `docs/review-workflow.md`
- `docs/architecture.md`

## Roadmap

Near-term next steps:

- scanned-PDF OCR fallback
- image preprocessing before OCR
- table-to-text normalization improvements
- configurable extraction profiles
- async job mode for slow OCR workloads
- optional local-LLM enrichers
