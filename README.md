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
- DOCX via **python-docx** (paragraphs and tables)
- XLSX via **openpyxl** (sheet summaries plus row-level provenance)
- PNG / JPG / JPEG / TIFF / WEBP via **Tesseract**
- TXT / MD / CSV / JSON as plain text

Current status:

- parser-first extraction is implemented
- image OCR is implemented through Tesseract when the `tesseract` binary is available, with lightweight preprocessing (orientation fix, grayscale, autocontrast, thresholding, upscale)
- PDF text extraction is implemented through PyMuPDF
- scanned-PDF OCR fallback is implemented through a conditional Tesseract path when no PDF text layer is present
- PDF `ocr_strategy=always` is supported, with page-level provenance metadata
- DOCX extraction preserves paragraphs, headings/list-item structure, and tables
- XLSX extraction preserves sheet summaries plus row-level provenance
- when OCR dependencies are missing, the API returns structured warnings instead of crashing
- upload-size and PDF-page guardrails are enforced through configurable limits
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

### How you invoke the tool

`loci-extract` is currently an **HTTP API service**, not a standalone CLI command.

That means the normal flow is:

1. start the FastAPI server
2. send a `POST /extract` request with a file attached as multipart form data
3. receive a JSON response containing the extracted result
4. save that JSON yourself if you want to persist it

At the moment, `loci-extract` does **not** automatically write extraction results into a database, vector store, or local output folder. It processes the uploaded file and returns the extracted payload in the HTTP response.

So the answer to “where does the extracted data go?” is:

- the uploaded source file is written to a temporary file during request handling
- extraction happens against that temporary file
- the temp file is deleted before the request finishes
- the extracted JSON is returned to the caller in the API response
- if you want to keep it, **your caller must save it**

### Fastest way to use it

Start the server:

```bash
cd ~/projects/loci-extract
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then, in another terminal, send a PDF:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@/absolute/path/to/contract.pdf" \
  -F "include_chunks=true" \
  -F "ocr_strategy=auto"
```

If you want to keep the result, redirect it to a file:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@/absolute/path/to/contract.pdf" \
  -F "include_chunks=true" \
  -F "ocr_strategy=auto" \
  > contract.extracted.json
```

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

### Save PDF extraction output to disk

```bash
mkdir -p output
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@./contract.pdf" \
  -F "include_chunks=true" \
  -F "ocr_strategy=auto" \
  | tee output/contract.extracted.json
```

### Inspect just the main extracted text

```bash
curl -s -X POST http://127.0.0.1:8000/extract \
  -F "file=@./contract.pdf" \
  -F "include_chunks=false" \
  -F "ocr_strategy=auto" \
  | jq -r '.raw_text'
```

### Inspect page-level provenance for a PDF

```bash
curl -s -X POST http://127.0.0.1:8000/extract \
  -F "file=@./contract.pdf" \
  -F "include_chunks=true" \
  -F "ocr_strategy=always" \
  | jq '.extra.page_provenance'
```

### Extract a PDF from Python

```python
from pathlib import Path
import requests

pdf_path = Path('/absolute/path/to/contract.pdf')

with pdf_path.open('rb') as fh:
    response = requests.post(
        'http://127.0.0.1:8000/extract',
        files={'file': (pdf_path.name, fh, 'application/pdf')},
        data={
            'include_chunks': 'true',
            'ocr_strategy': 'auto',
        },
        timeout=120,
    )

response.raise_for_status()
payload = response.json()

print(payload['metadata'])
print(payload['extraction'])
print(payload['raw_text'][:1000])

Path('contract.extracted.json').write_text(response.text)
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

How to think about it for PDFs:

- `auto`: use parser text first; if no PDF text layer is found, try OCR if available
- `always`: attempt OCR even when parser text exists; useful for scanned or low-quality PDFs when you want OCR-first behavior and provenance reporting
- `never`: do not attempt OCR; return parser output only, plus warnings if the PDF has no usable text layer

This setting is recorded in the response payload. For PDFs and images, behavior depends on whether OCR backends are available on the host. Check `/capabilities` to see what is currently installed.

## What the returned JSON means

The response payload is designed to support both “show me the extracted text” and “feed this into a downstream system.”

### Top-level fields

- `document_id` — per-request extraction ID
- `metadata` — file-level metadata such as filename, mime type, page count, sheet names
- `extraction` — extractor used, status, whether OCR was used, warnings
- `raw_text` — canonical full extracted text for the document
- `segments` — structurally meaningful pieces of the source (pages, tables, paragraphs, sheets, sections)
- `chunks` — retrieval-oriented derived chunks built from segments
- `extra` — stable operational metadata and provenance hints

### Which field should you actually use?

Use cases usually break down like this:

- want the full extracted text: use `raw_text`
- want structured per-page/per-table/per-paragraph data: use `segments`
- want RAG-ready chunked content: use `chunks`
- want to understand quality/provenance/fallback behavior: use `extraction` + `extra`

### Where the data lives after extraction

By default, nowhere permanent inside `loci-extract` itself.

`loci-extract` is intentionally **ephemeral** right now:

- it accepts an upload
- extracts the content
- returns JSON
- deletes the temp upload file
- leaves persistence to the caller

That means you can plug it into:

- a webhook receiver
- a queue worker
- Locigram ingestion
- a cron job
- a custom Python/Node pipeline
- manual `curl` usage from the shell

If you need persistence, common patterns are:

- save the full response JSON to object storage
- store `raw_text` plus metadata in a database
- store `chunks` in a vector database
- store both raw payload and downstream transformed artifacts

The response `extra` block reports stable extraction metadata such as:

- `segment_count`
- `chunk_count`
- `warning_codes`
- `has_warnings`
- `content_detected`
- `empty_content`
- `partial_reason` — currently `empty_content` for blank/empty partial extractions
- `non_empty_page_count`
- `non_empty_sheet_count`
- `table_segment_count`

For PDFs specifically, `extra` also reports:

- `ocr_attempted` — whether OCR actually ran
- `result_source` — overall result classification: `parser`, `ocr`, `parser_fallback`, or `none`
- `page_limit_applied` — whether a PDF page cap truncated processing
- `processed_page_count` — how many pages were actually processed
- `max_pdf_pages` — the configured processing cap
- `page_provenance` — one entry per processed page with `page_number`, `source`, `has_text`, and `text_length`

Example `page_provenance`:

```json
[
  {"page_number": 1, "source": "ocr", "has_text": true, "text_length": 1842},
  {"page_number": 2, "source": "parser_fallback", "has_text": true, "text_length": 972},
  {"page_number": 3, "source": "none", "has_text": false, "text_length": 0}
]
```

Interpretation:

- `parser` — page text came directly from the PDF text layer
- `ocr` — page text came from OCR output
- `parser_fallback` — OCR was attempted, but this page ultimately used parser text
- `none` — no text was recovered for that processed page

Environment variables for operational guardrails:

- `LOCI_EXTRACT_MAX_UPLOAD_BYTES` — maximum upload size accepted by `/extract`
- `LOCI_EXTRACT_MAX_PDF_PAGES` — maximum number of PDF pages processed per request

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
- `docs/tax-ingestion.md`

## Roadmap

Remaining improvements worth considering:

- image preprocessing before OCR
- richer PDF mixed-mode provenance summaries
- XLSX header inference / logical table grouping
- configurable extraction profiles
- async job mode for slow OCR workloads
- optional local-LLM enrichers
- legacy Office format support if needed (`.doc`, `.xls`)
