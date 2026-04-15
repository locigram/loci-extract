# loci-extract

**Local-first tax document extraction** — turn W-2s, 1099s, 1098s, K-1s,
SSA-1099, and RRB-1099 PDFs into validated JSON, CSV, Lacerte, or TXF
import files, using a local LLM of your choice. Nothing leaves the machine.

Three surfaces, one library:

- `loci-extract` — CLI tool for one-off and batch extraction
- `loci-extract-api` — OpenAPI-compatible HTTP service (FastAPI)
- `loci_extract.*` — importable Python library for embedding in your own code

All three share the same `core.extract_document()` function — same behavior,
same schemas, same output formats.

---

## Install

```bash
pip install -e .                # core + CLI (tesseract-only OCR)
pip install -e .[ocr]           # adds EasyOCR + PaddleOCR + torch
pip install -e .[api]           # adds FastAPI + uvicorn for the HTTP service
pip install -e .[web]           # same as [api]; HTML UI is always served when the package is installed
pip install -e .[full]          # everything
pip install -e .[dev]           # + pytest + ruff + httpx (for tests)
```

System deps: `tesseract-ocr`, `poppler-utils`.

```bash
sudo apt install tesseract-ocr poppler-utils    # Ubuntu/Debian
brew install tesseract poppler                   # macOS
```

---

## Quick start

Point at a local OpenAI-compatible LLM endpoint (llama.cpp, Ollama, vLLM,
LM Studio, etc.) and extract:

```bash
# Single PDF, JSON to stdout
loci-extract 25-W2.pdf \
  --model http://localhost:11434/v1 --model-name qwen2.5:32b

# Batch directory, CSV to file
loci-extract --batch ~/tax-pdfs --format csv -o all.csv \
  --model http://localhost:8000/v1 --model-name qwen-vl-local

# Lacerte tab-delim import (W-2 / 1099-NEC / 1099-INT / 1099-DIV / 1099-R)
loci-extract 25-W2.pdf --format lacerte -o import.txt

# TXF v42 for TurboTax / TaxAct / UltraTax
loci-extract 25-W2.pdf --format txf -o import.txf

# Vision mode: send page images to a multimodal model (VLM)
loci-extract bad_scan.pdf --vision --vision-model qwen3-vl-32b \
  --model http://localhost:9020/v1
```

Env-var defaults let you omit `--model` and `--model-name`:

```bash
export LOCI_EXTRACT_MODEL_URL=http://10.10.100.20:9020/v1
export LOCI_EXTRACT_MODEL_NAME=qwen3-vl-32b
export LOCI_EXTRACT_VISION_MODEL=qwen3-vl-32b

loci-extract 25-W2.pdf --verbose
```

---

## HTTP API

Run the FastAPI service:

```bash
loci-extract-api --host 127.0.0.1 --port 8080
# UI at /, Swagger at /docs
```

Endpoints:

| Method | Path               | Purpose                                                |
|--------|--------------------|--------------------------------------------------------|
| GET    | `/healthz`         | Liveness check                                         |
| GET    | `/capabilities`    | OCR engines available, LLM config, auth status        |
| POST   | `/extract`         | One PDF → Extraction JSON (or csv/lacerte/txf)        |
| POST   | `/extract/batch`   | Many PDFs → per-file results                          |
| GET    | `/docs`            | Swagger UI                                             |
| GET    | `/`                | Drop-zone web UI                                       |

### Auth

If `LOCI_EXTRACT_API_KEY` is set, every non-health endpoint requires
`Authorization: Bearer <key>`. Leave unset for open local access.

### Example

```bash
curl -F "file=@25-W2.pdf" -F "format=json" http://localhost:8080/extract
curl -F "file=@25-W2.pdf" -F "format=csv"  http://localhost:8080/extract -o w2.csv
```

---

## Library

```python
from loci_extract import extract_document, ExtractionOptions

opts = ExtractionOptions(
    model_url="http://localhost:11434/v1",
    model_name="qwen2.5:32b",
)
extraction = extract_document("25-W2.pdf", opts)
for doc in extraction.documents:
    print(doc.document_type, doc.tax_year)
```

`extraction.model_dump()` returns a plain dict. `extraction.validate_all()`
runs every per-doc-type model's validator.

---

## Supported document types

W-2, 1099-NEC, 1099-MISC, 1099-INT, 1099-DIV, 1099-B, 1099-R, 1099-G,
1099-SA, 1099-K, 1099-S, 1099-C, 1099-A, 1098, 1098-T, 1098-E, SSA-1099,
RRB-1099, K-1 (1065 / 1120-S / 1041).

Model size recommendations:

| Doc complexity                 | Minimum | Recommended |
|--------------------------------|---------|-------------|
| Simple W-2, 1099-NEC/INT/DIV   | 7B      | 14B         |
| Multi-state W-2, complex 1099  | 14B     | 32B         |
| 1099-B with many transactions  | 32B     | 72B         |
| K-1 (any variant)              | 32B     | 72B         |

A 32B multimodal model (e.g. Qwen3-VL 32B) handles the common cases. K-1s
with many coded sub-line entries benefit from 72B.

---

## Output formats

- **json** — pretty-printed `Extraction` pydantic dump
- **csv** — one row per document; nested fields (box12, state, transactions)
  serialized as JSON inside their cells
- **lacerte** — tab-delimited Lacerte import (W-2, 1099-NEC, 1099-INT,
  1099-DIV, 1099-R); SSN masked as `XXXXX####`, user fills first 5 digits
  manually before import
- **txf** — TXF v42 for TurboTax / TaxAct / UltraTax (W-2 + 1099-INT/DIV/R)

---

## SSN / TIN redaction

Any full 9-digit SSN the model emits is rewritten to `XXX-XX-1234`.
Employer/payer EINs are NOT redacted — they're public information and
needed for downstream tax-prep imports. Disable with `--no-redact`.

The **input** to the model always contains the full document text; the
redaction happens on output. Keep the model local to keep SSNs local.

---

## Pipeline

```
PDF → detector.py      (pdfminer: text-layer pages vs image pages)
    → extractor.py     (pdfminer for text pages)
    + ocr.py           (tesseract | easyocr | paddleocr for image pages)
    + vision.py        (--vision: VLM direct read of image pages)
    → concatenated text with "--- PAGE N ---" separators
    → prompts.py       (SYSTEM_PROMPT + per-doc hints from detector)
    → llm.py           (OpenAI-compat client, JSON validate + retry)
    → schema.py        (Extraction pydantic validation)
    → core.py          (W-2 dedup across Copy B/C/2)
    → formatters/      (json | csv | lacerte | txf)
```

---

## Project layout

```
loci_extract/
  core.py            # extract_document(), extract_batch()
  schema.py          # pydantic v2 models for every doc type
  prompts.py         # SYSTEM_PROMPT, PER_DOC_HINTS, Box 12 reference
  detector.py        # text-vs-image per page + doc-type keyword hints
  extractor.py       # pdfminer.six with form-tuned LAParams
  ocr.py             # engine abstraction (tesseract / easyocr / paddleocr)
  vision.py          # VLM path (image_url via OpenAI-compat)
  llm.py             # OpenAI-compat client + retry + SSN redaction
  cli.py             # argparse entry point (loci-extract)
  api/server.py      # FastAPI (loci-extract-api)
  webapp/static/     # drop-zone UI (served by the API)
  formatters/        # json / csv / lacerte / txf
tests/               # pytest suite, 40+ tests, stubbed LLM
```

---

## Development

```bash
pip install -e .[dev,api,ocr]
pytest -q              # 40+ tests, no network
ruff check .           # lint

# Against a live LLM:
loci-extract /path/to/w2.pdf --model http://localhost:11434/v1 --verbose
```

---

## Legacy

The previous FastAPI web service (profiles, Donut IRS classifier, VLM
two-pass, compare-pipelines UI) is preserved on the `legacy-fastapi-v1`
branch on origin. The current CLI / API / library architecture supersedes
it.
