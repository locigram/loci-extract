# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

`loci-extract` (package `loci_extract`) is a **local-first tax document
extraction tool** with three surfaces sharing one library:

- **CLI** — `loci-extract` (argparse; `loci_extract/cli.py`)
- **API** — `loci-extract-api` (FastAPI; `loci_extract/api/server.py`)
- **Library** — `from loci_extract import extract_document, ExtractionOptions`

All three call the same `core.extract_document(pdf_path, opts) -> Extraction`.

Prior art: the `legacy-fastapi-v1` branch on origin has the old
multi-document FastAPI service + VLM compare/trace work. **Not imported
or used by the current tree** — pulled only as reference when writing the
new modules.

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,api]       # or [dev,api,ocr] for easyocr/paddleocr

pytest -q                       # all tests run offline with stubbed LLM
ruff check .                    # clean as of this writing

loci-extract 25-W2.pdf --verbose
loci-extract-api --port 8080
```

Env vars (defaults used when CLI/API flags not set):

- `LOCI_EXTRACT_MODEL_URL` (default `http://10.10.100.20:9020/v1`)
- `LOCI_EXTRACT_MODEL_NAME` (default `qwen3-vl-32b`)
- `LOCI_EXTRACT_VISION_MODEL` (default matches `MODEL_NAME`)
- `LOCI_EXTRACT_API_KEY` (if set, API server gates everything except `/healthz`)
- `LOCI_EXTRACT_MAX_UPLOAD_BYTES` (API server; default 50 MB)

System deps: `tesseract-ocr`, `poppler-utils`.

## Architecture

Library-first. The flow is:

```
PDF → detector.detect_page_types()          text/image per page
    + detector.identify_doc_types()          keyword doc-type hints
    → extractor.extract_text_pages()         pdfminer for text pages
    + ocr.extract_pages() or                 tesseract/easyocr/paddleocr
      vision.vision_extract_pages()          VLM (image_url OpenAI-compat)
    → concatenate with "--- PAGE N ---"
    → llm.parse_extraction()                 OpenAI client + retry + redact
    → core._dedup_documents()                collapse Copy B/C/2 W-2 repeats
    → Extraction (validated)
```

`SYSTEM_PROMPT` lives in `prompts.py` and covers:
- output schema for the `{documents: [...]}` wrapper
- per-doc-type `data` payloads
- dedup + redact rules
- Box 12 standard-code reference
- non-standard code handling (DI / FLI / UI-WF-SWF)

### Schema discriminator pattern

`schema.py` uses a `Document.document_type` string discriminator with
`Document.data: dict[str, Any]`. Per-type validation happens via
`Document.validated_data()`, which looks up `DATA_MODEL_BY_TYPE[document_type]`
and runs `model.model_validate(self.data)`. This keeps the prompt schema
flat (one big JSON) without requiring the LLM to emit a discriminated
union. `Extraction.validate_all()` runs validation across every document.

### SSN/TIN redaction

`llm.redact_ssn_in_output()` walks the dict tree and masks `\d{3}-\d{2}-\d{4}`
to `XXX-XX-<last4>`. Applied **after** schema validation so the model still
sees the full source text. EIN format `\d{2}-\d{7}` is NOT matched — EINs
are public. Disable via `ExtractionOptions.redact=False` (`--no-redact` on
the CLI).

### Dedup

Belt-and-suspenders. SYSTEM_PROMPT tells the model to emit one record per
(employee, employer, tax_year); `core._dedup_documents()` also removes
duplicate W-2 records by `(employer_ein|name, employee_ssn_last4|name, tax_year)`.
Only W-2 is deduplicated currently — other doc types don't have Copy B/C/2
multiplicity.

### OCR engine selection

`ocr.select_engine(engine, gpu)` auto-picks:
- `auto` + CUDA/MPS + easyocr available → easyocr
- else + tesseract available → tesseract
- else + paddleocr available → paddleocr
- else raises

Explicit engine falls back to auto if unavailable. EasyOCR and PaddleOCR
imports are inside try/except so a base install (tesseract only) works.

### Formatters

`formatters/` — one module per format (`json_fmt`, `csv_fmt`, `lacerte_fmt`,
`txf_fmt`). Dispatch via `formatters.format_extraction(extraction, fmt)`.
Lacerte and TXF are v1-partial: W-2 + 1099-NEC/INT/DIV/R only — unsupported
types raise `NotImplementedError` rather than emit silent garbage.

## Testing

- Everything is stubbed — no network, no OCR binaries, no real PDFs needed.
- `tests/conftest.py::StubLlmClient` is a canned-response fake used by
  `test_llm_parse.py`, `test_core.py`, `test_api.py`.
- Real PDF/LLM end-to-end testing is manual: `loci-extract path/to/w2.pdf
  --model http://10.10.100.20:9020/v1 --verbose`.

## Local LLM backend (surugpu)

Qwen3-VL 32B GGUF is on surugpu (10.10.100.20) at `/home/ale/models/qwen3-vl-32b/`.
To start the server:

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/ale/llama.cpp/build/bin/llama-server \
  -m /home/ale/models/qwen3-vl-32b/Qwen3-VL-32B-Instruct-Q4_K_M.gguf \
  --mmproj /home/ale/models/qwen3-vl-32b/mmproj-F16.gguf \
  --host 0.0.0.0 --port 9020 \
  --ctx-size 16384 --n-gpu-layers 999 \
  --flash-attn on --cont-batching --jinja \
  --alias qwen3-vl-32b \
  > /home/ale/logs/llama-qwen3-vl-32b.log 2>&1 &
```

See the auto-memory file `deploy_surugpu.md` for SSH access, GPU layout,
and how to redeploy. Ollama and the older loci-extract Docker container
are no longer used.

## Common pitfalls

- **FastAPI `File(...)` defaults.** Ruff B008 flags these. Ignored for
  `loci_extract/api/*.py` via `pyproject.toml` `per-file-ignores` — don't
  refactor away from the idiomatic FastAPI pattern.
- **Lacerte/TXF missing doc types.** Raise `NotImplementedError` with a
  clear message; don't silently emit blank fields. Callers should fall
  back to JSON/CSV for unsupported types.
- **pdfminer LAParams.** Tuned for W-2 grid layouts (`line_margin=0.3,
  char_margin=2.0`). 1099s and K-1s mostly work with the same settings,
  but if field accuracy regresses, re-tune per doc type rather than changing
  the global default.
- **Vision-path latency.** Qwen3-VL 32B at 300 DPI can run 30-90s per
  page. Keep it off for bulk batches unless scans are genuinely bad.
- **SSN redaction is output-only.** The model always sees the full text;
  don't try to redact pre-LLM — the model needs the SSN to know who to
  attach to which W-2.
