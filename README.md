# loci-extract

**Local-first tax + financial document extraction** — turn W-2s, 1099s, 1098s,
K-1s, SSA-1099, RRB-1099 **plus balance sheets, income statements, trial
balances, AR/AP aging, budget-vs-actual, reserve allocations, and general
ledgers** into validated JSON, CSV, Lacerte, or TXF output, using a local
LLM of your choice. Nothing leaves the machine.

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

System deps: `tesseract-ocr`, `poppler-utils` (the `pdffonts` binary in poppler
is used for the encoding-broken PDF detector — install it if you work with
AppFolio / Yardi / QB Desktop print-to-PDF exports).

```bash
sudo apt install tesseract-ocr poppler-utils    # Ubuntu/Debian
brew install tesseract poppler                   # macOS
```

---

## Quick start

Point at a local OpenAI-compatible LLM endpoint (llama.cpp, Ollama, vLLM,
LM Studio, etc.) and extract:

```bash
# Tax: single W-2 PDF → JSON on stdout
loci-extract 25-W2.pdf \
  --model http://localhost:11434/v1 --model-name qwen2.5:32b

# Financial: AppFolio balance sheet → JSON (pipeline auto-detects family)
loci-extract balance-sheet.pdf --verbose

# Batch directory → CSV
loci-extract --batch ~/tax-pdfs --format csv -o all.csv

# Lacerte tab-delim import (tax docs only: W-2 + 1099-NEC/INT/DIV/R)
loci-extract 25-W2.pdf --format lacerte -o import.txt

# TXF v42 for TurboTax / TaxAct / UltraTax
loci-extract 25-W2.pdf --format txf -o import.txf

# Vision mode: send page images directly to a multimodal model (VLM)
loci-extract bad_scan.pdf --vision --vision-model qwen3-vl-32b \
  --model http://localhost:9020/v1

# Force a specific document family (override detector)
loci-extract statement.pdf --family financial_simple

# Skip Python-side totals verification
loci-extract statement.pdf --no-verify-totals
```

Env-var defaults let you omit `--model` and `--model-name`:

```bash
export LOCI_EXTRACT_MODEL_URL=http://10.10.100.20:9020/v1
export LOCI_EXTRACT_MODEL_NAME=qwen3-vl-32b
export LOCI_EXTRACT_VISION_MODEL=qwen3-vl-32b

loci-extract 25-W2.pdf --verbose
```

For LLM endpoints that require authentication:

```bash
export LOCI_EXTRACT_API_KEY_LLM=your-bearer-token
loci-extract 25-W2.pdf --model http://10.10.100.80:30911/v1

# or inline:
loci-extract 25-W2.pdf --model http://10.10.100.80:30911/v1 \
  --api-key your-bearer-token
```

Additional CLI flags:

| Flag | Purpose |
|---|---|
| `--detect-only` | Detect document type and exit without calling the LLM (fast — regex/heuristics only) |
| `--api-key <token>` | Bearer token for authenticated LLM endpoints (default: `$LOCI_EXTRACT_API_KEY_LLM` or `local`) |
| `--no-fix-orientation` | Disable Tesseract OSD rotation correction (default: enabled — catches scanned-sideways pages) |
| `--chunk-size 6000` | Max input tokens per LLM chunk for long GL / statement exports |
| `--no-verify-totals` | Skip Python-side totals verification for financial documents |
| `--family {tax,financial_simple,financial_multi,financial_txn,financial_reserve}` | Force family dispatch (overrides detector) |
| `--parallel-chunks N` | Concurrent LLM calls for chunked financial docs (default: 4; 1 = sequential) |
| `--no-redact` | Disable SSN/TIN last-4 masking on output |

---

## Supported document types

### Tax (21 types)

W-2, 1099-NEC, 1099-MISC, 1099-INT, 1099-DIV, 1099-B, 1099-R, 1099-G,
1099-SA, 1099-K, 1099-S, 1099-C, 1099-A, 1098, 1098-T, 1098-E, SSA-1099,
RRB-1099, K-1 (1065 / 1120-S / 1041).

### Financial (9 types)

- **BALANCE_SHEET** — Assets / Liabilities / Equity with nested sections,
  inline section totals, balance-sheet equation verified in Python
- **INCOME_STATEMENT** — Revenue / expenses / operating + net income
- **INCOME_STATEMENT_COMPARISON** — multi-period (12-month / YTD Actual vs
  Budget) with explicit column definitions
- **BUDGET_VS_ACTUAL** — variance reporting
- **TRIAL_BALANCE** — all GL accounts with debit/credit + verification
- **ACCOUNTS_RECEIVABLE_AGING / ACCOUNTS_PAYABLE_AGING** — aging buckets
  (Current / 1-30 / 31-60 / 61-90 / >90)
- **RESERVE_ALLOCATION** — HOA reserve fund by component
- **GENERAL_LEDGER** — transaction-level detail with chunking support for
  long exports (auto-splits by account boundary)

### Model size recommendations

| Doc complexity                                    | Minimum | Recommended |
|---------------------------------------------------|---------|-------------|
| Simple W-2, 1099-NEC/INT/DIV                      | 7B      | 14B         |
| Multi-state W-2, complex 1099                     | 14B     | 32B         |
| 1099-B / GL with many transactions                | 32B     | 72B         |
| K-1 (any variant)                                 | 32B     | 72B         |
| Balance sheet / single-period P&L                 | 7B      | 14B         |
| 12-month comparison P&L (many columns)            | 14B     | 32B         |
| General ledger / multi-property AppFolio batch    | 32B     | 72B         |

A 32B multimodal model (e.g. Qwen3-VL 32B) handles the common cases across
both tax and financial documents.

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
| GET    | `/capabilities`    | OCR engines available, LLM config, auth status         |
| POST   | `/detect`          | One file → document type detection (no LLM, fast)      |
| POST   | `/ocr`             | One file → per-page OCR text (no LLM)                  |
| POST   | `/vision`          | One file → per-page VLM transcription                  |
| POST   | `/verify`          | JSON body → totals verification + derived fields       |
| POST   | `/boundaries`      | One file → multi-section boundary detection             |
| POST   | `/format`          | Re-format Extraction JSON → CSV/Lacerte/TXF (no LLM)  |
| POST   | `/extract`         | One PDF → Extraction JSON (or csv/lacerte/txf)         |
| POST   | `/extract/batch`   | Many PDFs → per-file results                           |
| GET    | `/docs`            | Swagger UI                                             |
| GET    | `/`                | Drop-zone web UI                                       |

### Auth

If `LOCI_EXTRACT_API_KEY` is set, every non-health endpoint requires
`Authorization: Bearer <key>`. Leave unset for open local access.

### Examples

**Extract a W-2 to JSON or CSV:**

```bash
curl -F "file=@25-W2.pdf" -F "format=json" http://localhost:8080/extract
curl -F "file=@balance-sheet.pdf" -F "format=csv" http://localhost:8080/extract -o bs.csv
```

**Detect document type (no LLM, fast):**

```bash
curl -F "file=@mystery.pdf" http://localhost:8080/detect
# → {"document_type": "W2", "document_family": "tax", "confidence": 0.95, ...}
```

**OCR only — get raw per-page text without extraction:**

```bash
curl -F "file=@scan.pdf" http://localhost:8080/ocr
# → {"pages": {"1": "text...", "2": "text..."}, "total_pages": 2, ...}
```

**Vision only — VLM transcription with a specific model:**

```bash
curl -F "file=@scan.pdf" \
  -F "model_url=http://10.10.100.80:30911/v1" \
  -F "model_name=mlx-community/Qwen3.6-35B-A3B-8bit" \
  -F "api_key=<token>" \
  http://localhost:8080/vision
# → {"pages": {"1": "text...", ...}, "model": "...", "total_pages": 9}
```

**Extract with vision mode + CSV output (full pipeline via VLM):**

```bash
curl -F "file=@scan.pdf" -F "vision=true" -F "format=csv" \
  -F "model_url=http://10.10.100.80:30911/v1" \
  -F "api_key=<token>" \
  http://localhost:8080/extract -o result.csv
```

**Detect multi-section boundaries (e.g. BS + P&L in one PDF):**

```bash
curl -F "file=@combined-financials.pdf" http://localhost:8080/boundaries
# → {"sections": [{"start_page": 1, "end_page": 2, "document_type": "BALANCE_SHEET", ...}, ...]}
```

**Verify totals on already-extracted data (no file upload):**

```bash
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{"document_type": "BALANCE_SHEET", "data": {"assets": {...}, ...}}'
# → {"verified": true, "mismatches": [], "balance_sheet_balanced": true, "derived_fields": {...}}
```

**Re-format extraction results (e.g. JSON → CSV without re-extracting):**

```bash
curl -X POST "http://localhost:8080/format?format=csv" \
  -H "Content-Type: application/json" \
  -d @extraction.json -o result.csv
```

See [docs/integrations.md](docs/integrations.md) for client snippets in
Python / Node / Go / PowerShell / curl / Make / n8n / Zapier, OpenAPI
client-code generation, auth setup, and latency/error notes.

---

## Library

```python
from loci_extract import extract_document, ExtractionOptions

opts = ExtractionOptions(
    model_url="http://localhost:11434/v1",
    model_name="qwen2.5:32b",
)
extraction = extract_document("balance-sheet.pdf", opts)
for doc in extraction.documents:
    print(doc.document_type, doc.tax_year)
    # Financial docs: inspect verification + derived fields
    print(doc.metadata.totals_verified, doc.metadata.balance_sheet_balanced)
```

`extraction.model_dump()` returns a plain dict. `extraction.validate_all()`
runs every per-doc-type model's validator.

---

## Output formats

- **json** — pretty-printed `Extraction` pydantic dump with full metadata
  (encoding_broken, pages_rotated, totals_verified, balance_sheet_balanced,
  llm_calls/retries, notes, …)
- **csv** — shape dispatched by document family:
  - **W-2**: flat one-row-per-W-2 layout with every IRS box as its own column
    (employer/employee info, boxes 1–11, box 12a–d code+amount, box 13 Y/N
    checkboxes, box 14 other, state 1–2, local 1–2). No JSON blobs — open
    in Excel and every field is immediately readable for human verification.
  - **Other tax** (1099s, K-1s, etc.): one row per document with parties +
    primary amount.
  - **Financial Shape A** (BalanceSheet, IncomeStatement, MultiColumn, etc.):
    one row per account with dynamic period columns + subtotal + total rows
  - **Financial Shape B** (GeneralLedger): one row per transaction with
    balance_header / balance_footer boundary rows
  - **Financial Shape B-aging**: one row per customer/vendor + totals row
- **lacerte** — tab-delimited Lacerte import (tax only: W-2, 1099-NEC, 1099-INT,
  1099-DIV, 1099-R). Financial types raise `NotImplementedError` with guidance.
- **txf** — TXF v42 for TurboTax / TaxAct / UltraTax (W-2 + 1099-INT/DIV/R).

---

## Pipeline

```
PDF → detector.get_extraction_strategy()     pdffonts / non-printable /
                                             anchor-words / word-density
                                             → text | pdfplumber | ocr | vision
    → detector.detect_tax_document_type()    weighted regex scoring,
      + detect_financial_document_type()     ambiguity resolvers
    → boundary_detector.detect_boundaries()  multi-section split per PDF
    →
      ┌─ tax family ──────────────────────────────────────────────────┐
      │  pdfminer text / OCR / vision → parse_extraction → Extraction  │
      └────────────────────────────────────────────────────────────────┘
      ┌─ financial family ─────────────────────────────────────────────┐
      │  chunker.chunk_for_llm()          3-tier: account boundary      │
      │                                   → page break → fixed          │
      │  → llm.call_llm_raw()             per chunk, finish_reason=     │
      │                                   length auto-bump              │
      │  → core_chunked._merge_chunks()   per-doc-type merge            │
      │  → verifier.verify_section_totals()   Decimal tolerance $0.02   │
      │  → verifier.compute_derived_fields()  retained_earnings, etc.    │
      │  → Extraction with FinancialMetadata                             │
      └──────────────────────────────────────────────────────────────────┘
    → formatters/     json | csv | lacerte | txf
```

### Encoding-broken PDF detection

PScript5/Distiller "print to PDF" workflows (common with AppFolio, Yardi,
Sage 100/300, some QuickBooks Desktop) produce a non-empty text layer made
of glyph IDs with no ToUnicode map — pdfminer extracts long strings of
garbage instead of failing visibly. The detector runs `pdffonts` and flags
Identity-H CID encoding with `uni=no` as encoding-broken, routing the PDF
to OCR (or `--vision`) automatically.

### Rotation correction

Scanned-sideways or 180°-rotated pages (common in printed reports) pass
through `pytesseract.image_to_osd()` and are rotated before OCR. Enabled
by default; disable with `--no-fix-orientation`.

### Python-side derived fields and totals verification

The LLM extracts values verbatim; Python computes everything that requires
arithmetic. Each financial document is verified against its declared
section totals and (for balance sheets) the `Assets = Liabilities + Equity`
equation with a `Decimal('0.02')` tolerance for OCR rounding. Derived
fields (`retained_earnings_calculated`, `net_income_calculated`,
`total_reserve_balance_calculated`) are computed post-LLM. Results land in
`FinancialMetadata.totals_verified`, `.totals_mismatches`, and
`.balance_sheet_balanced`.

---

## SSN / TIN redaction

Any full 9-digit SSN the model emits is rewritten to `XXX-XX-1234`.
Employer/payer EINs are NOT redacted — they're public information and
needed for downstream tax-prep imports. Disable with `--no-redact`.

The **input** to the model always contains the full document text; the
redaction happens on output. Keep the model local to keep SSNs local.

---

## Project layout

```
loci_extract/
  core.py               # extract_document() — orchestrates the whole pipeline
  core_chunked.py       # financial path: chunk → multi-LLM → merge → verify
  schema.py             # pydantic v2 models for every tax + financial doc type
  prompts.py            # DocumentFamily + 5 family prompts + get_prompt()
  detector.py           # extraction strategy + tax detection + financial detection
                        # + master detect() returning DocumentDetectionResult
  boundary_detector.py  # multi-section PDF splits (Balance Sheet + P&L in one PDF)
  extractor.py          # pdfminer.six + pdfplumber coordinate-aware path
  ocr.py                # tesseract / easyocr / paddleocr + Tesseract OSD rotation
  vision.py             # VLM path (image_url via OpenAI-compat)
  llm.py                # OpenAI-compat client + TOKEN_BUDGETS + finish_reason retry
  chunker.py            # 3-tier text chunking (account boundary / page break / fixed)
  verifier.py           # Decimal totals verifier + derived-field computation
  cli.py                # argparse entry point (loci-extract)
  api/server.py         # FastAPI — /extract, /detect, /ocr, /vision, /verify,
                        # /boundaries, /format, /extract/batch, /healthz, /capabilities
  webapp/static/        # drop-zone UI with endpoint selector + batch CSV export
  formatters/           # json / csv / lacerte / txf (CSV: flat W-2 columns, 3 financial shapes)
tests/                  # 120+ tests, stubbed LLM; guarded live-LLM integration tests
tests/fixtures/         # sanitized real PDF fixtures for integration tests
```

---

## Development

```bash
pip install -e .[dev,api,ocr]
pytest -q              # 90+ tests, no network
ruff check .           # lint

# Against a live LLM:
loci-extract /path/to/w2.pdf --model http://localhost:11434/v1 --verbose

# Integration tests against the configured LLM:
LOCI_EXTRACT_LIVE_LLM=1 pytest tests/test_real_fixtures.py -v
```

---

## Design decisions worth knowing about

- **`section_total` is inline** on every section dict — there is no separate
  `subtotals[]` list anywhere in any schema.
- **`FinancialMetadata` is the one metadata model** used by every document
  type (tax and financial). The LLM populates only `notes[]`; every other
  field is owned by a specific Python pipeline stage (detector, ocr,
  boundary_detector, llm, verifier).
- **LLM extracts values verbatim. Python computes derived fields.** If a
  field requires arithmetic, it is not an LLM output field.
- **Tax extraction is preserved byte-exact** across the financial refactor.
  The existing W-2 / 1099 / K-1 outputs are identical modulo additive
  metadata fields.

See `DESIGN_DECISIONS.md`, `SPEC_PATCH_V3.md`, `TAX_DETECTION_SPEC.md`,
`FINANCIAL_STATEMENTS_SPEC_V2.md`, `FINANCIAL_STATEMENTS_SPEC.md`, and
`EXTRACT_SPEC.md` for the full specification stack.

---

## Legacy

The previous FastAPI web service (profiles, Donut IRS classifier, VLM
two-pass, compare-pipelines UI) is preserved on the `legacy-fastapi-v1`
branch on origin. The current CLI / API / library architecture supersedes
it.
