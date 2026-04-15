# w2extract

A local-first tax document extraction CLI. Pulls structured JSON from W2s, 1099s, 1098s,
K-1s, and other standard IRS forms using local LLMs (Ollama, vLLM) and local OCR (EasyOCR,
PaddleOCR, Tesseract). No data leaves your machine.

---

## Table of Contents

- [What Claude Does vs What You Run Locally](#what-claude-does-vs-what-you-run-locally)
  - [Document Ingestion](#document-ingestion)
  - [OCR](#ocr-imagescanned-documents)
  - [Document Understanding / Field Extraction](#document-understanding--field-extraction)
  - [Vision / VLM Path](#vision--vlm-path-alternative-to-ocr--llm)
  - [Structured Output / JSON Validation](#structured-output--json-validation)
  - [Model Recommendations by Task](#model-recommendations-by-task)
  - [PII Protection](#pii-protection)
- [Install](#install)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Supported Document Types](#supported-document-types)
- [Local Model Setup](#local-model-setup)
- [GPU Acceleration](#gpu-acceleration)
- [Output Formats](#output-formats)
- [JSON Schema Reference](#json-schema-reference)
- [Project Structure](#project-structure)
- [Pipeline Internals](#pipeline-internals)
- [Troubleshooting](#troubleshooting)

---

## What Claude Does vs What You Run Locally

This section maps every capability Claude uses when processing tax documents in
this conversation to its offline equivalent. The goal is full parity — same
pipeline, no PII leaving your machine.

---

### Document Ingestion

| What Claude does | Offline equivalent | Notes |
|---|---|---|
| Reads PDF text layer from uploaded file | `pdfminer.six` — `extract_text(path)` | Handles digitally-generated PDFs (W2 from payroll software, ADP, Paychex) |
| Reads image/scanned PDF | `pdf2image` → `pytesseract` or `easyocr` | Converts pages to PNG at 300dpi, then OCR |
| Views embedded images in PDF | `pdfplumber` — `.pages[n].images` | Extracts inline images for separate OCR pass |
| Sees the whole document at once | Concatenate all page text before LLM call | pdfminer processes page by page; join with `\n---PAGE BREAK---\n` |

```python
# pdfminer basic extraction
from pdfminer.high_level import extract_text
text = extract_text("25-W2.pdf")

# pdf2image + pytesseract for scanned pages
from pdf2image import convert_from_path
import pytesseract
pages = convert_from_path("scan.pdf", dpi=300)
text = "\n".join(pytesseract.image_to_string(p) for p in pages)
```

---

### OCR (Image/Scanned Documents)

| What Claude does | Offline equivalent | GPU | Notes |
|---|---|---|---|
| Reads a scanned W2 image | `pytesseract` | No | CPU only, slowest, most compatible |
| Reads a scanned W2 image | `easyocr` | Yes (CUDA/MPS) | Best accuracy for forms, handles skew |
| Reads a scanned W2 image | `paddleocr` | Yes (CUDA) | Fast, good on dense tabular layouts |
| Reads a photographed/phone-scanned doc | VLM vision path (llava, minicpm-v) | Yes | Better than OCR for low quality scans |
| Understands form layout visually | VLM vision path | Yes | OCR gives raw text; VLM understands boxes |

```python
# easyocr with auto GPU
import easyocr, torch
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
result = reader.readtext("page.png", detail=0)
text = "\n".join(result)

# paddleocr
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
result = ocr.ocr("page.png", cls=True)
text = "\n".join([line[1][0] for line in result[0]])
```

---

### Document Understanding / Field Extraction

This is the core capability. Claude reads unstructured or semi-structured text
and maps it to the correct IRS box numbers with no explicit parsing rules.

| What Claude does | Offline equivalent | Model size needed | Notes |
|---|---|---|---|
| Identifies document type (W2 vs 1099-NEC etc) | Local LLM via system prompt | 7B+ | Keyword detection in `detector.py` can pre-filter before LLM call |
| Extracts all fields and maps to box numbers | Local LLM via structured JSON prompt | 7B–14B for simple docs | |
| Handles multi-state W2s (NJ wages + NY local) | Local LLM | 14B+ | Smaller models miss multi-jurisdiction splits |
| Decodes non-standard box 12 codes (DI, FLI, UI/WF/SWF) | Local LLM with hints in system prompt | 14B+ | Add a code lookup table to your system prompt |
| Deduplicates Copy B/C/2 repetitions | Local LLM instruction in system prompt | Any | Explicitly tell the model "output one record per employee" |
| Parses K-1 line items and codes | Local LLM | 32B+ | K-1s are the most complex; smaller models hallucinate line items |
| Extracts 1099-B transaction tables | Local LLM | 32B+ | Multi-row tables with many fields; needs large context window |
| Flags unusual situations in metadata.notes | Local LLM instruction | 14B+ | Tell model to flag multi-state credits, non-standard codes, etc |

```python
# OpenAI-compatible call to any local endpoint
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="local"
)

response = client.chat.completions.create(
    model="qwen2.5:32b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": extracted_text}
    ],
    temperature=0,
    max_tokens=4096
)

result = response.choices[0].message.content
```

---

### Vision / VLM Path (alternative to OCR + LLM)

For badly scanned or photographed documents, skip OCR entirely and send the
page image directly to a vision-capable model. The VLM reads the layout visually
and extracts fields in one pass.

| What Claude does | Offline equivalent | Notes |
|---|---|---|
| Reads document as an image | Send base64 PNG to VLM via vision message | Requires multimodal model |
| Understands form grid layout | VLM spatial reasoning | Better than OCR + text LLM for degraded scans |
| Handles rotated or skewed scans | VLM (most handle rotation) | pytesseract needs pre-deskewing |

```python
# Vision message to Ollama VLM (llava, minicpm-v, llama3.2-vision)
import base64, openai
from pdf2image import convert_from_path

pages = convert_from_path("bad_scan.pdf", dpi=300)
img_bytes = pages[0].tobytes("raw", "RGB")

# Save to PNG and encode
pages[0].save("/tmp/page0.png")
with open("/tmp/page0.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="local")
response = client.chat.completions.create(
    model="llava:34b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",      "text": "Extract all W2 fields. Return JSON only. Schema: ..."}
        ]
    }],
    max_tokens=4096
)
```

---

### Structured Output / JSON Validation

| What Claude does | Offline equivalent | Notes |
|---|---|---|
| Returns valid JSON reliably | Pydantic v2 model validation + retry | LLMs sometimes produce markdown-fenced JSON; strip before parse |
| Self-corrects on schema errors | Retry loop with stricter prompt | Send the validation error back to the model on retry |
| Handles missing fields gracefully | Pydantic with default `None` fields | Never omit keys; use null |

```python
import json
from pydantic import ValidationError
from w2extract.schema import W2Document

def parse_with_retry(llm_client, text, schema_class, retries=2):
    prompt = text
    for attempt in range(retries + 1):
        raw = llm_client.chat.completions.create(...).choices[0].message.content
        # Strip markdown fences if present
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(clean)
            return schema_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                prompt = f"Your previous response was invalid: {e}\nReturn only valid JSON matching the schema."
            else:
                raise
```

---

### Model Recommendations by Task

| Task | Minimum | Recommended | Your hardware options |
|---|---|---|---|
| Simple W2 (single state, no box 12) | 7B | 14B | Any node |
| W2 with box 12 codes + multi-state | 14B | 32B | 3090 NVLink (48GB), PRO 6000, Mac Studio |
| 1099-NEC / 1099-INT / 1099-DIV | 7B | 14B | Any node |
| 1099-B with many transactions | 32B | 72B | PRO 6000 or Mac Studio |
| K-1 (1065/1120S/1041) | 32B | 72B | PRO 6000 or Mac Studio |
| Mixed batch (multiple doc types) | 32B | 72B | PRO 6000 or Mac Studio |
| Scanned/degraded (VLM path) | llava:13b | llava:34b / minicpm-v | Any GPU node |

Specific models tested to follow JSON schema reliably:

```
Ollama:
  qwen2.5:32b          # Best JSON discipline at 32B, fits 3090 NVLink
  qwen2.5:72b          # Best overall, requires PRO 6000 or Mac Studio
  mistral-nemo:12b     # Good for simple docs, fast on single 3090
  llava:34b            # Vision path
  minicpm-v:8b         # Vision path, lighter weight

vLLM (for batch throughput):
  Qwen/Qwen2.5-32B-Instruct   # --tensor-parallel-size 2 on 3090 NVLink
  Qwen/Qwen2.5-72B-Instruct   # Single PRO 6000
  
LM Studio / MLX (Mac Studio M3 Ultra 256GB):
  Qwen2.5-72B-Instruct-MLX    # Runs comfortably, fast on Metal
```

---

### PII Protection

Since this is the whole point of going local:

| Risk | Mitigation |
|---|---|
| SSN in extracted text sent to cloud API | Use local endpoint only — `--model http://localhost:...` |
| SSN in output JSON stored on disk | `--redact true` (default) masks to last 4 digits in output |
| SSN in LLM prompt/context | Unavoidable for extraction — keep model local, no cloud |
| PDF stored in temp files during OCR | `pdf2image` writes to `/tmp` by default; override with `output_folder` param and wipe after |
| Model logs requests | Ollama: no request logging by default. vLLM: disable with `--disable-log-requests` |
| vLLM prometheus metrics expose prompt content | Don't expose vLLM port beyond localhost |

```python
# Secure temp dir for OCR intermediate files
import tempfile, shutil
from pdf2image import convert_from_path

tmpdir = tempfile.mkdtemp()
try:
    pages = convert_from_path("scan.pdf", dpi=300, output_folder=tmpdir)
    # ... process pages ...
finally:
    shutil.rmtree(tmpdir)  # wipe PNG intermediates

# Redact SSN in output
import re
def redact_ssn(text: str) -> str:
    return re.sub(r'\b\d{3}-\d{2}-(\d{4})\b', r'XXX-XX-\1', text)
```

---

## Install



### System dependencies

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows (choco)
choco install tesseract poppler
```

### Python install

```bash
git clone https://github.com/yourorg/w2extract
cd w2extract
pip install -e .
```

### requirements.txt

```
openai>=1.0.0
pdfminer.six
pdf2image
pytesseract
easyocr
paddlepaddle
paddleocr
pydantic>=2.0
torch
Pillow
rich
```

> For GPU-accelerated OCR install the CUDA build of torch first:
> `pip install torch --index-url https://download.pytorch.org/whl/cu124`
> then install the rest.

---

## Quick Start

```bash
# Single file, stdout JSON
w2extract 25-W2.pdf --model http://localhost:11434/v1

# Scanned PDF with GPU OCR
w2extract scan.pdf --ocr-engine easyocr --gpu true --model http://localhost:11434/v1

# Batch directory to CSV
w2extract --batch ./w2s/ --format csv -o all_w2s.csv --model http://localhost:8000/v1

# Lacerte import format
w2extract 25-W2.pdf --format lacerte -o import.txt --model http://localhost:11434/v1

# Vision model for degraded scans (bypasses OCR entirely)
w2extract bad_scan.pdf --vision --model http://localhost:11434/v1 --vision-model llava:34b
```

---

## CLI Reference

```
w2extract <input> [options]

Arguments:
  input                   PDF file or directory (with --batch)

Options:
  -o, --output            Output file path (default: stdout)
  --model                 OpenAI-compatible local endpoint (required)
                          Examples:
                            http://localhost:11434/v1     Ollama
                            http://localhost:8000/v1      vLLM
                            http://localhost:1234/v1      LM Studio
  --model-name            Model name to pass in API call (default: "local")
                          For Ollama: qwen2.5:32b, mistral, etc.
                          For vLLM: matches loaded model path
  --vision                Use vision/VLM model instead of OCR for image pages
  --vision-model          VLM model name for image pages (default: llava:34b)
  --ocr-engine            tesseract | easyocr | paddleocr (default: auto)
                          auto = easyocr if CUDA available, else tesseract
  --gpu                   true | false | auto (default: auto)
  --dpi                   OCR scan DPI, 300 or 400 (default: 300)
  --batch                 Process all PDFs in input directory
  --format                json | csv | lacerte | txf (default: json)
  --redact                Mask full SSN/TIN in output (default: true)
  --temperature           LLM temperature (default: 0)
  --max-tokens            LLM max tokens (default: 4096)
  --retry                 Retry count on invalid JSON response (default: 2)
  --verbose               Print pipeline steps to stderr
```

---

## Supported Document Types

| Form       | Description                              | Auto-detected |
|------------|------------------------------------------|---------------|
| W-2        | Wage and Tax Statement                   | Yes           |
| 1099-NEC   | Non-Employee Compensation                | Yes           |
| 1099-MISC  | Miscellaneous Income                     | Yes           |
| 1099-INT   | Interest Income                          | Yes           |
| 1099-DIV   | Dividends and Distributions              | Yes           |
| 1099-B     | Proceeds from Broker Transactions        | Yes           |
| 1099-R     | Distributions from Pensions/IRAs         | Yes           |
| 1099-G     | Government Payments / Unemployment       | Yes           |
| 1099-SA    | HSA/MSA Distributions                   | Yes           |
| 1099-K     | Payment Card and Network Transactions    | Yes           |
| 1099-S     | Proceeds from Real Estate Transactions   | Yes           |
| 1099-C     | Cancellation of Debt                     | Yes           |
| 1099-A     | Acquisition or Abandonment of Property  | Yes           |
| 1098       | Mortgage Interest Statement              | Yes           |
| 1098-T     | Tuition Statement                        | Yes           |
| 1098-E     | Student Loan Interest Statement          | Yes           |
| SSA-1099   | Social Security Benefit Statement        | Yes           |
| RRB-1099   | Railroad Retirement Benefits             | Yes           |
| K-1 1065   | Partner's Share of Income (Partnership)  | Yes           |
| K-1 1120S  | Shareholder's Share of Income (S-Corp)   | Yes           |
| K-1 1041   | Beneficiary's Share of Income (Estate)   | Yes           |

---

## Local Model Setup

All backends use the OpenAI-compatible chat completions API. No API keys required
for local endpoints — the client sends `api_key="local"` which is ignored by
Ollama and vLLM.

### Ollama

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (recommend 32B+ for complex multi-doc batches)
ollama pull qwen2.5:32b
ollama pull llava:34b          # vision model for scanned docs

# Ollama serves at http://localhost:11434/v1 by default
w2extract input.pdf --model http://localhost:11434/v1 --model-name qwen2.5:32b
```

### vLLM (recommended for batch processing and PRO 6000)

```bash
pip install vllm

# Single GPU
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --port 8000

# Tensor parallel across two 3090s (48GB NVLink pool)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000

# RTX PRO 6000 Blackwell (96GB) — fits 70B models
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --port 8000

w2extract --batch ./w2s/ --model http://localhost:8000/v1 --model-name local
```

### LM Studio

Start the local server in LM Studio (port 1234 by default), then:

```bash
w2extract input.pdf --model http://localhost:1234/v1 --model-name local
```

### Mac (Apple Silicon — MLX via Ollama or LM Studio)

Ollama on Apple Silicon automatically routes to Metal. No extra flags needed.
The Mac Studio M3 Ultra (256GB) can run 70B+ models comfortably.

```bash
ollama pull qwen2.5:72b
w2extract input.pdf --model http://localhost:11434/v1 --model-name qwen2.5:72b
```

---

## GPU Acceleration

### OCR GPU support

| Engine     | GPU Backend      | Auto-detected | Flag                        |
|------------|------------------|---------------|-----------------------------|
| EasyOCR    | CUDA / MPS       | Yes           | `--ocr-engine easyocr`      |
| PaddleOCR  | CUDA             | Yes           | `--ocr-engine paddleocr`    |
| Tesseract  | CPU only         | N/A           | `--ocr-engine tesseract`    |

Auto-select logic in `ocr.py`:
```python
import torch
if torch.cuda.is_available():
    engine = "easyocr"    # CUDA
elif torch.backends.mps.is_available():
    engine = "easyocr"    # Apple Silicon
else:
    engine = "tesseract"  # CPU fallback
```

Force GPU on/off:
```bash
w2extract scan.pdf --ocr-engine easyocr --gpu true
w2extract scan.pdf --ocr-engine tesseract --gpu false
```

### Vision model path (alternative to OCR)

For badly degraded or handwritten scans, skip OCR entirely and send page images
directly to a VLM. The VLM both reads and structures the document.

```bash
w2extract bad_scan.pdf --vision --vision-model llava:34b \
  --model http://localhost:11434/v1
```

Pipeline in this mode:
1. pdf2image converts each page to PNG at --dpi resolution
2. Images base64-encoded and sent as vision messages to the VLM
3. VLM returns structured JSON directly (same schema)

---

## Output Formats

### json (default)

Pretty-printed JSON matching the schema below. One `documents` array containing
all records found across the entire PDF.

### csv

One row per document/employee. Multi-value fields (box12, state, local, transactions)
are serialized as JSON strings within their cells. Headers match schema field names.

### lacerte

Tab-delimited .txt for Lacerte W2/1099 import. Field order:

```
W2:   SSN | EmployerName | EIN | Box1 | Box2 | Box3 | Box4 | Box5 | Box6 |
      Box12Code1 | Box12Amt1 | Box12Code2 | Box12Amt2 | Box12Code3 | Box12Amt3 |
      Box12Code4 | Box12Amt4 | Box14Label1 | Box14Amt1 | Box14Label2 | Box14Amt2 |
      StateAbbr | StateID | Box16 | Box17 | LocalityName | Box18 | Box19

1099-NEC: RecipientSSN | PayerName | PayerTIN | Box1 | Box4 |
          StateAbbr | StateID | Box5 | Box6

1099-INT: RecipientSSN | PayerName | PayerTIN | Box1 | Box2 | Box3 | Box4 |
          Box8 | StateAbbr | StateID | Box16 | Box17

1099-DIV: RecipientSSN | PayerName | PayerTIN | Box1a | Box1b | Box2a |
          Box4 | Box5 | StateAbbr | StateID | Box14 | Box15

1099-R:   RecipientSSN | PayerName | PayerTIN | Box1 | Box2a | Box4 |
          Box7DistCode | Box7IRACheckbox | StateAbbr | StateID | Box14 | Box15
```

### txf

TXF (Tax Exchange Format) for import into TurboTax, TaxAct, UltraTax.
Produces standard TXF v42 format per document type.

---

## JSON Schema Reference

All document types share the top-level wrapper:

```json
{
  "documents": [
    {
      "document_type": "W2",
      "tax_year": 2025,
      "data": { },
      "metadata": {
        "is_corrected": false,
        "is_void": false,
        "is_summary_sheet": false,
        "payer_tin_type": "EIN",
        "notes": []
      }
    }
  ]
}
```

---

### W-2

```json
{
  "employer": {
    "name": "Acme Corp",
    "ein": "12-3456789",
    "address": "123 Main St, Anytown CA 90001",
    "state_id": "CA 123-456-789"
  },
  "employee": {
    "name": "Jane Smith",
    "ssn_last4": "XXX-XX-1234",
    "address": "456 Elm St, Anytown CA 90001"
  },
  "federal": {
    "box1_wages": 75000.00,
    "box2_federal_withheld": 12000.00,
    "box3_ss_wages": 75000.00,
    "box4_ss_withheld": 4650.00,
    "box5_medicare_wages": 75000.00,
    "box6_medicare_withheld": 1087.50,
    "box7_ss_tips": null,
    "box8_allocated_tips": null,
    "box10_dependent_care": null,
    "box11_nonqualified_plans": null
  },
  "box12": [
    { "code": "AA", "amount": 3833.36, "description": "Designated Roth contributions under 401(k)" },
    { "code": "DD", "amount": 3206.32, "description": "Cost of employer-sponsored health coverage (non-taxable)" },
    { "code": "DI", "amount": 176.32,  "description": "NJ Disability Insurance (non-standard ADP code)" }
  ],
  "box13": {
    "statutory_employee": false,
    "retirement_plan": true,
    "third_party_sick_pay": false
  },
  "box14_other": [
    { "label": "CA SDI", "amount": 92.16 },
    { "label": "FLI",    "amount": 252.97 }
  ],
  "state": [
    {
      "state_abbr": "NJ",
      "state_id": "352-844-385/000",
      "box16_state_wages": 76666.65,
      "box17_state_withheld": 2643.35
    }
  ],
  "local": [
    {
      "locality_name": "NYRES",
      "box18_local_wages": 76666.65,
      "box19_local_withheld": 2907.36
    }
  ]
}
```

---

### 1099-NEC

```json
{
  "payer": {
    "name": "Client LLC",
    "tin": "98-7654321",
    "address": "789 Oak Ave, Chicago IL 60601",
    "phone": "312-555-0100"
  },
  "recipient": {
    "name": "John Contractor",
    "tin_last4": "XXX-XX-5678",
    "address": "321 Pine St, Denver CO 80201"
  },
  "account_number": null,
  "box1_nonemployee_compensation": 15000.00,
  "box2_direct_sales": false,
  "box4_federal_withheld": 0.0,
  "state": [
    {
      "state_abbr": "CO",
      "state_id": "CO-123456",
      "box5_state_income": 15000.00,
      "box6_state_withheld": 450.00
    }
  ]
}
```

---

### 1099-MISC

```json
{
  "payer": {
    "name": "Property Mgmt Co",
    "tin": "11-2233445",
    "address": "100 Commerce Blvd, Dallas TX 75201",
    "phone": null
  },
  "recipient": {
    "name": "Mary Landlord",
    "tin_last4": "XXX-XX-9012",
    "address": "200 Rental Rd, Austin TX 78701"
  },
  "account_number": null,
  "box1_rents": 24000.00,
  "box2_royalties": 0.0,
  "box3_other_income": 0.0,
  "box4_federal_withheld": 0.0,
  "box5_fishing_boat_proceeds": 0.0,
  "box6_medical_health_payments": 0.0,
  "box7_direct_sales": false,
  "box8_substitute_payments": 0.0,
  "box9_crop_insurance": 0.0,
  "box10_gross_proceeds_attorney": 0.0,
  "box11_fish_purchased": 0.0,
  "box12_section_409a_deferrals": 0.0,
  "box13_fatca": false,
  "box14_excess_golden_parachute": 0.0,
  "box15_nonqualified_deferred": 0.0,
  "state": [
    {
      "state_abbr": "TX",
      "state_id": null,
      "box16_state_income": 24000.00,
      "box17_state_withheld": 0.0
    }
  ]
}
```

---

### 1099-INT

```json
{
  "payer": {
    "name": "First National Bank",
    "tin": "44-5566778",
    "address": "500 Bank St, New York NY 10001",
    "phone": "800-555-0200"
  },
  "recipient": {
    "name": "Bob Saver",
    "tin_last4": "XXX-XX-3456",
    "address": "789 Savings Ln, Portland OR 97201"
  },
  "account_number": "XXXX-1234",
  "box1_interest_income": 1250.00,
  "box2_early_withdrawal_penalty": 0.0,
  "box3_us_savings_bond_interest": 0.0,
  "box4_federal_withheld": 0.0,
  "box5_investment_expenses": 0.0,
  "box6_foreign_tax_paid": 0.0,
  "box7_foreign_country": null,
  "box8_tax_exempt_interest": 0.0,
  "box9_specified_private_activity_bond_interest": 0.0,
  "box10_market_discount": 0.0,
  "box11_bond_premium": 0.0,
  "box12_bond_premium_treasury": 0.0,
  "box13_bond_premium_tax_exempt": 0.0,
  "box14_tax_exempt_bond_cusip": null,
  "box15_fatca": false,
  "state": [
    {
      "state_abbr": "OR",
      "state_id": null,
      "box16_state_income": 1250.00,
      "box17_state_withheld": 0.0
    }
  ]
}
```

---

### 1099-DIV

```json
{
  "payer": {
    "name": "Vanguard",
    "tin": "23-4567890",
    "address": "PO Box 2600, Valley Forge PA 19482",
    "phone": "800-662-7447"
  },
  "recipient": {
    "name": "Alice Investor",
    "tin_last4": "XXX-XX-7890",
    "address": "100 Portfolio Dr, Seattle WA 98101"
  },
  "account_number": "XXXX-5678",
  "box1a_total_ordinary_dividends": 3200.00,
  "box1b_qualified_dividends": 2800.00,
  "box2a_total_capital_gain": 1500.00,
  "box2b_unrecap_sec1250_gain": 0.0,
  "box2c_section1202_gain": 0.0,
  "box2d_collectibles_gain": 0.0,
  "box2e_section897_ordinary_dividends": 0.0,
  "box2f_section897_capital_gain": 0.0,
  "box3_nondividend_distributions": 0.0,
  "box4_federal_withheld": 0.0,
  "box5_section199a_dividends": 120.00,
  "box6_investment_expenses": 0.0,
  "box7_foreign_tax_paid": 45.00,
  "box8_foreign_country": "Various",
  "box9_cash_liquidation": 0.0,
  "box10_noncash_liquidation": 0.0,
  "box11_fatca": false,
  "box12_exempt_interest_dividends": 0.0,
  "box13_specified_private_activity": 0.0,
  "state": [
    {
      "state_abbr": "WA",
      "state_id": null,
      "box14_state_income": 0.0,
      "box15_state_withheld": 0.0
    }
  ]
}
```

---

### 1099-B

```json
{
  "payer": {
    "name": "Fidelity Investments",
    "tin": "04-2382768",
    "address": "82 Devonshire St, Boston MA 02109"
  },
  "recipient": {
    "name": "Alice Investor",
    "tin_last4": "XXX-XX-7890",
    "address": "100 Portfolio Dr, Seattle WA 98101"
  },
  "account_number": "XXXX-9012",
  "transactions": [
    {
      "description": "100 SH AAPL",
      "cusip": "037833100",
      "date_acquired": "2023-03-15",
      "date_sold": "2025-09-10",
      "box1a_quantity": 100,
      "box1b_date_acquired": "2023-03-15",
      "box1c_date_sold": "2025-09-10",
      "box1d_proceeds": 18500.00,
      "box1e_cost_basis": 14200.00,
      "box1f_accrued_market_discount": 0.0,
      "box1g_wash_sale_loss_disallowed": 0.0,
      "box2_term": "LONG",
      "box3_basis_reported_to_irs": true,
      "box4_federal_withheld": 0.0,
      "box5_noncovered_security": false,
      "box6_reported_gross_or_net": "GROSS",
      "box7_loss_not_allowed": false,
      "box8_proceeds_from_collectibles": false,
      "box9_unrealized_profit_open_contracts": 0.0,
      "box10_unrealized_profit_closed_contracts": 0.0,
      "box11_aggregate_profit_loss": 0.0,
      "box12_basis_reported": true,
      "box13_fatca": false,
      "state": [
        {
          "state_abbr": "WA",
          "state_id": null,
          "box14_state_income": 0.0,
          "box15_state_withheld": 0.0,
          "box16_state_income": 18500.00
        }
      ]
    }
  ],
  "summary": {
    "total_proceeds": 18500.00,
    "total_cost_basis": 14200.00,
    "total_federal_withheld": 0.0
  }
}
```

---

### 1099-R

```json
{
  "payer": {
    "name": "Fidelity Retirement Services",
    "tin": "04-3523567",
    "address": "PO Box 770001, Cincinnati OH 45277",
    "phone": "800-343-3548"
  },
  "recipient": {
    "name": "Robert Retiree",
    "tin_last4": "XXX-XX-2345",
    "address": "55 Sunset Blvd, Phoenix AZ 85001"
  },
  "account_number": "XXXX-3456",
  "box1_gross_distribution": 45000.00,
  "box2a_taxable_amount": 45000.00,
  "box2b_taxable_amount_not_determined": false,
  "box2b_total_distribution": false,
  "box3_capital_gain": 0.0,
  "box4_federal_withheld": 9000.00,
  "box5_employee_contributions": 0.0,
  "box6_net_unrealized_appreciation": 0.0,
  "box7_distribution_code": "7",
  "box7_ira_sep_simple": false,
  "box8_other": 0.0,
  "box9a_your_percentage_of_total": null,
  "box9b_total_employee_contributions": null,
  "box10_amount_allocable_to_ira": 0.0,
  "box11_1st_year_of_desig_roth": null,
  "box12_fatca": false,
  "box13_date_of_payment": null,
  "box14_state_withheld": 2700.00,
  "box15_state_id": "AZ-12345",
  "box16_state_distribution": 45000.00,
  "box17_local_withheld": 0.0,
  "box18_locality_name": null,
  "box19_local_distribution": 0.0
}
```

---

### 1099-G

```json
{
  "payer": {
    "name": "California Employment Development Department",
    "tin": "94-2650502",
    "address": "PO Box 826880, Sacramento CA 94280",
    "phone": "800-300-5616"
  },
  "recipient": {
    "name": "Susan Unemployed",
    "tin_last4": "XXX-XX-6789",
    "address": "77 Jobless Ave, Los Angeles CA 90001"
  },
  "account_number": null,
  "box1_unemployment_compensation": 12000.00,
  "box2_state_or_local_income_tax_refund": 0.0,
  "box3_box2_applies_to_tax_year": null,
  "box4_federal_withheld": 1200.00,
  "box5_rtaa_payments": 0.0,
  "box6_taxable_grants": 0.0,
  "box7_agriculture_payments": 0.0,
  "box8_market_gain": false,
  "box9_market_gain_amount": 0.0,
  "box10a_state_abbr": "CA",
  "box10b_state_id": "CA-0000001",
  "box11_state_income_tax_withheld": 600.00
}
```

---

### 1099-SA

```json
{
  "payer": {
    "name": "HSA Bank",
    "tin": "39-1983024",
    "address": "605 N. 8th St, Sheboygan WI 53081"
  },
  "recipient": {
    "name": "Michael HSA",
    "tin_last4": "XXX-XX-0123",
    "address": "888 Health Way, Nashville TN 37201"
  },
  "account_number": "XXXX-7890",
  "box1_gross_distribution": 3500.00,
  "box2_earnings_on_excess_contributions": 0.0,
  "box3_distribution_code": "1",
  "box4_fmv_on_date_of_death": null,
  "box5_account_type": "HSA"
}
```

---

### 1099-K

```json
{
  "payer": {
    "name": "PayPal Inc",
    "tin": "26-1080364",
    "address": "2211 North First Street, San Jose CA 95131",
    "phone": "888-221-1161"
  },
  "recipient": {
    "name": "Etsy Seller",
    "tin_last4": "XXX-XX-4567",
    "address": "432 Craft Lane, Portland OR 97201"
  },
  "account_number": "XXXX-1234",
  "box1a_gross_payment_card_transactions": 28000.00,
  "box1b_card_not_present_transactions": 28000.00,
  "box2_merchant_category_code": "5999",
  "box3_number_of_transactions": 342,
  "box4_federal_withheld": 0.0,
  "box5_january": 2100.00,
  "box5_february": 1900.00,
  "box5_march": 2300.00,
  "box5_april": 2200.00,
  "box5_may": 2500.00,
  "box5_june": 2400.00,
  "box5_july": 2600.00,
  "box5_august": 2300.00,
  "box5_september": 2200.00,
  "box5_october": 2500.00,
  "box5_november": 2800.00,
  "box5_december": 2200.00,
  "state": [
    {
      "state_abbr": "OR",
      "state_id": null,
      "box6_state_withheld": 0.0,
      "box7_state_income": 28000.00
    }
  ]
}
```

---

### 1099-S

```json
{
  "filer": {
    "name": "First American Title",
    "tin": "95-1234567",
    "address": "1 First American Way, Santa Ana CA 92707"
  },
  "transferor": {
    "name": "Home Seller",
    "tin_last4": "XXX-XX-8901",
    "address": "123 Sold St, Irvine CA 92602"
  },
  "account_number": null,
  "box1_date_of_closing": "2025-06-15",
  "box2_gross_proceeds": 850000.00,
  "box3_address_of_property": "123 Sold St, Irvine CA 92602",
  "box4_transferor_received_property": false,
  "box5_buyer_part_of_real_estate_tax": 3200.00
}
```

---

### 1099-C

```json
{
  "creditor": {
    "name": "Chase Bank",
    "tin": "13-4994650",
    "address": "PO Box 15298, Wilmington DE 19850"
  },
  "debtor": {
    "name": "Debt Cancelled",
    "tin_last4": "XXX-XX-2345",
    "address": "999 Fresh Start Ave, Miami FL 33101"
  },
  "account_number": "XXXX-6789",
  "box1_date_of_identifiable_event": "2025-04-01",
  "box2_amount_of_debt_discharged": 12500.00,
  "box3_interest_included_in_box2": 1200.00,
  "box4_debt_description": "Credit card debt",
  "box5_personally_liable": true,
  "box6_identifiable_event_code": "A",
  "box7_fmv_of_property": null
}
```

---

### 1099-A

```json
{
  "lender": {
    "name": "Wells Fargo Bank",
    "tin": "41-0449260",
    "address": "PO Box 10335, Des Moines IA 50306"
  },
  "borrower": {
    "name": "Former Owner",
    "tin_last4": "XXX-XX-3456",
    "address": "000 Old House Rd, Detroit MI 48201"
  },
  "account_number": "XXXX-0123",
  "box1_date_of_lender_acquisition": "2025-03-20",
  "box2_balance_of_principal_outstanding": 225000.00,
  "box4_fmv_of_property": 190000.00,
  "box5_personally_liable": true,
  "box6_description_of_property": "123 Old House Rd, Detroit MI 48201"
}
```

---

### 1098 (Mortgage Interest)

```json
{
  "recipient": {
    "name": "Rocket Mortgage",
    "tin": "38-3012700",
    "address": "1050 Woodward Ave, Detroit MI 48226",
    "phone": "800-726-3030"
  },
  "payer": {
    "name": "Home Owner",
    "tin_last4": "XXX-XX-4567",
    "address": "456 Mortgage Lane, Columbus OH 43201"
  },
  "account_number": "XXXX-3456",
  "box1_mortgage_interest": 14200.00,
  "box2_outstanding_mortgage_principal": 380000.00,
  "box3_mortgage_origination_date": "2019-05-01",
  "box4_refund_of_overpaid_interest": 0.0,
  "box5_mortgage_insurance_premiums": 1800.00,
  "box6_points_paid_on_purchase": 0.0,
  "box7_property_address": "456 Mortgage Lane, Columbus OH 43201",
  "box8_number_of_properties": 1,
  "box9_other": null,
  "box10_property_tax": null,
  "box11_acquisition_date": null
}
```

---

### 1098-T (Tuition)

```json
{
  "filer": {
    "name": "State University",
    "tin": "52-0987654",
    "address": "1000 University Ave, College Town OH 43210",
    "phone": "614-555-0300"
  },
  "student": {
    "name": "College Student",
    "tin_last4": "XXX-XX-5678",
    "address": "200 Dorm Hall, College Town OH 43210"
  },
  "account_number": "STU-789012",
  "box1_payments_received": 18500.00,
  "box2_reserved": null,
  "box3_change_in_reporting_method": false,
  "box4_adjustments_prior_year": 0.0,
  "box5_scholarships_grants": 5000.00,
  "box6_adjustments_scholarships_prior_year": 0.0,
  "box7_includes_amounts_for_next_period": false,
  "box8_at_least_half_time_student": true,
  "box9_graduate_student": false,
  "box10_insurance_contract_reimbursement": 0.0
}
```

---

### 1098-E (Student Loan Interest)

```json
{
  "recipient": {
    "name": "Navient",
    "tin": "46-0491342",
    "address": "PO Box 9500, Wilkes-Barre PA 18773",
    "phone": "800-722-1300"
  },
  "borrower": {
    "name": "Grad Student",
    "tin_last4": "XXX-XX-6789",
    "address": "300 Loan St, Boston MA 02101"
  },
  "account_number": "XXXX-4567",
  "box1_student_loan_interest": 2400.00,
  "box2_origination_fees_included": false
}
```

---

### SSA-1099

```json
{
  "payer": {
    "name": "Social Security Administration",
    "tin": "00-0000000",
    "address": "Wilkes-Barre, PA 18769"
  },
  "beneficiary": {
    "name": "Senior Citizen",
    "ssn_last4": "XXX-XX-7890",
    "address": "100 Retirement Dr, Boca Raton FL 33431",
    "medicare_id": null
  },
  "claim_number": "000-00-7890-A",
  "box3_benefits_paid": 24000.00,
  "box4_benefits_repaid": 0.0,
  "box5_net_benefits": 24000.00,
  "box6_voluntary_federal_withheld": 3600.00,
  "box7_address_of_payee": "100 Retirement Dr, Boca Raton FL 33431",
  "description": "Social Security benefits"
}
```

---

### RRB-1099

```json
{
  "payer": {
    "name": "Railroad Retirement Board",
    "tin": "36-3817054",
    "address": "844 N Rush St, Chicago IL 60611"
  },
  "recipient": {
    "name": "Railroad Retiree",
    "ssn_last4": "XXX-XX-8901",
    "address": "500 Rail Ave, Pittsburgh PA 15201"
  },
  "box1_railroad_retirement_tier1": 18000.00,
  "box2_non_contributory_tier1": 12000.00,
  "box3_contributory_tier1": 6000.00,
  "box4_federal_withheld": 1800.00,
  "box5_supplemental_annuity": 0.0,
  "box6_workers_comp_offset": 0.0
}
```

---

### K-1 (Form 1065 — Partnership)

```json
{
  "partnership": {
    "name": "Smith & Jones Partners LLC",
    "ein": "47-1234567",
    "address": "800 Partner Blvd, Houston TX 77001",
    "irs_center": "Ogden"
  },
  "partner": {
    "name": "Silent Partner",
    "tin_last4": "XXX-XX-9012",
    "address": "999 Investor Way, Austin TX 78701"
  },
  "tax_year": 2025,
  "partner_type": "LIMITED",
  "ownership_percentage": 25.0,
  "box1_ordinary_business_income": 45000.00,
  "box2_net_rental_real_estate": 0.0,
  "box3_other_net_rental_income": 0.0,
  "box4_guaranteed_payments_services": 0.0,
  "box5_guaranteed_payments_capital": 0.0,
  "box6_guaranteed_payments_total": 0.0,
  "box7_interest_income": 200.00,
  "box8_ordinary_dividends": 0.0,
  "box8a_qualified_dividends": 0.0,
  "box9a_net_short_term_capital_gain": 0.0,
  "box9b_net_long_term_capital_gain": 5000.00,
  "box9c_unrecaptured_1250_gain": 0.0,
  "box10_net_section_1231_gain": 0.0,
  "box11_other_income": [],
  "box12_section_179_deduction": 0.0,
  "box13_other_deductions": [],
  "box14_self_employment_earnings": 0.0,
  "box15_credits": [],
  "box16_international": null,
  "box17_amti": [],
  "box18_tax_exempt_income": 0.0,
  "box19_distributions": [
    { "code": "A", "amount": 20000.00 }
  ],
  "box20_other_information": [],
  "box21_foreign_taxes": 0.0
}
```

---

### K-1 (Form 1120-S — S-Corporation)

```json
{
  "corporation": {
    "name": "Acme S-Corp Inc",
    "ein": "83-4567890",
    "address": "200 Corporate Dr, Atlanta GA 30301",
    "irs_center": "Ogden",
    "is_publicly_traded": false
  },
  "shareholder": {
    "name": "Founder Owner",
    "tin_last4": "XXX-XX-0123",
    "address": "300 Owner Ave, Atlanta GA 30302"
  },
  "tax_year": 2025,
  "ownership_percentage": 60.0,
  "stock_basis_beginning": 150000.00,
  "box1_ordinary_business_income": 120000.00,
  "box2_net_rental_real_estate": 0.0,
  "box3_other_net_rental": 0.0,
  "box4_interest_income": 500.00,
  "box5a_ordinary_dividends": 0.0,
  "box5b_qualified_dividends": 0.0,
  "box6_royalties": 0.0,
  "box7_net_short_term_capital_gain": 0.0,
  "box8a_net_long_term_capital_gain": 8000.00,
  "box8b_collectibles_gain": 0.0,
  "box8c_unrecaptured_1250_gain": 0.0,
  "box9_net_section_1231_gain": 0.0,
  "box10_other_income": [],
  "box11_section_179_deduction": 0.0,
  "box12_other_deductions": [],
  "box13_credits": [],
  "box14_foreign_transactions": null,
  "box15_amti": [],
  "box16_items_affecting_basis": [
    { "code": "D", "amount": 60000.00, "description": "Distributions" }
  ],
  "box17_other_information": []
}
```

---

### K-1 (Form 1041 — Estate or Trust)

```json
{
  "estate_or_trust": {
    "name": "Smith Family Trust",
    "ein": "91-2345678",
    "address": "c/o Smith Law Group, 100 Trustee St, San Francisco CA 94101",
    "fiduciary_name": "Smith Law Group"
  },
  "beneficiary": {
    "name": "Trust Beneficiary",
    "tin_last4": "XXX-XX-1234",
    "address": "400 Heir Way, San Francisco CA 94102"
  },
  "tax_year": 2025,
  "box1_interest_income": 4500.00,
  "box2a_ordinary_dividends": 3200.00,
  "box2b_qualified_dividends": 2800.00,
  "box3_net_short_term_capital_gain": 0.0,
  "box4a_net_long_term_capital_gain": 12000.00,
  "box4b_unrecaptured_1250_gain": 0.0,
  "box4c_section1202_gain": 0.0,
  "box5_other_portfolio_income": 0.0,
  "box6_ordinary_business_income": 0.0,
  "box7_net_rental_real_estate": 0.0,
  "box8_other_rental_income": 0.0,
  "box9_directly_apportioned_deductions": [],
  "box10_estate_tax_deduction": 0.0,
  "box11_final_year_deductions": [],
  "box12_amti": [],
  "box13_credits": [],
  "box14_other_information": [],
  "box14h_foreign_tax_paid": 0.0,
  "box14i_gross_foreign_income": 0.0
}
```

---

## Project Structure

```
w2extract/
  __init__.py
  cli.py            # argparse entry point
  detector.py       # PDF type detection + document type identification
  extractor.py      # pdfminer text extraction
  ocr.py            # OCR engine abstraction (tesseract/easyocr/paddleocr)
  vision.py         # VLM path for image-first extraction
  llm.py            # Local LLM backend (Ollama/vLLM/LM Studio)
  formatter.py      # json/csv/lacerte/txf output formatters
  schema.py         # pydantic v2 models for all document types
  prompts.py        # System prompt and per-type extraction hints
requirements.txt
setup.py
README.md
```

---

## Pipeline Internals

```
PDF input
    │
    ▼
detector.py
  ├─ pdfminer text extraction per page
  ├─ If page text < 100 chars → flag as IMAGE PAGE
  └─ Identify document type from keywords

    │
    ▼
Per page:
  ├─ TEXT PAGE  → extractor.py (pdfminer)
  └─ IMAGE PAGE → --vision flag?
                   ├─ YES → vision.py (base64 image → VLM)
                   └─ NO  → ocr.py
                              ├─ CUDA/MPS available → easyocr (gpu=True)
                              └─ CPU only           → tesseract (dpi=300)

    │
    ▼
llm.py
  ├─ Combine all page text
  ├─ Send to local endpoint (Ollama/vLLM/LM Studio)
  ├─ System prompt: identify doc type + extract to JSON schema
  ├─ On invalid JSON: retry up to --retry times
  └─ Validate response against pydantic schema

    │
    ▼
formatter.py
  └─ json | csv | lacerte | txf → stdout or --output file
```

---

## Troubleshooting

**OCR produces garbage on a scanned W2**
Increase DPI: `--dpi 400`. If still bad, try `--vision` with a capable VLM (llava:34b,
minicpm-v, or any multimodal model loaded in Ollama).

**LLM returns invalid JSON**
Increase `--max-tokens` (complex K-1s with many line items can overflow 4096).
Try a larger/more capable model. The retry logic will prompt with a stricter
"JSON only" reminder before giving up.

**vLLM not seeing both 3090s**
Confirm NVLink bridge is populated on both cards, then:
`nvidia-smi nvlink --status` should show active links.
Start vLLM with `--tensor-parallel-size 2` explicitly.

**Ollama model not following JSON schema**
Some smaller models ignore structured output instructions. Recommended minimums:
- 7B models: adequate for simple W2/1099-NEC/1099-INT
- 14-32B models: needed for multi-state W2, 1099-B with many transactions, K-1s
- 70B+ models: best for complex K-1s, mixed-document batches

**EasyOCR not using GPU**
Verify CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`.
If False with a CUDA GPU present, reinstall torch with the correct CUDA index URL.
