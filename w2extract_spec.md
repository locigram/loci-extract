# w2extract — Claude Code Build Spec

## Overview

Build a production-ready W2 PDF extraction CLI tool called `w2extract`. It extracts IRS Form W-2 data from PDFs (digital, scanned, or mixed), parses fields using an LLM, and outputs structured data in JSON, CSV, or Lacerte import format.

This spec is complete and self-contained. Implement everything described here. Do not ask clarifying questions — make reasonable decisions where details are ambiguous and document them in the README.

---

## Technology Stack

| Layer | Library | Why |
|---|---|---|
| PDF text extraction | `pdfminer.six` | Best text-layer fidelity for digital PDFs |
| PDF → image | `pdf2image` (poppler) | Required for OCR pipeline |
| OCR default | `pytesseract` | CPU fallback, no install complexity |
| OCR GPU preferred | `easyocr` | Auto CUDA detection, best accuracy on forms |
| OCR alternative | `paddleocr` | Strong on structured/tabular documents |
| LLM default | `anthropic` SDK | Claude API backend |
| LLM local | `openai` SDK | OpenAI-compatible endpoint (Ollama, vLLM) |
| GPU detection | `torch` | `torch.cuda.is_available()` for auto mode |
| Data validation | `pydantic` v2 | Schema enforcement on LLM output |
| CLI | `argparse` | Standard, no extra deps |
| Output | stdlib `json`, `csv` | No extra deps for formatters |

---

## Project Structure

```
w2extract/
├── __init__.py
├── cli.py            # argparse entry point, orchestrates pipeline
├── detector.py       # PDF type detection: text vs image vs mixed
├── extractor.py      # pdfminer.six text extraction
├── ocr.py            # OCR engine abstraction (tesseract / easyocr / paddleocr)
├── llm.py            # LLM backend abstraction (claude / local openai-compat)
├── formatter.py      # Output formatters: json / csv / lacerte
├── schema.py         # Pydantic v2 models for W2Record
└── prompts.py        # System prompt constant

requirements.txt
setup.py
README.md
tests/
└── test_extract.py   # Basic smoke tests against sample outputs
```

---

## CLI Interface

```
w2extract <input> [options]

Positional:
  input               PDF file path, or directory if --batch is set

Options:
  -o, --output        Output file path (default: stdout)
  --model             LLM backend. "claude" (default) or HTTP URL for local
                        e.g. --model http://localhost:11434/v1  (Ollama)
                             --model http://localhost:8000/v1   (vLLM)
  --model-name        Model name to pass to local endpoint (default: "local")
  --gpu               GPU mode: auto | true | false (default: auto)
  --ocr-engine        tesseract | easyocr | paddleocr (default: auto)
  --format            json | csv | lacerte (default: json)
  --batch             Treat input as directory, process all PDFs inside
  --dpi               DPI for PDF→image rendering (default: 300)
  --verbose           Print pipeline steps to stderr
```

### Usage Examples

```bash
# Basic digital PDF
w2extract 25-W2.pdf

# Scanned PDF, GPU OCR
w2extract scan.pdf --ocr-engine easyocr --gpu true

# Local Ollama endpoint
w2extract 25-W2.pdf --model http://localhost:11434/v1 --model-name qwen2.5:32b

# Local vLLM endpoint (PRO 6000 / tensor parallel setup)
w2extract 25-W2.pdf --model http://localhost:8000/v1

# Batch directory → CSV
w2extract --batch ./w2_pdfs/ --format csv -o all_employees.csv

# Lacerte import
w2extract 25-W2.pdf --format lacerte -o lacerte_import.txt

# High DPI for degraded photocopies
w2extract old_scan.pdf --ocr-engine easyocr --dpi 400 --gpu true
```

---

## Extraction Pipeline

### Step 1 — PDF Type Detection (`detector.py`)

```python
def detect_page_types(pdf_path: str) -> dict[int, str]:
    """
    Returns {page_number: "text" | "image"} for each page.
    Threshold: if extracted text < 100 meaningful characters, classify as image.
    Strip whitespace and W2 boilerplate before measuring.
    """
```

Logic:
1. Open with pdfminer, extract text per page
2. Strip whitespace, common boilerplate ("Copy B", "OMB No.", "Dept. of the Treasury")
3. If remaining length < 100 chars → `"image"` page
4. Otherwise → `"text"` page
5. Return map. If ALL pages are text → pure digital. If ALL image → pure scanned. Otherwise → mixed.

### Step 2 — Text Extraction (`extractor.py`)

```python
def extract_text_pages(pdf_path: str, page_numbers: list[int]) -> dict[int, str]:
    """Use pdfminer.six LAParams for W2 form layout. Return {page: raw_text}."""
```

Use `LAParams(line_margin=0.3, char_margin=2.0)` — these values improve W2 field grouping vs defaults.

### Step 3 — OCR (`ocr.py`)

```python
class OCREngine:
    def __init__(self, engine: str = "auto", gpu: str = "auto"): ...
    def extract(self, pdf_path: str, page_numbers: list[int], dpi: int = 300) -> dict[int, str]: ...
```

#### Auto-selection logic

```python
def select_engine(engine_arg: str, gpu_arg: str) -> tuple[str, bool]:
    if engine_arg == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        engine = "easyocr" if use_gpu else "tesseract"
    else:
        engine = engine_arg
        use_gpu = resolve_gpu(gpu_arg)
    return engine, use_gpu
```

#### EasyOCR implementation

```python
import easyocr
import torch

use_gpu = torch.cuda.is_available() if gpu_arg == "auto" else (gpu_arg == "true")
reader = easyocr.Reader(['en'], gpu=use_gpu)

# Per page:
results = reader.readtext(image_path, detail=1)
# results: list of (bbox, text, confidence)
avg_confidence = sum(r[2] for r in results) / len(results) if results else 0
if avg_confidence < 0.6:
    warn(f"Low OCR confidence ({avg_confidence:.2f}) on page {page}. Results may be inaccurate.")
text = " ".join(r[1] for r in results)
```

#### PaddleOCR implementation

```python
from paddleocr import PaddleOCR
import paddle

use_gpu = paddle.device.is_compiled_with_cuda() if gpu_arg == "auto" else (gpu_arg == "true")
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)

result = ocr.ocr(image_path, cls=True)
text = " ".join([line[1][0] for line in result[0]]) if result[0] else ""
```

#### Tesseract implementation

```python
import pytesseract
from PIL import Image

# Config optimized for printed forms
config = "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-/#&"
text = pytesseract.image_to_string(image, config=config)
```

#### pdf2image settings

```python
from pdf2image import convert_from_path
import os

images = convert_from_path(
    pdf_path,
    dpi=dpi,
    first_page=page_num,
    last_page=page_num,
    thread_count=os.cpu_count()  # parallel in batch mode
)
```

### Step 4 — LLM Parsing (`llm.py`)

```python
class LLMBackend:
    def __init__(self, model: str = "claude", model_name: str = "local"): ...
    def parse(self, raw_text: str) -> dict: ...
```

#### Claude API backend

```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=4096,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": raw_text}]
)
raw_json = response.content[0].text
```

#### Local OpenAI-compatible backend

```python
import openai

client = openai.OpenAI(base_url=model_url, api_key="local")
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text}
    ],
    max_tokens=4096,
    temperature=0
)
raw_json = response.choices[0].message.content
```

Auto-detect which backend: if `--model` starts with `http`, use OpenAI-compatible client.

#### JSON parse with retry

```python
import json

def safe_parse(raw: str, client, raw_text: str) -> dict:
    try:
        # Strip accidental markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # One retry with explicit instruction
        retry_text = raw_text + "\n\nIMPORTANT: Return ONLY valid JSON. No explanation, no markdown, no backticks."
        raw2 = client.parse(retry_text)
        cleaned2 = raw2.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned2)
```

---

## LLM System Prompt (`prompts.py`)

Store as a module-level constant `SYSTEM_PROMPT`:

```
You are a strict IRS Form W-2 tax document parser. Extract ALL W2 records from 
the provided text and return ONLY valid JSON matching the schema below.

Rules:
1. Return ONLY JSON. No explanation, no markdown, no backticks.
2. Deduplicate: each W2 is printed 4 times (Copy B, C, 2, 2). Output ONE record per employee.
3. SSN: output last 4 digits only, format as "XXX-XX-1234". Never output full SSN.
4. Missing numeric fields: use 0.0 (never null for dollar amounts).
5. Missing non-numeric fields: use null.
6. Box 12 non-standard codes (DI, FLI, UI/WF/SWF, etc.): include as-is.
7. is_summary_sheet: set true if this is an ADP/Paychex/payroll processor batch 
   summary page rather than an individual employee copy.
8. multi_state: true if the employee has W2 income reported in more than one state.
9. notes: flag anything unusual — multi-state NJ/NY credit situations, non-standard 
   box 12 codes, ADP summary pages, zero federal withholding, missing fields.
10. tax_year: extract from document. Default 2025 if not found.

Output schema:
{
  "employees": [
    {
      "employer": {
        "name": "string",
        "ein": "string",
        "address": "string",
        "state_id": "string | null"
      },
      "employee": {
        "name": "string",
        "ssn_last4": "string",
        "address": "string"
      },
      "federal": {
        "box1_wages": 0.0,
        "box2_federal_withheld": 0.0,
        "box3_ss_wages": 0.0,
        "box4_ss_withheld": 0.0,
        "box5_medicare_wages": 0.0,
        "box6_medicare_withheld": 0.0,
        "box7_ss_tips": null,
        "box8_allocated_tips": null,
        "box10_dependent_care": null,
        "box11_nonqualified_plans": null
      },
      "box12": [
        { "code": "AA", "amount": 0.0, "description": "Designated Roth contributions under 401(k)" }
      ],
      "box13": {
        "statutory_employee": false,
        "retirement_plan": false,
        "third_party_sick_pay": false
      },
      "box14_other": [
        { "label": "CA SDI", "amount": 0.0 }
      ],
      "state": [
        {
          "state_abbr": "CA",
          "state_id": "string",
          "box16_state_wages": 0.0,
          "box17_state_withheld": 0.0
        }
      ],
      "local": [
        {
          "locality_name": "NYRES",
          "box18_local_wages": 0.0,
          "box19_local_withheld": 0.0
        }
      ],
      "metadata": {
        "tax_year": 2025,
        "document_type": "string",
        "is_summary_sheet": false,
        "multi_state": false,
        "notes": []
      }
    }
  ]
}
```

---

## Pydantic Schema (`schema.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional

class Employer(BaseModel):
    name: str
    ein: str
    address: str
    state_id: Optional[str] = None

class Employee(BaseModel):
    name: str
    ssn_last4: str
    address: str

class Federal(BaseModel):
    box1_wages: float = 0.0
    box2_federal_withheld: float = 0.0
    box3_ss_wages: float = 0.0
    box4_ss_withheld: float = 0.0
    box5_medicare_wages: float = 0.0
    box6_medicare_withheld: float = 0.0
    box7_ss_tips: Optional[float] = None
    box8_allocated_tips: Optional[float] = None
    box10_dependent_care: Optional[float] = None
    box11_nonqualified_plans: Optional[float] = None

class Box12Item(BaseModel):
    code: str
    amount: float
    description: str = ""

class Box13(BaseModel):
    statutory_employee: bool = False
    retirement_plan: bool = False
    third_party_sick_pay: bool = False

class Box14Item(BaseModel):
    label: str
    amount: float

class StateRecord(BaseModel):
    state_abbr: str
    state_id: str
    box16_state_wages: float = 0.0
    box17_state_withheld: float = 0.0

class LocalRecord(BaseModel):
    locality_name: str
    box18_local_wages: float = 0.0
    box19_local_withheld: float = 0.0

class Metadata(BaseModel):
    tax_year: int = 2025
    document_type: str = ""
    is_summary_sheet: bool = False
    multi_state: bool = False
    notes: list[str] = Field(default_factory=list)

class W2Record(BaseModel):
    employer: Employer
    employee: Employee
    federal: Federal
    box12: list[Box12Item] = Field(default_factory=list)
    box13: Box13 = Field(default_factory=Box13)
    box14_other: list[Box14Item] = Field(default_factory=list)
    state: list[StateRecord] = Field(default_factory=list)
    local: list[LocalRecord] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

class W2Extraction(BaseModel):
    employees: list[W2Record]
```

After parsing LLM JSON, validate with:
```python
result = W2Extraction.model_validate(parsed_json)
```

---

## Output Formatters (`formatter.py`)

### JSON

Pretty-print the validated pydantic model:
```python
def to_json(extraction: W2Extraction) -> str:
    return extraction.model_dump_json(indent=2)
```

### CSV

One row per employee. Multi-value fields (box12, state, local, box14) serialized as 
JSON strings in their cells.

Column order:
```
tax_year, employer_name, employer_ein, employer_address,
employee_name, employee_ssn_last4, employee_address,
box1_wages, box2_federal_withheld, box3_ss_wages, box4_ss_withheld,
box5_medicare_wages, box6_medicare_withheld,
box7_ss_tips, box8_allocated_tips, box10_dependent_care, box11_nonqualified_plans,
box12_json, box13_retirement_plan, box13_statutory_employee,
box14_json, state_json, local_json,
is_summary_sheet, multi_state, notes_json
```

### Lacerte Tab-Delimited Import

One row per employee. Tab-separated. No header row. Lacerte W2 import field order:

```
SSN (full — prompt user to fill in before import, output as XXXXX{ssn_last4})
EmployerName
EIN
Box1_Wages
Box2_FederalWithheld
Box3_SSWages
Box4_SSWithheld
Box5_MedicareWages
Box6_MedicareWithheld
Box12_Code1
Box12_Amount1
Box12_Code2
Box12_Amount2
Box12_Code3
Box12_Amount3
Box12_Code4
Box12_Amount4
Box14_Label1
Box14_Amount1
Box14_Label2
Box14_Amount2
Box14_Label3
Box14_Amount3
StateAbbr
StateID
Box16_StateWages
Box17_StateWithheld
LocalityName
Box18_LocalWages
Box19_LocalWithheld
```

Pad box12 and box14 to 4 slots each with empty strings if fewer entries exist.
For multi-state employees, output one row per state (repeat federal fields).

---

## Known W2 Document Types & Handling Notes

Three document types encountered in development — use as test cases:

### Type 1 — Digital PDF, Simple (e.g. `25-W2.pdf`)
- pdfminer extracts clean text
- 5 employees, all identical wages $7,680.00
- CA only, no box 12, zero federal withholding
- 4 copies per employee per page — deduplication critical
- No OCR needed

### Type 2 — Scanned/Image PDF (e.g. `D2_2__W-2.pdf`)
- ADP W-2 and Earnings Summary cover page
- Image embedded in PDF — pdfminer yields near-zero text
- Needs OCR pipeline
- `is_summary_sheet: true` — employer reconciliation copy, no individual SSN
- Single employee batch (QPL, EIN 82-3306061)
- CA SDI $114.00 in Box 14

### Type 3 — Digital PDF, Complex (e.g. `2025_W-2.pdf`)
- Neural Concept Inc, one employee: Andrew Saxe
- Multi-state: NJ employer, NY resident
- Box 12: AA (Roth 401k $3,833.36), DD (health coverage $3,206.32)
- Box 14: DI $176.32 (NJ Disability), FLI $252.97 (NJ Family Leave), UI/WF/SWF $184.03
- Local locality: NYRES (New York Resident — triggers NJ→NY credit on return)
- `multi_state: true`, note NJ/NY credit situation in metadata.notes
- `DI` is not a standard IRS box 12 code — it's ADP surfacing NJ SDI. Include as-is.

---

## Box 12 Standard Code Reference

Include this as a comment in `prompts.py` for LLM context enrichment if needed:

```
A  = Uncollected SS tax on tips
B  = Uncollected Medicare tax on tips
C  = Taxable cost of group-term life insurance over $50,000
D  = 401(k) elective deferrals
E  = 403(b) salary reduction
F  = 408(k)(6) SEP salary reduction
G  = 457(b) deferrals
H  = 501(c)(18)(D) plan deferrals
J  = Nontaxable sick pay
K  = 20% excise tax on golden parachute payments
L  = Substantiated employee business expense reimbursements
M  = Uncollected SS tax on group-term life (former employees)
N  = Uncollected Medicare on group-term life (former employees)
P  = Excludable moving expense reimbursements (Armed Forces)
Q  = Nontaxable combat pay
R  = Employer Archer MSA contributions
S  = 408(p) SIMPLE salary reduction
T  = Adoption benefits
V  = Income from nonstatutory stock options
W  = Employer HSA contributions
Y  = 409A nonqualified deferred compensation deferrals
Z  = 409A income (also in box 1, subject to 20% additional tax)
AA = Designated Roth 401(k) contributions
BB = Designated Roth 403(b) contributions
DD = Cost of employer-sponsored health coverage (informational, non-taxable)
EE = Designated Roth 457(b) contributions
FF = Qualified small employer HRA benefits
GG = Income from qualified equity grants (83(i))
HH = Aggregate 83(i) election deferrals
II = Medicaid waiver payments excluded under Notice 2014-7
```

---

## Error Handling

| Condition | Behavior |
|---|---|
| LLM returns invalid JSON | Retry once with explicit JSON-only instruction. If still fails, write error to stderr and skip record. |
| pdfminer returns empty string | Flag page as image, route to OCR |
| EasyOCR avg confidence < 0.6 | Warn to stderr: `WARNING: Low OCR confidence (0.XX) on page N of file.pdf` |
| Page produces no text after both methods | Log warning, continue, add note to metadata |
| `ANTHROPIC_API_KEY` not set | Raise clear error: `ANTHROPIC_API_KEY environment variable not set` |
| Local endpoint unreachable | Raise clear error with URL and suggestion to check server |
| Pydantic validation fails | Log raw LLM output to stderr in verbose mode, raise with field-level detail |

---

## `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="w2extract",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "openai",
        "pdfminer.six",
        "pdf2image",
        "pytesseract",
        "easyocr",
        "paddlepaddle",
        "paddleocr",
        "pydantic>=2.0",
        "torch",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "w2extract=w2extract.cli:main",
        ],
    },
)
```

---

## README.md Content

The generated README.md should include:

### System Dependencies

```bash
# macOS
brew install poppler tesseract

# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr

# Windows
# Install poppler: https://github.com/oschwartz10612/poppler-windows
# Install tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Add both to PATH
```

### Python Install

```bash
pip install -e .

# GPU support (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install easyocr

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Environment

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### vLLM Local Setup Note

For high-volume batch processing, a local vLLM endpoint (e.g. on an RTX PRO 6000 
or dual 3090 NVLink) avoids API costs and latency. Start vLLM with:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \
  --port 8000
```

Then use:
```bash
w2extract --batch ./w2s/ --model http://localhost:8000/v1 --model-name Qwen/Qwen2.5-32B-Instruct --format csv -o all.csv
```

---

## Implementation Notes for Claude Code

1. Implement modules in this order: `schema.py` → `prompts.py` → `detector.py` → `extractor.py` → `ocr.py` → `llm.py` → `formatter.py` → `cli.py`
2. Keep all OCR engine imports inside try/except — easyocr and paddleocr are optional. If not installed, fall back to tesseract gracefully.
3. The `--batch` flag should collect all `.pdf` files in the directory recursively and process them sequentially, merging all employees into one output file.
4. For CSV batch output, write header once then append rows per file.
5. For Lacerte batch output, concatenate all rows — Lacerte expects one row per W2, no grouping by file.
6. `verbose` mode should print to stderr only so stdout stays clean for pipe usage.
7. All dollar amounts should be rounded to 2 decimal places in output.
8. The CLI should exit with code 0 on success, 1 on any extraction failure.
