# w2extract — Financial Statements Spec v2

Supersedes FINANCIAL_STATEMENTS_SPEC.md. Append to main README.
Incorporates corrections: Python-side totals verification, verbatim extraction
only (no derived fields), pdfplumber trigger logic, robust GL chunking,
document-family prompt selector, multi-section PDF boundary detection,
token-budget awareness, and CSV output spec.

---

## Table of Contents

- [Tool Map: What I Use vs What You Run](#tool-map-what-i-use-vs-what-you-run)
- [Corrections to v1](#corrections-to-v1)
- [Encoding Detection Pipeline](#encoding-detection-pipeline)
- [pdfplumber Trigger Logic](#pdfplumber-trigger-logic)
- [Document Boundary Detection](#document-boundary-detection)
- [Document-Family Prompt Selector](#document-family-prompt-selector)
- [Token Budget Awareness](#token-budget-awareness)
- [Totals Verification in Python](#totals-verification-in-python)
- [GL Chunking — Robust Strategy](#gl-chunking--robust-strategy)
- [CSV Output Spec](#csv-output-spec)
- [JSON Schema Reference — Financial Statements](#json-schema-reference--financial-statements)
- [Test Fixture Requirements](#test-fixture-requirements)

---

## Tool Map: What I Use vs What You Run

Everything I do when processing a financial PDF maps to a specific offline tool.
None of this is magic — it's a pipeline of well-defined steps.

| What I do | My capability | Your offline equivalent | Notes |
|---|---|---|---|
| Read PDF text layer | Native document parsing in context | `pdfminer.six` — `extract_text()` | Works for clean text-layer PDFs |
| Detect font encoding issues | None — I see the already-extracted text | `pdffonts` (poppler CLI) + `subprocess` | Run before any extraction attempt |
| Read garbled / image PDFs | See image if uploaded as image | `pdf2image` → `pytesseract` / `easyocr` / `paddleocr` | Rasterize first, then OCR |
| Fix upside-down pages | Spatial reasoning over image | `pytesseract.image_to_osd()` → `PIL.Image.rotate()` | Tesseract OSD detects rotation angle |
| Reconstruct column layout | Visual understanding of tables | `pdfplumber` — coordinate-based extraction | Use when pdfminer gives low word density |
| Identify document type | Pattern matching + understanding | `detector.py` regex signatures | Run on clean OCR output |
| Detect section boundaries within one PDF | Context understanding | `boundary_detector.py` regex + blank-line heuristics | See Document Boundary Detection below |
| Extract hierarchical accounts | Semantic understanding | Local LLM via structured JSON prompt | 14B+ for simple, 32B+ for multi-column |
| Map values to correct columns | Semantic + spatial understanding | Local LLM — explicit column mapping in prompt | Most failure-prone step for small models |
| Arithmetic / totals verification | Python post-processing (NOT LLM) | `verifier.py` — `sum(line_items) == section_total` | Never delegate math to the LLM |
| Normalize negative number formats | Python post-processing | `normalize_amount()` in `formatter.py` | `(1,234.56)` and `-1,234.56` → `-1234.56` |
| Compute derived fields | Python post-processing (NOT LLM) | `post_processor.py` — `due_to_from`, `retained_earnings` | Extract raw accounts verbatim, compute after |
| Output CSV | Python | `formatter.py` — `csv.DictWriter` with section-aware flattening | See CSV Output Spec |
| Validate JSON schema | Python | `pydantic v2` — `.model_validate()` with retry | Catch truncated or malformed LLM output |

---

## Corrections to v1

### 1. Totals verification is Python, not LLM

The v1 spec told the model to verify `sum(line_items) == section_total` and set
`metadata.totals_verified`. This is wrong. Models below 32B fail at multi-item
sums reliably; even 32B gets tripped by rounding differences (`0.01` off due to
OCR misreading a digit). Verification belongs in post-processing.

See [Totals Verification in Python](#totals-verification-in-python).

### 2. No derived fields in LLM output

v1 schema included `due_to_from: -227116.11` and `retained_earnings_calculated`.
These are computed values, not values read directly from the document.
`due_to_from` on the sample document is the net of two accounts:
`1020-0000 DUE TO/FROM RESERVES (-227,116.11)` and
`1030-0000 DUE TO/FROM OPERATING (227,116.11)` which sum to zero — the
`-227,116.11` figure is the value of one specific account, extracted verbatim.
`retained_earnings_calculated` is `total_equity - sum(equity_accounts)`, computed
in Python.

**Rule:** The LLM extracts what is on the page, verbatim. Python computes
everything else. If a field requires arithmetic, it is not an LLM output field.

### 3. pdfplumber trigger is now well-defined

v1 said "use pdfplumber for QBO" without a hard trigger. See
[pdfplumber Trigger Logic](#pdfplumber-trigger-logic) for the exact decision rule.

### 4. GL chunking uses token counter + schema boundaries

v1 regex assumed `NNNN-NNNN` account format. See
[GL Chunking — Robust Strategy](#gl-chunking--robust-strategy).

### 5. Document-family prompt selector added

v1 had one system prompt. See [Document-Family Prompt Selector](#document-family-prompt-selector).

### 6. Multi-section PDF boundary detection added

See [Document Boundary Detection](#document-boundary-detection).

### 7. Token budget is auto-computed per doc type

See [Token Budget Awareness](#token-budget-awareness).

---

## Encoding Detection Pipeline

Run in this exact order. Stop at first match.

```python
# detector.py

import re
import subprocess
import unicodedata
from pdfminer.high_level import extract_text


def get_extraction_strategy(pdf_path: str) -> dict:
    """
    Returns:
      strategy: 'text' | 'pdfplumber' | 'ocr' | 'vision'
      reason:   human-readable explanation
      encoding_broken: bool
    """

    # Step 1 — pdffonts: most reliable encoding check
    # Requires poppler-utils (apt install poppler-utils / brew install poppler)
    # Offline equivalent: subprocess.run(["pdffonts", pdf_path])
    uni_missing = _check_pdffonts(pdf_path)
    if uni_missing:
        return {
            "strategy": "ocr",
            "reason": f"Font(s) {uni_missing} have no ToUnicode map "
                      f"(Identity-H, PScript5/Distiller workflow). "
                      f"Text layer is glyph IDs. Must rasterize and OCR.",
            "encoding_broken": True,
        }

    # Step 2 — Extract a sample and check character quality
    try:
        sample = extract_text(pdf_path, maxpages=2)
    except Exception as e:
        return {"strategy": "ocr", "reason": f"pdfminer failed: {e}",
                "encoding_broken": True}

    # Step 3 — No text at all: scanned PDF
    if len(sample.strip()) < 100:
        return {"strategy": "ocr",
                "reason": "No text layer — image/scanned PDF.",
                "encoding_broken": False}

    # Step 4 — High ratio of non-printable characters: encoding corruption
    non_printable = sum(
        1 for c in sample
        if unicodedata.category(c) in ('Cc', 'Cs', 'Co', 'Cn')
        and c not in '\n\t\r'
    )
    ratio = non_printable / len(sample)
    if ratio > 0.15:
        return {
            "strategy": "ocr",
            "reason": f"Non-printable character ratio {ratio:.1%} > 15%. "
                      f"Encoding corruption (ToUnicode missing but pdffonts unavailable).",
            "encoding_broken": True,
        }

    # Step 5 — Text present but no recognizable domain words: garbage text
    lower = sample.lower()
    anchor_words = {
        'total', 'balance', 'account', 'income', 'expense', 'asset',
        'wages', 'tax', 'amount', 'date', 'name', 'net', 'gross', 'paid',
        'cash', 'revenue', 'debit', 'credit', 'equity', 'liability',
    }
    found = sum(1 for w in anchor_words if w in lower)
    if len(sample) > 300 and found < 2:
        return {
            "strategy": "ocr",
            "reason": "Text layer present but no recognizable domain words. "
                      "Likely encoding corruption.",
            "encoding_broken": True,
        }

    # Step 6 — Check word density for pdfplumber trigger (see next section)
    word_density = _word_density_per_page(pdf_path)
    if word_density is not None and word_density < 0.4:
        return {
            "strategy": "pdfplumber",
            "reason": f"Text layer OK but low word density ({word_density:.2f} "
                      f"words/char). Likely coordinate-dependent layout (QBO, "
                      f"Sage). Use pdfplumber for column reconstruction.",
            "encoding_broken": False,
        }

    return {"strategy": "text", "reason": "Text layer clean.", "encoding_broken": False}


def _check_pdffonts(pdf_path: str) -> list[str]:
    """Returns list of font names with uni=no. Empty list means all fonts OK."""
    try:
        result = subprocess.run(
            ["pdffonts", pdf_path],
            capture_output=True, text=True, timeout=10
        )
        bad = []
        for line in result.stdout.strip().split("\n")[2:]:
            cols = line.split()
            # pdffonts columns: name type encoding emb sub uni object-ID
            if len(cols) >= 7 and cols[6] == "no":
                bad.append(cols[0])
        return bad
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []  # pdffonts not available — fall through to heuristics


def _word_density_per_page(pdf_path: str, maxpages: int = 3) -> float | None:
    """
    Ratio of word characters to total characters per page.
    Low density = lots of whitespace/padding = coordinate layout = use pdfplumber.
    """
    try:
        text = extract_text(pdf_path, maxpages=maxpages)
        if not text:
            return None
        word_chars = sum(1 for c in text if c.isalnum())
        return word_chars / len(text)
    except Exception:
        return None
```

---

## pdfplumber Trigger Logic

The rule: use `pdfplumber` when the text layer is present and clean (no encoding
issues) but `pdfminer` produces misaligned columns. This happens when the PDF
was generated by a browser-based tool (QuickBooks Online, Sage Cloud) that places
text at absolute pixel coordinates rather than in reading order.

The trigger is `word_density < 0.4` from `_word_density_per_page()` above —
meaning the text layer has lots of padding/whitespace relative to actual words,
which is the signature of coordinate-placed text.

```python
# extractor.py

import pdfplumber
from pdfminer.high_level import extract_text


def extract_with_strategy(pdf_path: str, strategy: str) -> list[dict]:
    """
    Returns list of {page: int, text: str, tables: list | None}
    """
    if strategy == "text":
        return _extract_pdfminer(pdf_path)
    elif strategy == "pdfplumber":
        return _extract_pdfplumber(pdf_path)
    elif strategy in ("ocr", "vision"):
        raise ValueError("OCR/vision pages must go through ocr.py, not extractor.py")


def _extract_pdfminer(pdf_path: str) -> list[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    results = []
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        text = "".join(
            el.get_text() for el in page_layout
            if isinstance(el, LTTextContainer)
        )
        results.append({"page": i, "text": text, "tables": None})
    return results


def _extract_pdfplumber(pdf_path: str) -> list[dict]:
    """
    Use pdfplumber's coordinate-aware extraction.
    Reconstructs columns from x-position of text elements.
    """
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Try structured table extraction first
            tables = page.extract_tables({
                "vertical_strategy":   "text",
                "horizontal_strategy": "text",
                "min_words_vertical":  2,
                "min_words_horizontal": 1,
            })

            # Fall back to word-level extraction with x-position grouping
            if not tables:
                words = page.extract_words(
                    x_tolerance=5,
                    y_tolerance=5,
                    keep_blank_chars=False,
                )
                text = _reconstruct_text_from_words(words)
                results.append({"page": i, "text": text, "tables": None})
            else:
                # Convert table to text representation
                text = _tables_to_text(tables)
                results.append({"page": i, "text": text, "tables": tables})
    return results


def _reconstruct_text_from_words(words: list[dict]) -> str:
    """
    Group words by y-position (rows) then sort by x-position (columns).
    Handles coordinate-placed text that pdfminer reads out of order.
    """
    if not words:
        return ""
    # Group into rows by y-position with 3pt tolerance
    rows: dict[int, list] = {}
    for w in words:
        y_key = round(w["top"] / 3) * 3
        rows.setdefault(y_key, []).append(w)
    lines = []
    for y_key in sorted(rows.keys()):
        row_words = sorted(rows[y_key], key=lambda w: w["x0"])
        lines.append("  ".join(w["text"] for w in row_words))
    return "\n".join(lines)


def _tables_to_text(tables: list) -> str:
    lines = []
    for table in tables:
        for row in table:
            if row:
                lines.append("  ".join(str(c or "") for c in row))
    return "\n".join(lines)
```

---

## Document Boundary Detection

A single PDF can contain multiple distinct financial statements — the sample
document has a Balance Sheet (pages 1-2) and an Income Statement (pages 3-9).
Each statement needs its own extraction pass with the correct per-type prompt.

```python
# boundary_detector.py

import re
from dataclasses import dataclass


@dataclass
class DocumentSection:
    start_page: int
    end_page: int          # inclusive
    document_type: str
    title: str
    confidence: float      # 0.0-1.0


# Boundary signals: patterns that indicate a new document is starting
BOUNDARY_PATTERNS = [
    # Strong signals — new report header
    (r"^balance sheet", "BALANCE_SHEET", 0.95),
    (r"^income statement", "INCOME_STATEMENT", 0.95),
    (r"^profit\s+(?:and\s+)?loss", "INCOME_STATEMENT", 0.95),
    (r"^trial balance", "TRIAL_BALANCE", 0.95),
    (r"^general ledger", "GENERAL_LEDGER", 0.95),
    (r"^accounts? receivable aging", "ACCOUNTS_RECEIVABLE_AGING", 0.95),
    (r"^accounts? payable aging", "ACCOUNTS_PAYABLE_AGING", 0.95),
    (r"^cash flow", "CASH_FLOW_STATEMENT", 0.95),
    (r"^budget\s+vs\.?\s+actual", "BUDGET_VS_ACTUAL", 0.90),
    (r"^owner statement", "APPFOLIO_OWNER_STATEMENT", 0.90),

    # Weaker signals — report subheaders within multi-report PDFs
    (r"as of\s+\d{2}/\d{2}/\d{4}", "BALANCE_SHEET", 0.60),
    (r"for the (?:month|period|year)\s+(?:ended|ending)", "INCOME_STATEMENT", 0.60),
    (r"from\s+\d{2}/\d{2}/\d{4}\s+to\s+\d{2}/\d{2}/\d{4}", "INCOME_STATEMENT", 0.60),
]


def detect_boundaries(pages: list[dict]) -> list[DocumentSection]:
    """
    Input:  list of {page: int, text: str} from extractor
    Output: list of DocumentSection with page ranges
    """
    sections = []
    current_type = None
    current_title = ""
    current_start = 0
    current_conf = 0.0

    for page_info in pages:
        page_num = page_info["page"]
        # Check first 20 lines of each page for boundary signals
        first_lines = "\n".join(
            page_info["text"].strip().split("\n")[:20]
        ).lower()

        matched_type = None
        matched_title = ""
        matched_conf = 0.0

        for pattern, doc_type, conf in BOUNDARY_PATTERNS:
            if re.search(pattern, first_lines, re.MULTILINE | re.IGNORECASE):
                if conf > matched_conf:
                    matched_type = doc_type
                    matched_conf = conf
                    # Extract title from first non-empty line
                    for line in page_info["text"].strip().split("\n"):
                        if line.strip():
                            matched_title = line.strip()
                            break

        # New section detected
        if matched_type and matched_type != current_type and matched_conf >= 0.85:
            if current_type is not None:
                sections.append(DocumentSection(
                    start_page=current_start,
                    end_page=page_num - 1,
                    document_type=current_type,
                    title=current_title,
                    confidence=current_conf,
                ))
            current_type = matched_type
            current_title = matched_title
            current_start = page_num
            current_conf = matched_conf

    # Close final section
    if current_type is not None:
        sections.append(DocumentSection(
            start_page=current_start,
            end_page=pages[-1]["page"],
            document_type=current_type,
            title=current_title,
            confidence=current_conf,
        ))

    # If no boundaries found, treat whole PDF as one unknown document
    if not sections:
        sections.append(DocumentSection(
            start_page=0,
            end_page=pages[-1]["page"],
            document_type="FINANCIAL_UNKNOWN",
            title="",
            confidence=0.0,
        ))

    return sections
```

---

## Document-Family Prompt Selector

```python
# prompts.py

from enum import Enum


class DocumentFamily(str, Enum):
    TAX = "tax"
    FINANCIAL_SIMPLE = "financial_simple"      # Balance Sheet, single-period P&L
    FINANCIAL_MULTICOLUMN = "financial_multi"  # Comparison, Budget vs Actual
    FINANCIAL_TRANSACTION = "financial_txn"   # GL, AR/AP Aging, Transaction Detail
    FINANCIAL_RESERVE = "financial_reserve"   # HOA reserve allocation


# Document type → family mapping
DOCUMENT_FAMILY_MAP = {
    # Tax forms
    "W2": DocumentFamily.TAX,
    "1099-NEC": DocumentFamily.TAX,
    "1099-MISC": DocumentFamily.TAX,
    "1099-INT": DocumentFamily.TAX,
    "1099-DIV": DocumentFamily.TAX,
    "1099-B": DocumentFamily.TAX,
    "1099-R": DocumentFamily.TAX,
    "1099-G": DocumentFamily.TAX,
    "1099-SA": DocumentFamily.TAX,
    "1099-K": DocumentFamily.TAX,
    "1099-S": DocumentFamily.TAX,
    "1099-C": DocumentFamily.TAX,
    "1099-A": DocumentFamily.TAX,
    "1098": DocumentFamily.TAX,
    "1098-T": DocumentFamily.TAX,
    "1098-E": DocumentFamily.TAX,
    "SSA-1099": DocumentFamily.TAX,
    "RRB-1099": DocumentFamily.TAX,
    "K1-1065": DocumentFamily.TAX,
    "K1-1120S": DocumentFamily.TAX,
    "K1-1041": DocumentFamily.TAX,
    # Simple financial
    "BALANCE_SHEET": DocumentFamily.FINANCIAL_SIMPLE,
    "INCOME_STATEMENT": DocumentFamily.FINANCIAL_SIMPLE,
    "CASH_FLOW_STATEMENT": DocumentFamily.FINANCIAL_SIMPLE,
    "TRIAL_BALANCE": DocumentFamily.FINANCIAL_SIMPLE,
    "RESERVE_ALLOCATION": DocumentFamily.FINANCIAL_RESERVE,
    # Multi-column financial
    "INCOME_STATEMENT_COMPARISON": DocumentFamily.FINANCIAL_MULTICOLUMN,
    "BUDGET_VS_ACTUAL": DocumentFamily.FINANCIAL_MULTICOLUMN,
    "QB_PROFIT_LOSS": DocumentFamily.FINANCIAL_MULTICOLUMN,
    "QB_BALANCE_SHEET": DocumentFamily.FINANCIAL_SIMPLE,
    # Transaction-level
    "GENERAL_LEDGER": DocumentFamily.FINANCIAL_TRANSACTION,
    "ACCOUNTS_RECEIVABLE_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "ACCOUNTS_PAYABLE_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "QB_GENERAL_LEDGER": DocumentFamily.FINANCIAL_TRANSACTION,
    "QB_AR_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "QB_AP_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "QB_TRANSACTION_LIST": DocumentFamily.FINANCIAL_TRANSACTION,
}


TAX_SYSTEM_PROMPT = """
You are a tax document parser. Extract all W2/1099/1098/K-1 records from the
provided text and return ONLY valid JSON. No preamble, no markdown fences.

Rules:
- Deduplicate: W2s repeat 4x (Copy B/C/2/2). Output ONE record per employee.
- SSN: output last 4 digits only as XXX-XX-1234
- Missing fields: use null, never omit keys
- Zero withholding: use 0.0 not null
- Non-standard Box 12 codes (DI, FLI, UI/WF/SWF): include verbatim
- multi_state: true if employee has more than one state in Box 15/16
- notes: flag multi-state credit situations, non-standard codes, ADP/Paychex
  summary pages, missing fields

Schema: {schema}
"""

FINANCIAL_SIMPLE_SYSTEM_PROMPT = """
You are a financial document parser. Extract the balance sheet or income
statement from the provided text and return ONLY valid JSON.
No preamble, no markdown fences.

Rules:
- Extract accounts VERBATIM. Do not compute totals, balances, or derived values.
  Those are computed in post-processing. Your job is faithful extraction only.
- Preserve section hierarchy exactly as it appears in the document.
  Accounts nest under section names. Section totals are labeled lines.
- Negative numbers: normalize ALL formats to negative float.
    (1,234.56) → -1234.56
    -1,234.56  → -1234.56
    1,234.56-  → -1234.56
- Account numbers: extract as shown (e.g. 1018-0000). null if not present.
- Section total rows: include them in a separate "subtotals" list, not in
  "accounts". The verifier will confirm they match.
- Software: detect from headers. Set entity.software to one of:
  QuickBooks Desktop, QuickBooks Online, AppFolio, Yardi, Buildium, Sage, Unknown.
- AppFolio header fields to extract: properties, accounting_basis,
  gl_account_map, level_of_detail, include_zero_balance_accounts, created_on.

Schema: {schema}
"""

FINANCIAL_MULTICOLUMN_SYSTEM_PROMPT = """
You are a financial document parser specializing in multi-period comparison
reports. Extract ALL columns and ALL line items and return ONLY valid JSON.
No preamble, no markdown fences.

Rules:
- COLUMN MAPPING IS CRITICAL. Read the column headers at the top of the report
  carefully. Every account line has one value per column. Map each value to its
  exact column key.
- Common column patterns:
    Monthly actual columns: "Jul 2024", "Aug 2024", ... "Jan 2025"
    YTD columns: "YTD Actual", "YTD Budget"
    Variance columns: "$ Var" or "$ Variance" (can be negative)
    Percent variance: "% Var" (ratio, not decimal — 5.2 means 5.2%)
    Budget: "Annual Budget" or "Annual"
- Missing values in a column for a given account: use null, not 0.0.
  0.0 means the account had zero activity. null means the value was absent.
- Section hierarchy: same as simple financial. Accounts nest under sections.
- OCR ARTIFACTS: Some pages may be rotated 180° producing reversed text.
  If you see reversed strings, interpret by context:
    "yoBpng jenuuy" → "Annual Budget"
    "yoBpng GLA"    → "YTD Budget"
    "anjoy GLA"     → "YTD Actual"
    "awodu|"        → "Income"
    "asuadxy"       → "Expense"
    "saquiny yunossv" → "Account Number"
  Use surrounding numeric context to determine which column a value belongs to
  even if the header text is garbled.
- DO NOT compute variance or totals. Extract only what is on the page.

Schema: {schema}
"""

FINANCIAL_TRANSACTION_SYSTEM_PROMPT = """
You are a financial document parser specializing in transaction-level reports
(General Ledger, AR/AP Aging, Transaction Lists).

Rules:
- Each transaction row is one object in the transactions array.
- Preserve ALL columns: date, type, number/ref, name/payee, memo, split
  account, debit, credit, running balance.
- Null vs 0.0: a blank debit cell is null (no debit). A zero debit is 0.0.
- Running balance: extract verbatim. Do not recompute.
- Beginning/ending balance rows: flag with "row_type": "balance_header" or
  "balance_footer", not "transaction".
- Aging reports: extract each customer/vendor row with one value per age bucket.
  Age bucket headers: extract exactly as labeled (Current, 1-30, 31-60, etc.)

Schema: {schema}
"""

FINANCIAL_RESERVE_SYSTEM_PROMPT = """
You are a financial document parser specializing in HOA reserve fund reports.

Rules:
- Extract every reserve component as a line item with account number, name,
  and current balance verbatim.
- Negative balances are real and meaningful (overspent component). Keep them.
- Bank accounts section: extract separately from reserve allocation components.
- Do NOT compute net reserve balance, percent funded, or due-to/from figures.
  These are computed in post-processing.
- Equity section: extract ALL equity accounts verbatim including opening
  balance equity accounts (e.g. "Appfolio Opening Balance Equity").

Schema: {schema}
"""


def get_prompt(document_type: str, schema: str) -> str:
    family = DOCUMENT_FAMILY_MAP.get(document_type, DocumentFamily.FINANCIAL_SIMPLE)
    template = {
        DocumentFamily.TAX: TAX_SYSTEM_PROMPT,
        DocumentFamily.FINANCIAL_SIMPLE: FINANCIAL_SIMPLE_SYSTEM_PROMPT,
        DocumentFamily.FINANCIAL_MULTICOLUMN: FINANCIAL_MULTICOLUMN_SYSTEM_PROMPT,
        DocumentFamily.FINANCIAL_TRANSACTION: FINANCIAL_TRANSACTION_SYSTEM_PROMPT,
        DocumentFamily.FINANCIAL_RESERVE: FINANCIAL_RESERVE_SYSTEM_PROMPT,
    }[family]
    return template.format(schema=schema)
```

---

## Token Budget Awareness

`--max-tokens 4096` is the correct default for tax forms. It is too small for
multi-column financial reports. Auto-compute based on detected document type.

```python
# llm.py

# Estimated output token budget by document type and size
TOKEN_BUDGETS = {
    # Tax forms
    "W2":              2048,
    "1099-NEC":        1024,
    "1099-MISC":       1024,
    "1099-INT":        1024,
    "1099-DIV":        1024,
    "1099-B":          4096,   # transaction list
    "1099-R":          1024,
    "K1-1065":         3000,
    "K1-1120S":        3000,
    "K1-1041":         3000,
    # Simple financial
    "BALANCE_SHEET":           4096,
    "INCOME_STATEMENT":        4096,
    "CASH_FLOW_STATEMENT":     3000,
    "TRIAL_BALANCE":           6000,
    # Multi-column — size scales with (accounts × columns)
    "INCOME_STATEMENT_COMPARISON": 12000,
    "BUDGET_VS_ACTUAL":            8000,
    "QB_PROFIT_LOSS":              8000,
    # Transaction-level — unbounded, use chunking
    "GENERAL_LEDGER":              8000,   # per chunk
    "ACCOUNTS_RECEIVABLE_AGING":   4096,
    "ACCOUNTS_PAYABLE_AGING":      4096,
    "QB_GENERAL_LEDGER":           8000,   # per chunk
    # Reserve
    "RESERVE_ALLOCATION":          6000,
    # Unknown
    "FINANCIAL_UNKNOWN":           8000,
}

DEFAULT_TOKEN_BUDGET = 4096


def get_token_budget(document_type: str, override: int | None = None) -> int:
    if override:
        return override
    budget = TOKEN_BUDGETS.get(document_type, DEFAULT_TOKEN_BUDGET)

    # Warn if model context might be tight
    if budget > 6000:
        import sys
        print(
            f"WARNING: {document_type} has token budget {budget}. "
            f"Ensure --max-tokens is set >= {budget} and model context "
            f"window is >= {budget + 8000} (response + prompt).",
            file=sys.stderr
        )
    return budget


def call_llm(
    client,
    model_name: str,
    system_prompt: str,
    user_text: str,
    document_type: str,
    max_tokens_override: int | None = None,
    temperature: float = 0,
    retries: int = 2,
) -> str:
    import json
    from pydantic import ValidationError

    max_tokens = get_token_budget(document_type, max_tokens_override)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content

            # Check for truncation (JSON cut off mid-object)
            if response.choices[0].finish_reason == "length":
                if attempt < retries:
                    # Bump token budget and retry
                    max_tokens = int(max_tokens * 1.5)
                    import sys
                    print(
                        f"WARNING: Response truncated (finish_reason=length). "
                        f"Retrying with max_tokens={max_tokens}.",
                        file=sys.stderr
                    )
                    continue
                else:
                    raise ValueError(
                        f"Response still truncated after {retries} retries. "
                        f"Use chunking for this document or increase --max-tokens."
                    )

            # Strip markdown fences
            clean = raw.strip()
            for fence in ("```json", "```"):
                clean = clean.removeprefix(fence)
            clean = clean.removesuffix("```").strip()

            # Validate JSON parseable
            json.loads(clean)
            return clean

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                user_text = (
                    f"Your previous response was invalid JSON: {e}\n"
                    f"Return ONLY valid JSON matching the schema. "
                    f"No explanation, no markdown fences."
                )
            else:
                raise
```

---

## Totals Verification in Python

The LLM extracts accounts and their values. The LLM also extracts labeled
subtotal rows (e.g. "Total Utilities: 33,491.92"). Post-processing verifies that
`sum(section_accounts) ≈ section_subtotal` and sets `metadata.totals_verified`.

```python
# verifier.py

from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass


ROUNDING_TOLERANCE = Decimal("0.02")   # $0.02 tolerance for OCR digit errors


@dataclass
class VerificationResult:
    verified: bool
    mismatches: list[dict]   # [{section, computed, reported, delta}]
    notes: list[str]


def verify_section_totals(extracted: dict) -> VerificationResult:
    """
    Works on any financial document schema that has:
      - sections[].accounts[].amount (or .balance)
      - sections[].section_total (the labeled total from the document)
    """
    mismatches = []
    notes = []

    sections = _get_sections(extracted)
    for section in sections:
        accounts = section.get("accounts", [])
        reported_total = section.get("section_total")
        if reported_total is None:
            continue

        # Sum account amounts — handle both 'amount' and 'balance' keys
        computed = sum(
            Decimal(str(a.get("amount") or a.get("balance") or 0))
            for a in accounts
        )
        reported = Decimal(str(reported_total))
        delta = abs(computed - reported)

        if delta > ROUNDING_TOLERANCE:
            mismatches.append({
                "section":  section.get("section_name", "unknown"),
                "computed": float(computed),
                "reported": float(reported),
                "delta":    float(delta),
            })

    # Verify top-level totals
    top_level_checks = [
        ("total_assets", "assets"),
        ("total_liabilities", "liabilities"),
        ("total_equity", "equity"),
        ("total_income", "income"),
        ("total_expenses", "expenses"),
    ]
    for total_key, section_key in top_level_checks:
        reported = extracted.get(total_key) or extracted.get(section_key, {}).get(total_key)
        computed = _sum_section(extracted.get(section_key, {}))
        if reported is not None and computed is not None:
            delta = abs(Decimal(str(computed)) - Decimal(str(reported)))
            if delta > ROUNDING_TOLERANCE:
                mismatches.append({
                    "section":  total_key,
                    "computed": float(computed),
                    "reported": float(reported),
                    "delta":    float(delta),
                })

    # Balance sheet check: total_assets == total_liabilities + total_equity
    assets = extracted.get("total_assets")
    liabilities = extracted.get("total_liabilities")
    equity = extracted.get("total_equity")
    if all(v is not None for v in [assets, liabilities, equity]):
        lhs = Decimal(str(assets))
        rhs = Decimal(str(liabilities)) + Decimal(str(equity))
        delta = abs(lhs - rhs)
        if delta > ROUNDING_TOLERANCE:
            mismatches.append({
                "section":  "balance_sheet_equation",
                "computed": float(rhs),
                "reported": float(lhs),
                "delta":    float(delta),
            })
            notes.append(
                f"Balance sheet does not balance: Assets={assets}, "
                f"Liabilities+Equity={float(rhs):.2f}, Delta={float(delta):.2f}"
            )

    return VerificationResult(
        verified=len(mismatches) == 0,
        mismatches=mismatches,
        notes=notes,
    )


def compute_derived_fields(extracted: dict, document_type: str) -> dict:
    """
    Compute fields that must NOT come from the LLM:
    - due_to_from net (balance sheet inter-fund accounts)
    - retained_earnings_calculated
    - net_income (income - expenses, verified against reported)
    - reserve net balance
    """
    derived = {}

    if document_type == "BALANCE_SHEET":
        # Retained earnings = total_equity - sum(all explicit equity accounts)
        equity = extracted.get("equity", {})
        explicit_sections = equity.get("sections", [])
        explicit_sum = sum(
            Decimal(str(a.get("balance", 0)))
            for s in explicit_sections
            for a in s.get("accounts", [])
        )
        total_equity = Decimal(str(equity.get("total_equity_reported", 0) or 0))
        derived["retained_earnings_calculated"] = float(total_equity - explicit_sum)

    if document_type in ("INCOME_STATEMENT", "INCOME_STATEMENT_COMPARISON"):
        total_income = Decimal(str(extracted.get("total_income", 0) or 0))
        total_expenses = Decimal(str(extracted.get("total_expenses", 0) or 0))
        derived["net_income_calculated"] = float(total_income - total_expenses)

    if document_type == "RESERVE_ALLOCATION":
        components = extracted.get("components", [])
        total_balance = sum(
            Decimal(str(c.get("current_balance", 0) or 0))
            for c in components
        )
        derived["total_reserve_balance_calculated"] = float(total_balance)

    return derived


def _get_sections(extracted: dict) -> list[dict]:
    sections = []
    for top_key in ("assets", "liabilities", "equity", "income", "expenses"):
        top = extracted.get(top_key, {})
        if isinstance(top, dict):
            sections.extend(top.get("sections", []))
            for sub_key in ("current_assets", "reserve_accounts", "fixed_assets",
                            "current_liabilities", "long_term_liabilities"):
                sub = top.get(sub_key)
                if isinstance(sub, dict):
                    sections.append(sub)
                elif isinstance(sub, list):
                    sections.append({"accounts": sub})
    return sections


def _sum_section(section: dict) -> float | None:
    if not section:
        return None
    accounts = []
    for key in ("accounts", "current_assets", "reserve_accounts",
                "other_assets", "current_liabilities", "sections"):
        val = section.get(key)
        if isinstance(val, list):
            if val and isinstance(val[0], dict) and ("amount" in val[0] or "balance" in val[0]):
                accounts.extend(val)
    if not accounts:
        return None
    return float(sum(
        Decimal(str(a.get("amount") or a.get("balance") or 0))
        for a in accounts
    ))
```

---

## GL Chunking — Robust Strategy

The v1 regex approach assumed `NNNN-NNNN` account numbers. The robust strategy
uses a token counter with schema-boundary detection as primary, per-page
chunking as fallback.

```python
# chunker.py

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_index: int
    total_chunks: int
    text: str
    account_context: str | None   # account name/number this chunk starts with


# Rough chars-per-token estimate for financial text (mostly numbers + short words)
CHARS_PER_TOKEN = 3.5


def chunk_for_llm(
    text: str,
    document_type: str,
    max_input_tokens: int = 6000,
) -> list[TextChunk]:
    """
    Split text for LLM processing.
    Strategy:
      1. Try schema-boundary split (account sections for GL)
      2. Fall back to per-page split (--- PAGE BREAK --- markers)
      3. Fall back to fixed-size chunks with overlap
    """
    max_chars = int(max_input_tokens * CHARS_PER_TOKEN)

    # Short enough to fit in one call — no chunking needed
    if len(text) <= max_chars:
        return [TextChunk(0, 1, text, None)]

    if document_type in ("GENERAL_LEDGER", "QB_GENERAL_LEDGER"):
        chunks = _chunk_by_account_boundary(text, max_chars)
        if chunks:
            return chunks

    # Per-page split (extractor inserts --- PAGE BREAK --- markers)
    chunks = _chunk_by_page_break(text, max_chars)
    if chunks:
        return chunks

    # Fixed-size with 10% overlap as final fallback
    return _chunk_fixed(text, max_chars, overlap=0.10)


def _chunk_by_account_boundary(text: str, max_chars: int) -> list[TextChunk]:
    """
    Detect account section starts. Works with any account number format:
    - NNNN-NNNN (AppFolio, QuickBooks)
    - NNNN (simple 4-digit)
    - alphanumeric (Yardi)
    Uses heuristic: a line that starts with a number or code followed by
    a capitalized account name, preceded by a blank line.
    """
    # Pattern: blank line, then line starting with account-like prefix
    section_starts = [
        m.start() for m in re.finditer(
            r'\n\n(?=[\d\w]{4,12}[-\s]\s*[A-Z])',
            text
        )
    ]

    if len(section_starts) < 2:
        return []   # Not enough boundaries found — caller will use fallback

    chunks = []
    chunk_start = 0
    chunk_text = ""
    account_context = None

    for boundary in section_starts:
        segment = text[chunk_start:boundary]
        if len(chunk_text) + len(segment) > max_chars and chunk_text:
            chunks.append(TextChunk(
                chunk_index=len(chunks),
                total_chunks=0,  # filled in after
                text=chunk_text.strip(),
                account_context=account_context,
            ))
            chunk_text = segment
            chunk_start = boundary
            # Extract account context from first line of new chunk
            first_line = segment.strip().split("\n")[0] if segment.strip() else ""
            account_context = first_line[:80]
        else:
            chunk_text += segment
            if account_context is None:
                first_line = segment.strip().split("\n")[0] if segment.strip() else ""
                account_context = first_line[:80]

    # Final chunk
    remaining = text[chunk_start:]
    if remaining.strip():
        chunk_text += remaining
        chunks.append(TextChunk(
            chunk_index=len(chunks),
            total_chunks=0,
            text=chunk_text.strip(),
            account_context=account_context,
        ))

    # Fill in total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    return chunks


def _chunk_by_page_break(text: str, max_chars: int) -> list[TextChunk]:
    """Split on --- PAGE BREAK --- markers inserted by extractor."""
    pages = re.split(r'\n---\s*PAGE BREAK\s*---\n', text)
    if len(pages) < 2:
        return []

    chunks = []
    current = ""
    for page in pages:
        if len(current) + len(page) > max_chars and current:
            chunks.append(TextChunk(len(chunks), 0, current.strip(), None))
            current = page
        else:
            current += "\n" + page
    if current.strip():
        chunks.append(TextChunk(len(chunks), 0, current.strip(), None))

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    return chunks


def _chunk_fixed(text: str, max_chars: int, overlap: float) -> list[TextChunk]:
    """Fixed-size chunks with overlap. Last resort."""
    step = int(max_chars * (1 - overlap))
    raw_chunks = [text[i:i + max_chars] for i in range(0, len(text), step)]
    total = len(raw_chunks)
    return [
        TextChunk(i, total, chunk, None)
        for i, chunk in enumerate(raw_chunks)
    ]
```

---

## CSV Output Spec

Financial documents have two fundamentally different CSV shapes:

**Shape A — Account rows** (Balance Sheet, P&L, Trial Balance, Budget vs Actual):
One row per account line item. Columns are metadata + one value column per period.

**Shape B — Transaction rows** (GL, AR/AP Aging, Transaction List):
One row per transaction. Columns are transaction fields.

```python
# formatter.py  — CSV section

import csv
import io
from typing import Any


def to_csv(extracted: dict, document_type: str) -> str:
    """
    Route to the correct CSV shape based on document type.
    Returns CSV string.
    """
    if document_type in ("GENERAL_LEDGER", "QB_GENERAL_LEDGER",
                         "QB_TRANSACTION_LIST"):
        return _csv_transactions(extracted)
    elif document_type in ("ACCOUNTS_RECEIVABLE_AGING", "ACCOUNTS_PAYABLE_AGING",
                           "QB_AR_AGING", "QB_AP_AGING"):
        return _csv_aging(extracted)
    else:
        return _csv_account_rows(extracted, document_type)


def _csv_account_rows(extracted: dict, document_type: str) -> str:
    """
    Shape A: one row per account.

    Fixed columns (always present):
      document_type, entity_name, software, accounting_basis,
      period_start, period_end,
      section, subsection, account_number, account_name,
      row_type (account | subtotal | total)

    Dynamic columns (one per period/column in the report):
      For single-period: amount
      For multi-period: [col_key_1, col_key_2, ...] from extracted.columns
    """
    output = io.StringIO()

    # Detect column keys
    period_columns = _get_period_columns(extracted, document_type)

    fixed_headers = [
        "document_type", "entity_name", "software", "accounting_basis",
        "period_start", "period_end",
        "section", "subsection", "account_number", "account_name", "row_type",
    ]
    headers = fixed_headers + period_columns
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()

    entity = extracted.get("entity", {})
    base_row = {
        "document_type":    document_type,
        "entity_name":      entity.get("name", ""),
        "software":         entity.get("software", ""),
        "accounting_basis": entity.get("accounting_basis", ""),
        "period_start":     entity.get("period_start") or entity.get("period_end", ""),
        "period_end":       entity.get("period_end", ""),
    }

    # Walk sections
    for top_section_name, top_section in _iter_top_sections(extracted):
        _write_section_rows(
            writer, base_row, top_section_name, top_section,
            period_columns, subsection=""
        )

    return output.getvalue()


def _write_section_rows(writer, base_row, section_name, section_data,
                        period_columns, subsection="", depth=0):
    """Recursively write account rows, handling nested sections."""
    if isinstance(section_data, list):
        # Direct list of accounts
        for account in section_data:
            if not isinstance(account, dict):
                continue
            row = {**base_row,
                   "section": section_name,
                   "subsection": subsection,
                   "account_number": account.get("account_number", ""),
                   "account_name":   account.get("account_name", ""),
                   "row_type": "account"}
            _fill_period_values(row, account, period_columns)
            writer.writerow(row)
        return

    if not isinstance(section_data, dict):
        return

    # Nested sections
    sub_sections = section_data.get("sections", [])
    for sub in sub_sections:
        sub_name = sub.get("section_name", "")
        _write_section_rows(
            writer, base_row, section_name, sub.get("accounts", []),
            period_columns, subsection=sub_name, depth=depth+1
        )
        # Write subtotal row
        sub_total = sub.get("section_total")
        if sub_total is not None:
            row = {**base_row,
                   "section": section_name,
                   "subsection": sub_name,
                   "account_number": "",
                   "account_name": f"Total {sub_name}",
                   "row_type": "subtotal"}
            if period_columns == ["amount"]:
                row["amount"] = sub_total
            writer.writerow(row)

    # Direct accounts at this level
    accounts = section_data.get("accounts", [])
    for account in accounts:
        if not isinstance(account, dict):
            continue
        row = {**base_row,
               "section": section_name,
               "subsection": subsection,
               "account_number": account.get("account_number", ""),
               "account_name":   account.get("account_name", ""),
               "row_type": "account"}
        _fill_period_values(row, account, period_columns)
        writer.writerow(row)

    # Section-level total
    for total_key in ("section_total", f"{section_name.lower()}_total",
                      "current_assets_total", "reserve_accounts_total",
                      "current_liabilities_total", "total_assets",
                      "total_liabilities", "total_equity", "total_income",
                      "total_expenses"):
        val = section_data.get(total_key)
        if val is not None:
            row = {**base_row,
                   "section": section_name,
                   "subsection": subsection,
                   "account_number": "",
                   "account_name": total_key.replace("_", " ").title(),
                   "row_type": "total"}
            if period_columns == ["amount"]:
                row["amount"] = val
            writer.writerow(row)


def _fill_period_values(row: dict, account: dict, period_columns: list[str]):
    for col in period_columns:
        row[col] = account.get(col, "")


def _get_period_columns(extracted: dict, document_type: str) -> list[str]:
    """Detect period column keys from extracted data."""
    # Multi-column report: columns are explicitly listed
    if "columns" in extracted:
        return [c["key"] for c in extracted["columns"]]
    # Single-period: just "amount" or "balance"
    if document_type in ("BALANCE_SHEET", "RESERVE_ALLOCATION"):
        return ["balance"]
    return ["amount"]


def _iter_top_sections(extracted: dict):
    """Yield (section_name, section_data) for all top-level financial sections."""
    for key in ("assets", "liabilities", "equity", "income", "expenses",
                "other_income", "components", "line_items"):
        if key in extracted:
            yield key.upper(), extracted[key]


def _csv_transactions(extracted: dict) -> str:
    """Shape B: GL / transaction list — one row per transaction."""
    output = io.StringIO()
    headers = [
        "entity_name", "account_number", "account_name",
        "date", "type", "number", "name", "memo", "split",
        "debit", "credit", "balance", "row_type",
    ]
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()

    entity = extracted.get("entity", {})
    entity_name = entity.get("name", "")

    for account in extracted.get("accounts", []):
        acct_num  = account.get("account_number", "")
        acct_name = account.get("account_name", "")
        # Beginning balance
        writer.writerow({
            "entity_name": entity_name,
            "account_number": acct_num,
            "account_name": acct_name,
            "balance": account.get("beginning_balance", ""),
            "row_type": "balance_header",
        })
        for txn in account.get("transactions", []):
            writer.writerow({
                "entity_name":    entity_name,
                "account_number": acct_num,
                "account_name":   acct_name,
                "date":           txn.get("date", ""),
                "type":           txn.get("type", ""),
                "number":         txn.get("number", ""),
                "name":           txn.get("name", ""),
                "memo":           txn.get("memo", ""),
                "split":          txn.get("split", ""),
                "debit":          txn.get("debit", ""),
                "credit":         txn.get("credit", ""),
                "balance":        txn.get("balance", ""),
                "row_type":       txn.get("row_type", "transaction"),
            })
        writer.writerow({
            "entity_name": entity_name,
            "account_number": acct_num,
            "account_name": acct_name,
            "balance": account.get("ending_balance", ""),
            "row_type": "balance_footer",
        })

    return output.getvalue()


def _csv_aging(extracted: dict) -> str:
    """Shape B variant: AR/AP Aging — one row per customer/vendor."""
    output = io.StringIO()
    buckets = extracted.get("aging_buckets",
                            ["current", "1_to_30", "31_to_60", "61_to_90", "over_90"])
    headers = (["entity_name", "customer_name"] + buckets +
               ["total", "row_type"])
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()

    entity_name = extracted.get("entity", {}).get("name", "")
    for customer in extracted.get("customers", []):
        row = {"entity_name": entity_name,
               "customer_name": customer.get("customer_name", ""),
               "total": customer.get("total", ""),
               "row_type": "customer"}
        for b in buckets:
            row[b] = customer.get(b, "")
        writer.writerow(row)

    # Totals row
    totals = extracted.get("totals", {})
    if totals:
        row = {"entity_name": entity_name,
               "customer_name": "TOTAL",
               "total": totals.get("total", ""),
               "row_type": "total"}
        for b in buckets:
            row[b] = totals.get(b, "")
        writer.writerow(row)

    return output.getvalue()
```

### CSV output examples

**Balance Sheet (Shape A):**
```
document_type,entity_name,software,accounting_basis,period_start,period_end,section,subsection,account_number,account_name,row_type,balance
BALANCE_SHEET,PMG - Niguel Villas,AppFolio,Cash,2025-01-31,2025-01-31,ASSETS,Cash,1018-0000,SUNWEST BANK-OPERATING,account,158678.65
BALANCE_SHEET,PMG - Niguel Villas,AppFolio,Cash,2025-01-31,2025-01-31,ASSETS,Cash,1021-0000,SUNWEST BANK-RESERVE,account,328402.40
BALANCE_SHEET,PMG - Niguel Villas,AppFolio,Cash,2025-01-31,2025-01-31,ASSETS,Cash,,Total Cash,subtotal,487081.05
```

**12-Month P&L Comparison (Shape A, multi-column):**
```
document_type,entity_name,software,accounting_basis,period_start,period_end,section,subsection,account_number,account_name,row_type,jul_2024_actual,aug_2024_actual,...,ytd_actual,ytd_budget,ytd_dollar_var,annual_budget
INCOME_STATEMENT_COMPARISON,PMG,AppFolio,Cash,2024-07-01,2025-01-31,INCOME,ASSESSMENT,4000-0000,ASSESSMENT,account,101379.00,101883.00,...,712116.00,714000.00,-1884.00,1222960.00
```

**GL (Shape B):**
```
entity_name,account_number,account_name,date,type,number,name,memo,split,debit,credit,balance,row_type
Acme Corp,1000,Checking,,,,,,,,,32000.00,balance_header
Acme Corp,1000,Checking,2025-01-05,Check,1042,Office Depot,Office supplies,Office Supplies,245.00,,31755.00,transaction
Acme Corp,1000,Checking,2025-01-10,Deposit,,Customer ABC,Invoice 1021,Accounts Receivable,,5000.00,36755.00,transaction
Acme Corp,1000,Checking,,,,,,,,,36755.00,balance_footer
```

---

## Test Fixture Requirements

Before claiming a document type is working, you need at least one real sample
per software family. The sample document in this repo
(`01_-_Jan_2025_Financials_Owner.pdf`) covers AppFolio with encoding-broken PDFs.

Minimum fixture set needed to validate all code paths:

| Fixture | Software | Doc Type | Encoding | Pages | Status |
|---|---|---|---|---|---|
| `01_-_Jan_2025_Financials_Owner.pdf` | AppFolio | Balance Sheet + P&L Comparison | Broken (PScript5) | 9 | ✅ Have |
| `qb_desktop_pl.pdf` | QuickBooks Desktop | P&L Standard | Clean or broken | 2-5 | ❌ Need |
| `qb_desktop_gl.pdf` | QuickBooks Desktop | General Ledger | Clean or broken | 10+ | ❌ Need |
| `qb_online_bs.pdf` | QuickBooks Online | Balance Sheet | Clean (browser) | 2-3 | ❌ Need |
| `qb_online_ar_aging.pdf` | QuickBooks Online | AR Aging | Clean (browser) | 2-3 | ❌ Need |
| `yardi_income.pdf` | Yardi | Income Statement | Often broken | 3-5 | ❌ Need |
| `appfolio_owner_stmt.pdf` | AppFolio | Owner Statement | Often broken | 2-4 | ❌ Need |
| `sage_trial_balance.pdf` | Sage 100/300 | Trial Balance | Often broken | 5-10 | ❌ Need |

To generate test fixtures from QuickBooks Desktop without real client data:
use QuickBooks sample company file (Help → Use Sample Company) and export
each report type via both File → Save as PDF (clean) and print to PDF via
the Windows print dialog (broken). Both paths need test coverage.

For AppFolio: demo accounts can be requested at appfolio.com/request-a-demo.
The encoding issue reproduces on any AppFolio report printed through their
PDF export flow on Windows.

---

## Updated CLI Flags

Add to `cli.py`:

```
--no-fix-orientation    Disable auto-rotation correction (default: enabled)
--chunk-size            Max input tokens per LLM chunk for long docs (default: 6000)
--verify-totals         Run Python totals verifier after extraction (default: true)
--no-verify-totals      Skip totals verification
--family                Force document family: tax | financial_simple |
                        financial_multi | financial_txn | financial_reserve
                        (default: auto-detect)
```
