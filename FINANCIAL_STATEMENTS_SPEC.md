# w2extract — Financial Statements & QuickBooks Addendum

Append this spec to the main README. Covers encoding-broken PDF detection and
fallback OCR, all standard financial statement types, QuickBooks-specific exports,
and property management software exports (AppFolio, Buildium, Yardi, PMG-style).

---

## Table of Contents

- [The Encoding Problem](#the-encoding-problem)
  - [What Breaks and Why](#what-breaks-and-why)
  - [Detection Heuristics](#detection-heuristics)
  - [Fallback OCR Pipeline](#fallback-ocr-pipeline)
  - [Upside-Down Page Detection](#upside-down-page-detection)
- [Supported Financial Document Types](#supported-financial-document-types)
- [Document Type Detection](#document-type-detection)
- [JSON Schema Reference — Financial Statements](#json-schema-reference--financial-statements)
- [QuickBooks-Specific Export Quirks](#quickbooks-specific-export-quirks)
- [Property Management Software Exports](#property-management-software-exports)
- [LLM Prompting for Financial Documents](#llm-prompting-for-financial-documents)
- [Model Recommendations](#model-recommendations-for-financial-documents)

---

## The Encoding Problem

### What Breaks and Why

The sample document (`01_-_Jan_2025_Financials_Owner.pdf`) is a perfect example
of the most common encoding failure you will encounter in practice. Running
`pdffonts` reveals the root cause:

```
name                      type        encoding    emb  sub  uni
------------------------- ----------- ----------- ---  ---  ---
NGDLJE+Arial-BoldMT       CID TrueType Identity-H yes  yes  no
NGDLKF+ArialMT            CID TrueType Identity-H yes  yes  no
NGDLMF+ArialNarrow        CID TrueType Identity-H yes  yes  no
```

The critical column is `uni = no`. This means the font has **no ToUnicode map**.
The PDF contains glyph IDs, not Unicode codepoints. `pdfminer` and `pdftotext`
both extract the raw glyph IDs as control characters — the text is there but
completely unreadable as strings.

**Creator:** `PScript5.dll Version 5.2.2` (Windows PostScript printer driver)
**Producer:** `Acrobat Distiller 25.0`

This is a "print to PDF" workflow — the source application (AppFolio, in this case)
printed to a PostScript file and Distiller converted it. Distiller subsets the font
but does not embed a ToUnicode table when the source font uses Identity-H encoding
without explicit Unicode mapping. This is extremely common with:

- AppFolio financial exports
- Yardi financial exports  
- Sage 100/300 reports
- Some QuickBooks Desktop "print to PDF" reports
- Any Windows app using PScript5 + Distiller

**The fix is always the same: rasterize and OCR.** You cannot recover the text
layer. Even though `pdfminer` extracts text (it's not empty), the output is
garbage. Your standard heuristic of `text length < 100 chars` will NOT catch this
because the garbled text IS long. You need a secondary heuristic.

---

### Detection Heuristics

Add these to `detector.py` as a pipeline of checks, in order:

```python
import re
import unicodedata
from pdfminer.high_level import extract_text
import subprocess

def detect_encoding_broken(pdf_path: str) -> dict:
    """
    Returns dict with keys:
      - is_broken: bool
      - reason: str
      - strategy: 'text' | 'ocr' | 'vision'
    """

    # --- Check 1: pdffonts ToUnicode flag (most reliable) ---
    # Run pdffonts and check if any font has uni=no
    try:
        result = subprocess.run(
            ["pdffonts", pdf_path],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")[2:]  # skip headers
        for line in lines:
            cols = line.split()
            if len(cols) >= 7 and cols[6] == "no":
                return {
                    "is_broken": True,
                    "reason": f"Font '{cols[0]}' has no ToUnicode map (Identity-H encoding). "
                              f"Created by print-to-PDF workflow (PScript5/Distiller). "
                              f"Text layer is glyph IDs, not Unicode. Must OCR.",
                    "strategy": "ocr"
                }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # pdffonts not available, fall through to text heuristics

    # --- Check 2: High ratio of non-printable / non-ASCII characters ---
    try:
        sample = extract_text(pdf_path, maxpages=2)
        if len(sample) > 50:
            non_printable = sum(
                1 for c in sample
                if unicodedata.category(c) in ('Cc', 'Cs', 'Co', 'Cn')
                or (ord(c) < 32 and c not in '\n\t\r')
            )
            ratio = non_printable / len(sample)
            if ratio > 0.15:
                return {
                    "is_broken": True,
                    "reason": f"High non-printable character ratio ({ratio:.1%}). "
                              f"Likely font encoding issue.",
                    "strategy": "ocr"
                }
    except Exception:
        pass

    # --- Check 3: Text present but no recognizable words ---
    # If we have text but can't find any common English words, it's garbage
    try:
        sample = extract_text(pdf_path, maxpages=1).lower()
        common_words = {'the', 'and', 'total', 'balance', 'account', 'income',
                        'expenses', 'assets', 'wages', 'tax', 'amount', 'date',
                        'name', 'number', 'net', 'gross', 'paid', 'year'}
        found = sum(1 for w in common_words if w in sample)
        if len(sample) > 200 and found < 2:
            return {
                "is_broken": True,
                "reason": "Text layer present but contains no recognizable words. "
                          "Likely encoding corruption.",
                "strategy": "ocr"
            }
    except Exception:
        pass

    # --- Check 4: Per-page text length (existing heuristic, kept for scanned PDFs) ---
    # A scanned PDF has no text layer at all, not garbled text
    try:
        sample = extract_text(pdf_path, maxpages=1)
        if len(sample.strip()) < 100:
            return {
                "is_broken": False,
                "reason": "No text layer — scanned/image PDF.",
                "strategy": "ocr"
            }
    except Exception:
        pass

    return {"is_broken": False, "reason": "Text layer OK", "strategy": "text"}


def detect_per_page(pdf_path: str) -> list[dict]:
    """
    Per-page detection for mixed PDFs.
    Returns list of {page: int, strategy: 'text'|'ocr'|'vision', reason: str}
    """
    from pdfminer.high_level import extract_pages
    results = []
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        text = "".join(
            element.get_text() for element in page_layout
            if hasattr(element, 'get_text')
        )
        non_printable = sum(
            1 for c in text
            if unicodedata.category(c) in ('Cc',) and c not in '\n\t\r'
        )
        ratio = non_printable / max(len(text), 1)
        if ratio > 0.15 or len(text.strip()) < 50:
            results.append({"page": i, "strategy": "ocr",
                            "reason": f"Encoding broken or no text (ratio={ratio:.1%})"})
        else:
            results.append({"page": i, "strategy": "text", "reason": "OK"})
    return results
```

---

### Fallback OCR Pipeline

Add to `ocr.py`:

```python
import os
import shutil
import tempfile
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np

def ocr_pdf(
    pdf_path: str,
    engine: str = "auto",      # auto | tesseract | easyocr | paddleocr
    dpi: int = 300,
    gpu: bool | str = "auto",
    fix_orientation: bool = True,
) -> list[dict]:
    """
    Rasterize PDF and OCR each page.
    Returns list of {page: int, text: str, confidence: float | None, rotated: bool}
    """
    tmpdir = tempfile.mkdtemp()
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            output_folder=tmpdir,
            fmt="png",
            thread_count=os.cpu_count(),
        )

        results = []
        for i, page_img in enumerate(pages):
            # Orientation detection and correction
            rotated = False
            if fix_orientation:
                page_img, rotated = correct_orientation(page_img)

            if engine == "auto":
                engine = _select_engine(gpu)

            if engine == "tesseract":
                text, confidence = _ocr_tesseract(page_img)
            elif engine == "easyocr":
                text, confidence = _ocr_easyocr(page_img, gpu)
            elif engine == "paddleocr":
                text, confidence = _ocr_paddle(page_img, gpu)
            else:
                raise ValueError(f"Unknown OCR engine: {engine}")

            results.append({
                "page": i,
                "text": text,
                "confidence": confidence,
                "rotated": rotated,
            })

            if confidence is not None and confidence < 0.6:
                import sys
                print(f"WARNING: Page {i+1} OCR confidence low ({confidence:.1%}). "
                      f"Consider --vision flag for this document.", file=sys.stderr)

        return results
    finally:
        shutil.rmtree(tmpdir)  # wipe PNG intermediates — contains PII


def _select_engine(gpu) -> str:
    try:
        import torch
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            return "easyocr"
    except ImportError:
        pass
    return "tesseract"


def _ocr_tesseract(img: Image.Image) -> tuple[str, None]:
    text = pytesseract.image_to_string(img, config="--psm 6")
    return text, None


def _ocr_easyocr(img: Image.Image, gpu) -> tuple[str, float]:
    import easyocr, torch
    use_gpu = torch.cuda.is_available() if gpu == "auto" else gpu is True
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    results = reader.readtext(np.array(img), detail=1)
    texts = [r[1] for r in results]
    confidences = [r[2] for r in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return "\n".join(texts), avg_conf


def _ocr_paddle(img: Image.Image, gpu) -> tuple[str, float]:
    import paddle
    from paddleocr import PaddleOCR
    use_gpu = paddle.device.is_compiled_with_cuda() if gpu == "auto" else gpu is True
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)
    result = ocr.ocr(np.array(img), cls=True)
    if not result or not result[0]:
        return "", 0.0
    lines = result[0]
    texts = [line[1][0] for line in lines]
    confs = [line[1][1] for line in lines]
    return "\n".join(texts), sum(confs) / len(confs)
```

---

### Upside-Down Page Detection

This document has pages that are physically rotated 180 degrees — the OCR output
on page 3 shows `yoBpng jenuuy` which is `Annual Budget` upside down, and
account numbers like `0000-0SS8` which are `8SS0-0000` reversed.

This is common in multi-page financial reports where some pages (budget comparison,
prior year columns) were generated in landscape and then rotated during PDF assembly.

```python
def correct_orientation(img: Image.Image) -> tuple[Image.Image, bool]:
    """
    Detect and correct upside-down or rotated pages using Tesseract OSD.
    Returns (corrected_image, was_rotated).
    """
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        confidence = osd.get("orientation_conf", 0)

        if confidence > 1.5 and angle != 0:
            # PIL rotate is counter-clockwise; Tesseract reports clockwise needed
            corrected = img.rotate(-angle, expand=True)
            return corrected, True
        return img, False
    except Exception:
        return img, False
```

Add to CLI flags:
```
--no-fix-orientation    Disable auto-rotation correction (default: enabled)
```

---

## Supported Financial Document Types

| Type | Description | Common Sources |
|---|---|---|
| `BALANCE_SHEET` | Assets, liabilities, equity at a point in time | QB, AppFolio, Yardi, Sage |
| `INCOME_STATEMENT` | Revenue and expenses over a period (P&L) | QB, AppFolio, Yardi, Sage |
| `INCOME_STATEMENT_COMPARISON` | P&L with multi-period columns | QB, AppFolio, PMG reports |
| `INCOME_STATEMENT_BUDGET` | P&L with budget vs actual columns | AppFolio, Yardi |
| `CASH_FLOW_STATEMENT` | Operating/investing/financing cash flows | QB |
| `TRIAL_BALANCE` | All GL accounts with debit/credit totals | QB, Sage |
| `GENERAL_LEDGER` | Transaction-level detail by account | QB, AppFolio |
| `ACCOUNTS_RECEIVABLE_AGING` | AR by age bucket | QB, AppFolio |
| `ACCOUNTS_PAYABLE_AGING` | AP by age bucket | QB |
| `BUDGET_VS_ACTUAL` | Budget comparison report | QB, AppFolio, Yardi |
| `RESERVE_STUDY` | HOA/condo reserve fund schedule | PMG, AppFolio |
| `RESERVE_ALLOCATION` | Reserve fund balance by component | AppFolio, Buildium |
| `BANK_RECONCILIATION` | Book vs bank balance reconciliation | QB |
| `CHART_OF_ACCOUNTS` | Account list with numbers and types | QB, AppFolio |
| `TRANSACTION_DETAIL` | Line-item transactions by account | QB, AppFolio |
| `VENDOR_SUMMARY` | Payments by vendor | QB |
| `QB_PROFIT_LOSS` | QuickBooks P&L (standard layout) | QuickBooks Desktop/Online |
| `QB_BALANCE_SHEET` | QuickBooks Balance Sheet | QuickBooks Desktop/Online |
| `QB_AR_AGING` | QuickBooks AR Aging Summary/Detail | QuickBooks Desktop/Online |
| `QB_AP_AGING` | QuickBooks AP Aging Summary/Detail | QuickBooks Desktop/Online |
| `QB_TRANSACTION_LIST` | QuickBooks Transaction List by Date | QuickBooks Desktop/Online |
| `QB_JOURNAL` | QuickBooks Journal report | QuickBooks Desktop/Online |
| `QB_GENERAL_LEDGER` | QuickBooks General Ledger | QuickBooks Desktop/Online |
| `APPFOLIO_INCOME_STMT` | AppFolio Income Statement (owner report) | AppFolio |
| `APPFOLIO_BALANCE_SHEET` | AppFolio Balance Sheet | AppFolio |
| `APPFOLIO_OWNER_STATEMENT` | AppFolio Owner Statement (cash basis) | AppFolio |
| `YARDI_INCOME_STMT` | Yardi Income Statement | Yardi Voyager/Breeze |

---

## Document Type Detection

Add to `detector.py`. Financial document detection runs after the encoding check,
on clean OCR or text-layer output:

```python
FINANCIAL_SIGNATURES = {
    # Balance Sheet signals
    "BALANCE_SHEET": [
        r"balance sheet",
        r"total assets",
        r"liabilities\s+[&and]+\s+(?:capital|equity|stockholders)",
        r"current assets",
        r"accounts receivable",
    ],
    # Income Statement / P&L
    "INCOME_STATEMENT": [
        r"(?:income|profit)\s+(?:and\s+loss|statement|p&l)",
        r"total (?:income|revenue)",
        r"total expenses",
        r"net (?:income|loss|operating)",
        r"operating income",
    ],
    # Multi-column comparison
    "INCOME_STATEMENT_COMPARISON": [
        r"(?:ytd|year.to.date)\s+(?:actual|budget)",
        r"(?:current|this)\s+(?:month|period)\s+(?:actual|budget)",
        r"(?:prior year|last year)\s+(?:actual|budget)",
        r"\$\s+var(?:iance)?",
        r"% var(?:iance)?",
    ],
    # Budget vs Actual
    "BUDGET_VS_ACTUAL": [
        r"annual budget",
        r"ytd budget",
        r"ytd actual",
        r"\$\s+var(?:iance)?",
    ],
    # Trial Balance
    "TRIAL_BALANCE": [
        r"trial balance",
        r"debit\s+credit",
        r"total debits",
    ],
    # General Ledger
    "GENERAL_LEDGER": [
        r"general ledger",
        r"transaction detail",
        r"(?:date|memo|ref)\s+(?:debit|credit)\s+balance",
    ],
    # AR/AP Aging
    "ACCOUNTS_RECEIVABLE_AGING": [
        r"a(?:ccounts)?\s*r(?:eceivable)?\s*aging",
        r"current\s+(?:1-30|0-30)\s+(?:31-60|30-60)",
    ],
    "ACCOUNTS_PAYABLE_AGING": [
        r"a(?:ccounts)?\s*p(?:ayable)?\s*aging",
    ],
    # Reserve allocation (HOA/condo)
    "RESERVE_ALLOCATION": [
        r"reserve allocation",
        r"reserve fund",
        r"(?:component|item)\s+(?:cost|balance|funded)",
        r"(?:tile|shake)\s+roof",
        r"asphalt replacement",
        r"contingency",
    ],
    # AppFolio-specific
    "APPFOLIO_OWNER_STATEMENT": [
        r"appfolio",
        r"owner statement",
        r"owner draw",
        r"management fee",
    ],
    # QuickBooks-specific
    "QB_PROFIT_LOSS": [
        r"quickbooks",
        r"intuit",
        r"profit\s+[&and]+\s+loss",
    ],
}

def detect_financial_document_type(text: str) -> str:
    """
    Returns the best-matching financial document type, or 'FINANCIAL_UNKNOWN'.
    Checks INCOME_STATEMENT_COMPARISON before INCOME_STATEMENT (more specific first).
    """
    import re
    text_lower = text.lower()

    # Priority order — more specific types first
    priority = [
        "INCOME_STATEMENT_COMPARISON",
        "BUDGET_VS_ACTUAL",
        "APPFOLIO_OWNER_STATEMENT",
        "QB_PROFIT_LOSS",
        "RESERVE_ALLOCATION",
        "ACCOUNTS_RECEIVABLE_AGING",
        "ACCOUNTS_PAYABLE_AGING",
        "GENERAL_LEDGER",
        "TRIAL_BALANCE",
        "BALANCE_SHEET",
        "INCOME_STATEMENT",
    ]

    scores = {}
    for doc_type in priority:
        patterns = FINANCIAL_SIGNATURES.get(doc_type, [])
        score = sum(1 for p in patterns if re.search(p, text_lower))
        scores[doc_type] = score

    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return best
    if scores[best] == 1:
        return best  # single signal — flag as low-confidence in metadata
    return "FINANCIAL_UNKNOWN"
```

---

## JSON Schema Reference — Financial Statements

All financial documents share the top-level wrapper from the main README with
`document_type` set to one of the types above.

---

### BALANCE_SHEET

```json
{
  "entity": {
    "name": "PMG - Niguel Villas",
    "type": "HOA",
    "accounting_basis": "Cash",
    "period_end": "2025-01-31",
    "prepared_by": null,
    "software": "AppFolio"
  },
  "assets": {
    "current_assets": {
      "cash": [
        { "account_number": "1018-0000", "account_name": "SUNWEST BANK-OPERATING", "balance": 158678.65 },
        { "account_number": "1021-0000", "account_name": "SUNWEST BANK-RESERVE",   "balance": 328402.40 }
      ],
      "cash_total": 487081.05,
      "accounts_receivable": [],
      "accounts_receivable_total": 0.0,
      "other_current_assets": [
        { "account_number": "1200-0000", "account_name": "DELINQUENT ASSESSMENTS", "balance": 4137.24 }
      ],
      "other_current_assets_total": 4137.24,
      "current_assets_total": 491218.29
    },
    "reserve_accounts": [
      { "account_number": "1025-0003", "account_name": "MERRILL LYNCH-NIGUEL VILLAS #3196", "balance": 548916.83 }
    ],
    "reserve_accounts_total": 548916.83,
    "fixed_assets": [],
    "fixed_assets_total": 0.0,
    "other_assets": [],
    "other_assets_total": 0.0,
    "total_assets": 1040135.12
  },
  "liabilities": {
    "current_liabilities": [
      { "account_number": "2025-0000", "account_name": "PREPAID ASSESSMENTS", "balance": 42311.84 }
    ],
    "current_liabilities_total": 42311.84,
    "long_term_liabilities": [],
    "long_term_liabilities_total": 0.0,
    "total_liabilities": 42311.84
  },
  "equity": {
    "sections": [
      {
        "section_name": "RESERVE ALLOCATION",
        "accounts": [
          { "account_number": "3001-0000", "account_name": "ACCESS GATES",        "balance": 7693.95 },
          { "account_number": "3003-0000", "account_name": "LIGHTING",            "balance": 68023.94 },
          { "account_number": "3011-0000", "account_name": "PAINTING",            "balance": -147545.64 },
          { "account_number": "3015-0000", "account_name": "TILE/SHAKE ROOF",     "balance": 694334.24 },
          { "account_number": "3021-0000", "account_name": "ASPHALT REPLACEMENT", "balance": 134781.17 }
        ],
        "section_total": 1104435.34
      },
      {
        "section_name": "EQUITY",
        "accounts": [
          { "account_number": "3998-0000", "account_name": "EQUITY-BEGIN OF YEAR",          "balance": -252184.23 },
          { "account_number": "3999-0000", "account_name": "CURRENT YR INCREASE/DECREASE",  "balance": -232.96 },
          { "account_number": "3999-0001", "account_name": "Appfolio Opening Balance Equity","balance": -538476.87 }
        ],
        "section_total": -790894.06
      }
    ],
    "retained_earnings_calculated": 106827.35,
    "prior_years_retained_earnings_calculated": 577454.65,
    "total_equity": 997823.28
  },
  "total_liabilities_and_equity": 1040135.12,
  "check_difference": 0.00
}
```

---

### INCOME_STATEMENT

```json
{
  "entity": {
    "name": "PMG",
    "accounting_basis": "Cash",
    "period_start": "2025-01-01",
    "period_end": "2025-01-31",
    "software": "AppFolio"
  },
  "income": {
    "sections": [
      {
        "section_name": "ASSESSMENT",
        "accounts": [
          { "account_number": "4000-0000", "account_name": "ASSESSMENT", "amount": 101379.00 }
        ],
        "section_total": 101379.00
      },
      {
        "section_name": "OTHER INCOME",
        "accounts": [
          { "account_number": "4060-0000", "account_name": "PARKING PERMIT",       "amount": 125.00 },
          { "account_number": "4070-0000", "account_name": "COLLECTION FEES",      "amount": 35.00 },
          { "account_number": "4080-0000", "account_name": "BANK INTEREST RESERVE","amount": 13468.15 },
          { "account_number": "4100-0000", "account_name": "LATE CHARGES",         "amount": 322.00 }
        ],
        "section_total": 13950.15
      }
    ],
    "total_income": 115329.15
  },
  "expenses": {
    "sections": [
      {
        "section_name": "UTILITIES",
        "accounts": [
          { "account_number": "5010-0000", "account_name": "ELECTRICITY", "amount": 3000.00 },
          { "account_number": "5020-0000", "account_name": "WATER",       "amount": 24358.33 },
          { "account_number": "5050-0000", "account_name": "TRASH/WASTE", "amount": 6133.59 }
        ],
        "section_total": 33491.92
      },
      {
        "section_name": "MAINTENANCE / CONTRACTS",
        "accounts": [
          { "account_number": "8010-0000", "account_name": "LIGHT MAINTENANCE",        "amount": 299.00 },
          { "account_number": "8015-0000", "account_name": "LIGHTING SUPPLIES",        "amount": -400.89 },
          { "account_number": "8025-0000", "account_name": "FOUNTAIN MAINTENANCE",     "amount": 10.00 },
          { "account_number": "8080-0000", "account_name": "COMMUNITY AREA",           "amount": 2273.69 }
        ],
        "section_total": 2181.80
      }
    ],
    "total_expenses": 115097.18
  },
  "operating_income": 231.97,
  "other_income": {
    "accounts": [
      { "account_number": "9206-0000", "account_name": "GAIN/LOSS ON INVESTMENTS", "amount": 9835.73 }
    ],
    "total_other_income": 9835.73
  },
  "net_income": 10067.70
}
```

---

### INCOME_STATEMENT_COMPARISON

Multi-column report with period columns. This is what the sample document's
income statement pages contain — July 2024 through January 2025 plus YTD
columns and annual budget.

```json
{
  "entity": {
    "name": "PMG",
    "fund_type": "All",
    "accounting_basis": "Cash",
    "period_range": { "start": "2024-07-01", "end": "2025-01-31" },
    "software": "AppFolio",
    "report_title": "Income Statement - 12 Month - PMG"
  },
  "columns": [
    { "key": "jul_2024_actual",  "label": "Jul 2024 Actual",  "period_start": "2024-07-01", "period_end": "2024-07-31" },
    { "key": "aug_2024_actual",  "label": "Aug 2024 Actual",  "period_start": "2024-08-01", "period_end": "2024-08-31" },
    { "key": "sep_2024_actual",  "label": "Sep 2024 Actual",  "period_start": "2024-09-01", "period_end": "2024-09-30" },
    { "key": "oct_2024_actual",  "label": "Oct 2024 Actual",  "period_start": "2024-10-01", "period_end": "2024-10-31" },
    { "key": "nov_2024_actual",  "label": "Nov 2024 Actual",  "period_start": "2024-11-01", "period_end": "2024-11-30" },
    { "key": "dec_2024_actual",  "label": "Dec 2024 Actual",  "period_start": "2024-12-01", "period_end": "2024-12-31" },
    { "key": "jan_2025_actual",  "label": "Jan 2025 Actual",  "period_start": "2025-01-01", "period_end": "2025-01-31" },
    { "key": "ytd_actual",       "label": "YTD Actual",       "period_start": "2024-07-01", "period_end": "2025-01-31" },
    { "key": "ytd_budget",       "label": "YTD Budget",       "period_start": "2024-07-01", "period_end": "2025-01-31" },
    { "key": "ytd_dollar_var",   "label": "$ Var",            "period_start": null,          "period_end": null },
    { "key": "annual_budget",    "label": "Annual Budget",    "period_start": "2024-07-01", "period_end": "2025-06-30" }
  ],
  "line_items": [
    {
      "account_number": "4000-0000",
      "account_name": "ASSESSMENT",
      "section": "Income",
      "jul_2024_actual":  101379.00,
      "aug_2024_actual":  101883.00,
      "sep_2024_actual":  100081.00,
      "oct_2024_actual":  103050.00,
      "nov_2024_actual":  101916.00,
      "dec_2024_actual":  102281.00,
      "jan_2025_actual":  101379.00,
      "ytd_actual":       712116.00,
      "ytd_budget":       714000.00,
      "ytd_dollar_var":   -1884.00,
      "annual_budget":    1222960.00
    }
  ],
  "subtotals": [
    { "label": "Total Income",       "section": "income",   "values": { "ytd_actual": 729862.73, "annual_budget": 1222960.00 } },
    { "label": "Total Expenses",     "section": "expenses", "values": { "ytd_actual": 729630.76, "annual_budget": 1222960.00 } },
    { "label": "Operating Income",   "section": "summary",  "values": { "ytd_actual": 231.97 } },
    { "label": "Net Income",         "section": "summary",  "values": { "ytd_actual": 10067.70 } }
  ]
}
```

---

### TRIAL_BALANCE

```json
{
  "entity": { "name": "Acme Corp", "period_end": "2025-01-31", "software": "QuickBooks Desktop" },
  "accounts": [
    {
      "account_number": "1000",
      "account_name": "Checking",
      "account_type": "Bank",
      "debit": 45230.00,
      "credit": 0.0
    },
    {
      "account_number": "2000",
      "account_name": "Accounts Payable",
      "account_type": "Accounts Payable",
      "debit": 0.0,
      "credit": 12400.00
    }
  ],
  "totals": {
    "total_debits": 245000.00,
    "total_credits": 245000.00,
    "difference": 0.00
  }
}
```

---

### GENERAL_LEDGER

```json
{
  "entity": { "name": "Acme Corp", "period_start": "2025-01-01", "period_end": "2025-01-31", "software": "QuickBooks Desktop" },
  "accounts": [
    {
      "account_number": "1000",
      "account_name": "Checking",
      "account_type": "Bank",
      "beginning_balance": 32000.00,
      "transactions": [
        {
          "date": "2025-01-05",
          "type": "Check",
          "number": "1042",
          "name": "Office Depot",
          "memo": "Office supplies",
          "split": "Office Supplies",
          "debit": 245.00,
          "credit": null,
          "balance": 31755.00
        },
        {
          "date": "2025-01-10",
          "type": "Deposit",
          "number": null,
          "name": "Customer ABC",
          "memo": "Invoice 1021",
          "split": "Accounts Receivable",
          "debit": null,
          "credit": 5000.00,
          "balance": 36755.00
        }
      ],
      "ending_balance": 36755.00
    }
  ]
}
```

---

### ACCOUNTS_RECEIVABLE_AGING

```json
{
  "entity": { "name": "Acme Corp", "as_of": "2025-01-31", "software": "QuickBooks Desktop" },
  "aging_buckets": ["current", "1_to_30", "31_to_60", "61_to_90", "over_90"],
  "customers": [
    {
      "customer_name": "Client A",
      "current": 5000.00,
      "1_to_30": 0.0,
      "31_to_60": 1200.00,
      "61_to_90": 0.0,
      "over_90": 0.0,
      "total": 6200.00
    }
  ],
  "totals": {
    "current": 45000.00,
    "1_to_30": 8200.00,
    "31_to_60": 3100.00,
    "61_to_90": 900.00,
    "over_90": 400.00,
    "total": 57600.00
  }
}
```

---

### BUDGET_VS_ACTUAL

```json
{
  "entity": {
    "name": "PMG",
    "period_start": "2024-07-01",
    "period_end": "2025-01-31",
    "software": "AppFolio"
  },
  "columns": {
    "ytd_actual":    "YTD Actual",
    "ytd_budget":    "YTD Budget",
    "ytd_var_dollar":"$ Variance",
    "ytd_var_pct":   "% Variance",
    "annual_budget": "Annual Budget"
  },
  "line_items": [
    {
      "account_number": "5020-0000",
      "account_name": "WATER",
      "section": "Utilities",
      "ytd_actual":    156403.89,
      "ytd_budget":    108501.00,
      "ytd_var_dollar": -47902.89,
      "ytd_var_pct":   -44.2,
      "annual_budget": 140000.00
    }
  ],
  "totals": {
    "total_income":   { "ytd_actual": 729862.73, "ytd_budget": 714000.00, "annual_budget": 1222960.00 },
    "total_expenses": { "ytd_actual": 729630.76, "ytd_budget": 714000.00, "annual_budget": 1222960.00 },
    "net_income":     { "ytd_actual": 231.97,    "ytd_budget": 0.00,       "annual_budget": 0.00 }
  }
}
```

---

### RESERVE_ALLOCATION

```json
{
  "entity": {
    "name": "Niguel Villas HOA",
    "as_of": "2025-01-31",
    "software": "AppFolio",
    "fund_type": "Reserve"
  },
  "components": [
    {
      "account_number": "3001-0000",
      "component_name": "ACCESS GATES",
      "current_balance": 7693.95,
      "annual_contribution": null,
      "fully_funded_balance": null,
      "percent_funded": null
    },
    {
      "account_number": "3011-0000",
      "component_name": "PAINTING",
      "current_balance": -147545.64,
      "annual_contribution": null,
      "fully_funded_balance": null,
      "percent_funded": null
    },
    {
      "account_number": "3015-0000",
      "component_name": "TILE/SHAKE ROOF",
      "current_balance": 694334.24,
      "annual_contribution": null,
      "fully_funded_balance": null,
      "percent_funded": null
    }
  ],
  "total_reserve_balance": 1104435.34,
  "bank_accounts": [
    { "account_number": "1021-0000", "account_name": "SUNWEST BANK-RESERVE", "balance": 328402.40 },
    { "account_number": "1025-0003", "account_name": "MERRILL LYNCH-NIGUEL VILLAS #3196", "balance": 548916.83 }
  ],
  "total_bank_balance": 877319.23,
  "due_to_from": -227116.11
}
```

---

## QuickBooks-Specific Export Quirks

### QuickBooks Desktop (print-to-PDF)

**Font encoding:** QB Desktop uses its own font rendering. When printing to PDF
via the built-in PDF export, fonts are usually embedded with ToUnicode maps and
pdfminer extracts correctly. When printing via the Windows print dialog to a PDF
printer (PDF995, Adobe PDF, Microsoft Print to PDF), you get the same
PScript5/Identity-H problem as the sample document. Detection is the same —
check `uni` column in pdffonts output.

**Report structure quirks:**
- Reports have a header block (company name, report title, date range, basis)
  followed by columnar data
- Subtotals are indented with leading spaces — use indentation level to infer
  hierarchy when parsing
- "Total [Section Name]" rows are always subtotals, not account lines
- Negative numbers appear as `-1,234.56` or `(1,234.56)` — normalize both to
  negative float
- QB uses `***` to indicate zero-balance suppressed accounts in some reports

```python
def normalize_qb_amount(s: str) -> float:
    """Handles QB Desktop amount formats: 1,234.56 / -1,234.56 / (1,234.56) / ***"""
    s = s.strip()
    if s in ('', '***', '-'):
        return 0.0
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace(',', '').replace('$', '').strip()
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return 0.0
```

**Common QB Desktop report layouts to handle:**

| Report | Header signals | Column signals |
|---|---|---|
| Profit & Loss Standard | "Profit & Loss", "Accrual Basis" or "Cash Basis" | Single amount column or Jan/Feb/... months |
| Profit & Loss YTD Comparison | "Profit & Loss", "This Year-to-date" | Two columns + $ Change + % Change |
| Balance Sheet Standard | "Balance Sheet", "As of [date]" | Single balance column |
| Balance Sheet Comparison | "Balance Sheet", "Comparison" | Two date columns + $ Change + % Change |
| A/R Aging Summary | "A/R Aging Summary", "As of [date]" | Current / 1-30 / 31-60 / 61-90 / >90 / Total |
| General Ledger | "General Ledger", "From [date] To [date]" | Date / Num / Name / Memo / Split / Debit / Credit / Balance |

---

### QuickBooks Online (browser export)

QBO exports PDFs via the browser print dialog. Font encoding is generally clean
(Unicode) so pdfminer usually works. However:

- Column alignment is often pixel-based and extracts out of order
- Use `pdfplumber` instead of pdfminer for QBO — it preserves x/y coordinates
  and you can reconstruct columns by x-position

```python
import pdfplumber

def extract_qbo_table(pdf_path: str) -> list[list[str]]:
    """
    QBO PDFs have tabular data that needs coordinate-based extraction.
    pdfplumber's extract_table() handles this better than pdfminer.
    """
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract_table uses cell boundaries
            table = page.extract_table({
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "min_words_vertical": 3,
            })
            if table:
                tables.extend(table)
    return tables
```

---

## Property Management Software Exports

### AppFolio

The sample document is an AppFolio export. Identifying signals:

```python
APPFOLIO_SIGNALS = [
    "appfolio",
    "appfolio opening balance equity",     # specific equity account AppFolio creates
    "gl account map",                       # AppFolio header field
    "level of detail",                      # AppFolio header field
    "include zero balance gl accounts",     # AppFolio header field
    "due to/from reserves",                 # AppFolio HOA account naming
    "due to/from operating",
]
```

AppFolio report headers always contain:
```
Properties:
As of: MM/DD/YYYY
Accounting Basis: Cash
GL Account Map: [None | custom]
Level of Detail: [Detail View | Summary View]
Include Zero Balance GL Accounts: [Yes | No]
```

Extract these into `entity.software_metadata`:

```json
{
  "software": "AppFolio",
  "software_metadata": {
    "properties": "Niguel Villas",
    "accounting_basis": "Cash",
    "gl_account_map": "None - use master chart of accounts",
    "level_of_detail": "Detail View",
    "include_zero_balance_accounts": false,
    "report_created_on": "2025-03-04"
  }
}
```

AppFolio multi-property reports will list multiple properties in the header —
detect and split by property if present.

---

### Yardi

```python
YARDI_SIGNALS = [
    "yardi",
    "voyager",
    "yardi breeze",
    "entity:",           # Yardi report header field
    "book:",             # Yardi accounting book
    "period:",           # Yardi period field
]
```

Yardi reports often include a "Book" field (Accrual, Cash, Budget) and an
"Entity" identifier. Extract both into `entity.software_metadata`.

---

### Buildium

```python
BUILDIUM_SIGNALS = [
    "buildium",
    "property manager",
    "rental owner",      # Buildium owner report header
]
```

---

### Generic HOA / Property Management Signals

If software cannot be identified but content matches HOA financials:

```python
HOA_SIGNALS = [
    r"homeowner[s]?\s+association",
    r"hoa",
    r"reserve\s+fund",
    r"assessment\s+income",
    r"common\s+area",
    r"prepaid\s+assessments",
    r"due\s+to/from\s+reserve",
    r"management\s+fee",
]
```

---

## LLM Prompting for Financial Documents

Financial documents need a different system prompt than tax forms. The key
differences are:

1. **Hierarchical structure** — accounts nest under sections which nest under
   categories. The model must preserve hierarchy, not flatten to a list.
2. **Multi-column layouts** — comparison reports have 8-12 columns; the model
   must map each value to the correct column header.
3. **Upside-down OCR artifacts** — some pages come in rotated. Tell the model
   explicitly that if it sees reversed text (e.g. `yoBpng jenuuy` = `Annual Budget`,
   `awodu|` = `Income`) to interpret by context, not raw OCR output.
4. **Negative number formats** — both `(1,234.56)` and `-1,234.56` are negative.

Add to `prompts.py`:

```python
FINANCIAL_SYSTEM_PROMPT = """
You are a financial document parser specializing in accounting reports, 
property management financials, and QuickBooks exports.

Extract ALL data from the provided text and return ONLY valid JSON. 
No preamble, no markdown fences, no explanation.

CRITICAL RULES:

1. HIERARCHY: Preserve account hierarchy exactly.
   - Line items nest under section names (e.g. "Utilities", "Maintenance")
   - Section totals are labeled "Total [Section Name]"
   - Never flatten hierarchy — keep section groupings

2. MULTI-COLUMN REPORTS: When the report has multiple date columns (months,
   YTD, Budget, Variance), map EVERY value to its correct column.
   - Column headers appear at the top of the report
   - Each account line has one value per column
   - Variance columns may be labeled "$ Var", "% Var", or "$ Variance"

3. NEGATIVE NUMBERS: Normalize all formats to negative float:
   - (1,234.56) → -1234.56
   - -1,234.56  → -1234.56
   - 1,234.56-  → -1234.56

4. OCR ARTIFACTS: This text may have been OCR'd from a rotated page.
   If you see reversed or garbled text that looks like account names 
   spelled backwards, interpret by context and surrounding values.
   Common reversed strings to watch for:
     "yoBpng jenuuy" = "Annual Budget"
     "yoBpng GLA"    = "YTD Budget"  
     "anjoy GLA"     = "YTD Actual"
     "awodu|"        = "Income"
     "asuadxy"       = "Expense"
     "saquiny yunossv" = "Account Number"

5. ACCOUNT NUMBERS: Always extract in format NNNN-NNNN if present.
   If no account number, use null — never invent one.

6. SOFTWARE DETECTION: Identify the source software from headers and set 
   entity.software to one of: QuickBooks Desktop, QuickBooks Online, 
   AppFolio, Yardi, Buildium, Sage, Unknown.

7. TOTALS VERIFICATION: If you can verify that section totals match the
   sum of their line items, set metadata.totals_verified = true.
   If they don't match (OCR errors), set false and add a note.

Output schema: [insert appropriate schema for detected document type]
"""
```

---

## Model Recommendations for Financial Documents

| Document | Complexity | Minimum | Recommended | Notes |
|---|---|---|---|---|
| Balance Sheet (single period) | Low | 7B | 14B | Simple list of accounts |
| P&L Standard | Low | 7B | 14B | |
| P&L Comparison (4-8 columns) | Medium | 14B | 32B | Column mapping is hard for small models |
| P&L 12-Month (12+ columns) | High | 32B | 72B | Context length matters |
| Budget vs Actual with variance | Medium | 14B | 32B | |
| General Ledger (many transactions) | High | 32B | 72B | Long context, transaction rows |
| Reserve Allocation (HOA) | Medium | 14B | 32B | Nested structure, negative balances |
| Multi-property AppFolio report | High | 32B | 72B | Multiple entities in one PDF |
| Upside-down/rotated OCR input | Any + rotation fix | 32B | 72B | OCR noise requires larger model |

**Context length matters more than parameter count here.** A 12-month comparison
report for a 200-account chart of accounts can easily be 8,000-15,000 tokens of
raw text. Prefer models with 32K+ context:

```
Ollama context window settings (add to Modelfile or via API):
  qwen2.5:32b    → default 32K, extend to 128K if needed
  qwen2.5:72b    → default 32K, extend to 128K if needed

vLLM:
  --max-model-len 32768    # minimum for large financials
  --max-model-len 131072   # for full GL exports
```

For very long GL exports, consider chunking by account:

```python
def chunk_gl_by_account(text: str, max_tokens: int = 6000) -> list[str]:
    """
    Split a General Ledger by account boundaries for LLM processing.
    Reassemble after extraction.
    """
    import re
    # Account sections start with account number pattern
    splits = re.split(r'\n(?=\d{4}-\d{4}\s)', text)
    chunks = []
    current = ""
    for section in splits:
        if len(current) + len(section) > max_tokens * 4:  # rough char estimate
            chunks.append(current)
            current = section
        else:
            current += "\n" + section
    if current:
        chunks.append(current)
    return chunks
```
