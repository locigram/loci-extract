# w2extract — Tax Document Detection Spec

Closes the detection gap. The financial detector (`FINANCIAL_SIGNATURES`,
`detect_financial_document_type()`) was fully specced. Tax form detection
was not. This document specifies it completely.

Read alongside DESIGN_DECISIONS.md (which supersedes earlier files on
section_total and metadata shape).

---

## The gap

`DOCUMENT_FAMILY_MAP` in `prompts.py` maps document type strings like `"W2"`
and `"1099-NEC"` to prompt families. But nothing in the existing spec
actually produces those strings from raw PDF text. `boundary_detector.py`
only has financial document patterns. `detector.py` only determines
extraction strategy (text/pdfplumber/ocr) — it does not identify document type.

The result: Claude Code has to guess where type detection happens, which means
it will either skip it or invent something inconsistent with the rest of the
pipeline.

---

## What I use vs what you run offline

| What I do | Offline equivalent | Notes |
|---|---|---|
| Recognize a W2 from visual layout and text | Regex signature matching on OCR/pdfminer output | Form numbers, checkbox labels, field names are stable across issuers |
| Distinguish 1099-NEC from 1099-MISC | Regex — "Nonemployee compensation" vs "Rents" box labels | Box 1 label is the fastest signal |
| Detect ADP/Paychex summary sheet vs employee copy | Regex — "TOTALS", "BATCH NO", "Total Employees" | Summary sheets need different handling |
| Detect multi-employee PDFs (one PDF, N employees) | Count SSN pattern occurrences | Each employee has one SSN; N matches = N records |
| Detect tax year | Regex on "2025", "20XX" near form title | Always present on every IRS form |
| Distinguish tax from financial document | Run tax signatures first; if no match, run financial | Tax forms have highly stable IRS form numbers |

---

## detector.py — complete tax detection

Add to `detector.py` alongside the existing `get_extraction_strategy()` and
`detect_financial_document_type()` functions.

```python
# detector.py  — tax document detection

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaxDetectionResult:
    document_type: str          # "W2" | "1099-NEC" | ... | "UNKNOWN"
    tax_year: Optional[int]
    confidence: float           # 0.0–1.0
    issuer_software: Optional[str]   # "ADP" | "Paychex" | "Intuit" | None
    is_summary_sheet: bool
    estimated_record_count: int      # number of employees/recipients detected
    notes: list[str]


# ---------------------------------------------------------------------------
# Signature table
# Each entry: (pattern, document_type, confidence_weight)
# Patterns run on lowercased text. Higher weight = stronger signal.
# Multiple patterns for the same type accumulate weight; threshold is 0.5.
# ---------------------------------------------------------------------------

TAX_SIGNATURES: list[tuple[str, str, float]] = [

    # ── W-2 ─────────────────────────────────────────────────────────────────
    # Strong signals
    (r"wage and tax statement",                    "W2", 0.9),
    (r"form\s+w-?2\b",                             "W2", 0.9),
    (r"box\s+1\b.*wages.*tips.*other\s+comp",      "W2", 0.8),
    (r"social security wages",                     "W2", 0.7),
    (r"medicare wages and tips",                   "W2", 0.7),
    # Weaker but useful when combined
    (r"employer.s\s+(?:id|ein)\s+number",          "W2", 0.4),
    (r"employee.s\s+soc(?:ial)?\s*sec",            "W2", 0.4),
    (r"statutory\s+employee",                      "W2", 0.5),
    (r"third.party\s+sick\s+pay",                  "W2", 0.5),
    # ADP/Paychex employer copies
    (r"w-?2\s+and\s+earnings\s+summary",           "W2", 0.95),
    (r"balancing\s+form\s+w-?2",                   "W2", 0.85),

    # ── 1099-NEC ─────────────────────────────────────────────────────────────
    (r"1099.nec\b",                                "1099-NEC", 0.95),
    (r"nonemployee\s+compensation",                "1099-NEC", 0.85),
    (r"form\s+1099.nec",                           "1099-NEC", 0.95),
    # Box 1 label is definitive for NEC vs MISC
    (r"box\s+1\b.*nonemployee",                    "1099-NEC", 0.90),

    # ── 1099-MISC ────────────────────────────────────────────────────────────
    (r"1099.misc\b",                               "1099-MISC", 0.95),
    (r"form\s+1099.misc",                          "1099-MISC", 0.95),
    (r"box\s+1\b.*rents\b",                        "1099-MISC", 0.85),
    (r"fishing\s+boat\s+proceeds",                 "1099-MISC", 0.90),
    (r"crop\s+insurance\s+proceeds",               "1099-MISC", 0.85),

    # ── 1099-INT ─────────────────────────────────────────────────────────────
    (r"1099.int\b",                                "1099-INT", 0.95),
    (r"form\s+1099.int\b",                         "1099-INT", 0.95),
    (r"interest\s+income.*1099",                   "1099-INT", 0.80),
    (r"early\s+withdrawal\s+penalty",              "1099-INT", 0.70),
    (r"tax.exempt\s+interest",                     "1099-INT", 0.65),
    (r"bond\s+premium",                            "1099-INT", 0.60),

    # ── 1099-DIV ─────────────────────────────────────────────────────────────
    (r"1099.div\b",                                "1099-DIV", 0.95),
    (r"form\s+1099.div\b",                         "1099-DIV", 0.95),
    (r"dividends\s+and\s+distributions",           "1099-DIV", 0.85),
    (r"qualified\s+dividends",                     "1099-DIV", 0.75),
    (r"total\s+ordinary\s+dividends",              "1099-DIV", 0.80),
    (r"section\s*199a\s+dividends",                "1099-DIV", 0.80),

    # ── 1099-B ───────────────────────────────────────────────────────────────
    (r"1099.b\b",                                  "1099-B", 0.95),
    (r"form\s+1099.b\b",                           "1099-B", 0.95),
    (r"proceeds\s+from\s+broker",                  "1099-B", 0.85),
    (r"date\s+(?:acquired|sold)",                  "1099-B", 0.70),
    (r"cost\s+or\s+other\s+basis",                 "1099-B", 0.75),
    (r"wash\s+sale\s+loss\s+disallowed",           "1099-B", 0.85),
    (r"covered\s+security",                        "1099-B", 0.65),

    # ── 1099-R ───────────────────────────────────────────────────────────────
    (r"1099.r\b",                                  "1099-R", 0.95),
    (r"form\s+1099.r\b",                           "1099-R", 0.95),
    (r"distributions\s+from\s+pensions",           "1099-R", 0.85),
    (r"gross\s+distribution",                      "1099-R", 0.70),
    (r"taxable\s+amount\s+not\s+determined",       "1099-R", 0.80),
    (r"distribution\s+code",                       "1099-R", 0.70),
    (r"ira\s*/\s*sep\s*/\s*simple",                "1099-R", 0.75),

    # ── 1099-G ───────────────────────────────────────────────────────────────
    (r"1099.g\b",                                  "1099-G", 0.95),
    (r"form\s+1099.g\b",                           "1099-G", 0.95),
    (r"unemployment\s+compensation",               "1099-G", 0.85),
    (r"state\s+or\s+local\s+income\s+tax\s+refund","1099-G", 0.80),
    (r"taxable\s+grants",                          "1099-G", 0.75),
    (r"rtaa\s+payments",                           "1099-G", 0.85),

    # ── 1099-SA ──────────────────────────────────────────────────────────────
    (r"1099.sa\b",                                 "1099-SA", 0.95),
    (r"form\s+1099.sa\b",                          "1099-SA", 0.95),
    (r"distributions\s+from\s+an?\s+hsa",          "1099-SA", 0.85),
    (r"archer\s+msa",                              "1099-SA", 0.80),
    (r"medicare\s+advantage\s+msa",                "1099-SA", 0.80),

    # ── 1099-K ───────────────────────────────────────────────────────────────
    (r"1099.k\b",                                  "1099-K", 0.95),
    (r"form\s+1099.k\b",                           "1099-K", 0.95),
    (r"payment\s+card\s+and\s+(?:third.party\s+)?network", "1099-K", 0.85),
    (r"merchant\s+category\s+code",                "1099-K", 0.80),
    (r"gross\s+payment\s+card\s+transactions",     "1099-K", 0.85),

    # ── 1099-S ───────────────────────────────────────────────────────────────
    (r"1099.s\b",                                  "1099-S", 0.95),
    (r"form\s+1099.s\b",                           "1099-S", 0.95),
    (r"proceeds\s+from\s+real\s+estate",           "1099-S", 0.85),
    (r"date\s+of\s+closing",                       "1099-S", 0.75),
    (r"gross\s+proceeds.*real\s+estate",           "1099-S", 0.80),

    # ── 1099-C ───────────────────────────────────────────────────────────────
    (r"1099.c\b",                                  "1099-C", 0.95),
    (r"form\s+1099.c\b",                           "1099-C", 0.95),
    (r"cancellation\s+of\s+debt",                  "1099-C", 0.90),
    (r"amount\s+of\s+debt\s+discharged",           "1099-C", 0.85),
    (r"identifiable\s+event\s+code",               "1099-C", 0.80),

    # ── 1099-A ───────────────────────────────────────────────────────────────
    (r"1099.a\b",                                  "1099-A", 0.95),
    (r"form\s+1099.a\b",                           "1099-A", 0.95),
    (r"acquisition\s+or\s+abandonment",            "1099-A", 0.90),
    (r"balance\s+of\s+principal\s+outstanding",    "1099-A", 0.80),

    # ── 1098 (Mortgage Interest) ─────────────────────────────────────────────
    (r"\bform\s+1098\b(?!.(?:t|e|c|f|ma|q))",     "1098",   0.90),
    (r"mortgage\s+interest\s+statement",           "1098",   0.90),
    (r"mortgage\s+interest\s+received",            "1098",   0.85),
    (r"outstanding\s+mortgage\s+principal",        "1098",   0.75),
    (r"points\s+paid\s+on\s+purchase",             "1098",   0.70),
    (r"mortgage\s+insurance\s+premiums",           "1098",   0.65),

    # ── 1098-T ───────────────────────────────────────────────────────────────
    (r"1098.t\b",                                  "1098-T", 0.95),
    (r"form\s+1098.t\b",                           "1098-T", 0.95),
    (r"tuition\s+statement",                       "1098-T", 0.90),
    (r"qualified\s+tuition",                       "1098-T", 0.75),
    (r"at\s+least\s+half.time\s+student",          "1098-T", 0.80),
    (r"scholarships\s+or\s+grants",                "1098-T", 0.65),

    # ── 1098-E ───────────────────────────────────────────────────────────────
    (r"1098.e\b",                                  "1098-E", 0.95),
    (r"form\s+1098.e\b",                           "1098-E", 0.95),
    (r"student\s+loan\s+interest\s+statement",     "1098-E", 0.90),
    (r"student\s+loan\s+interest\s+received",      "1098-E", 0.85),

    # ── SSA-1099 ─────────────────────────────────────────────────────────────
    (r"ssa.1099\b",                                "SSA-1099", 0.95),
    (r"social\s+security\s+benefit\s+statement",   "SSA-1099", 0.90),
    (r"net\s+benefits\s+for\s+\d{4}",             "SSA-1099", 0.80),
    (r"benefit\s+statement.*social\s+security",    "SSA-1099", 0.85),

    # ── RRB-1099 ─────────────────────────────────────────────────────────────
    (r"rrb.1099\b",                                "RRB-1099", 0.95),
    (r"railroad\s+retirement\s+(?:board|benefits)","RRB-1099", 0.85),
    (r"tier\s+[12]\s+(?:tax|benefit)",             "RRB-1099", 0.75),

    # ── K-1 variants ─────────────────────────────────────────────────────────
    # 1065 (Partnership) — checked before 1120-S
    (r"schedule\s+k.1.*1065",                      "K1-1065", 0.95),
    (r"partner.s\s+share\s+of\s+income",           "K1-1065", 0.85),
    (r"partnership.*schedule\s+k.?1",              "K1-1065", 0.85),
    (r"partner.s\s+capital\s+account",             "K1-1065", 0.75),
    (r"guaranteed\s+payments",                     "K1-1065", 0.70),

    # 1120-S (S-Corp)
    (r"schedule\s+k.1.*1120.?s",                   "K1-1120S", 0.95),
    (r"shareholder.s\s+share\s+of\s+income",       "K1-1120S", 0.85),
    (r"s\s+corporation.*schedule\s+k.?1",          "K1-1120S", 0.85),
    (r"shareholder.s\s+(?:pro.rata|percentage)",   "K1-1120S", 0.75),

    # 1041 (Estate/Trust)
    (r"schedule\s+k.1.*1041",                      "K1-1041", 0.95),
    (r"beneficiary.s\s+share\s+of\s+income",       "K1-1041", 0.85),
    (r"estate\s+or\s+trust.*schedule\s+k.?1",      "K1-1041", 0.85),
    (r"fiduciary",                                  "K1-1041", 0.55),
]


# ---------------------------------------------------------------------------
# Issuer software detection
# ---------------------------------------------------------------------------

ISSUER_SIGNATURES: list[tuple[str, str]] = [
    (r"\badp\b",                                   "ADP"),
    (r"automatic\s+data\s+processing",             "ADP"),
    (r"w-?2\s+and\s+earnings\s+summary",           "ADP"),
    (r"balancing\s+form\s+w-?2",                   "ADP"),
    (r"\bpaychex\b",                               "Paychex"),
    (r"\bintuit\b",                                "Intuit"),
    (r"\bquickbooks\b",                            "Intuit"),
    (r"\bgusto\b",                                 "Gusto"),
    (r"\bbamboohr\b",                              "BambooHR"),
    (r"\brippling\b",                              "Rippling"),
    (r"\bjustworks\b",                             "Justworks"),
    (r"\bturbotax\b",                              "TurboTax"),
    (r"\bhr\s*block\b",                            "H&R Block"),
    (r"\bfidelity\b",                              "Fidelity"),
    (r"\bvanguard\b",                              "Vanguard"),
    (r"\bcharles\s+schwab\b",                      "Schwab"),
    (r"\btd\s+ameritrade\b",                       "TD Ameritrade"),
    (r"\betrade\b",                                "E*Trade"),
    (r"\bnavient\b",                               "Navient"),
    (r"\bgreat\s+lakes\b",                         "Great Lakes"),
    (r"\bnelnet\b",                                "Nelnet"),
]

# ADP/Paychex summary sheet signals (employer copy, not employee copy)
SUMMARY_SHEET_SIGNALS: list[str] = [
    r"total\s+employees",
    r"total\s+forms\s+(?:count|processed)",
    r"batch\s+no",
    r"for\s*:\s*batch",
    r"w-?2/w-?3\s+totals",
    r"balancing\s+form\s+w-?2",
    r"total\s+eforms",
    r"\btotals\b.*\bfor\b.*\bcompany\b",
]


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_tax_document_type(text: str) -> TaxDetectionResult:
    """
    Identify tax document type from extracted text.
    Runs before the LLM call — output feeds into prompt selection,
    token budget, and schema routing.
    """
    lower = text.lower()

    # ── Score each candidate type ────────────────────────────────────────────
    scores: dict[str, float] = {}
    for pattern, doc_type, weight in TAX_SIGNATURES:
        if re.search(pattern, lower):
            scores[doc_type] = scores.get(doc_type, 0.0) + weight

    # Cap at 1.0
    scores = {k: min(v, 1.0) for k, v in scores.items()}

    # ── Resolve ambiguities ──────────────────────────────────────────────────
    # 1098 vs 1098-T vs 1098-E: form number alone is ambiguous
    # Prefer the more specific match if both scored
    for specific, general in [("1098-T", "1098"), ("1098-E", "1098")]:
        if scores.get(specific, 0) >= 0.7 and scores.get(general, 0) > 0:
            scores[general] = 0.0   # suppress general match

    # 1099-NEC vs 1099-MISC: box 1 label is decisive
    # "Nonemployee compensation" in box 1 = NEC, not MISC
    if re.search(r"nonemployee\s+compensation", lower):
        scores.pop("1099-MISC", None)
    if re.search(r"box\s*1\b.*rents\b", lower):
        scores.pop("1099-NEC", None)

    # K-1 variants: form number is decisive; fall back to content signals
    # If both 1065 and 1120-S scored, pick the higher one
    k1_variants = {k: v for k, v in scores.items()
                   if k in ("K1-1065", "K1-1120S", "K1-1041")}
    if len(k1_variants) > 1:
        best_k1 = max(k1_variants, key=k1_variants.get)
        for k in k1_variants:
            if k != best_k1:
                scores[k] = 0.0

    # ── Pick winner ──────────────────────────────────────────────────────────
    if not scores or max(scores.values()) < 0.5:
        best_type = "UNKNOWN"
        confidence = 0.0
    else:
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

    # ── Detect tax year ──────────────────────────────────────────────────────
    tax_year = _detect_tax_year(text)

    # ── Detect issuer software ───────────────────────────────────────────────
    issuer = None
    for pattern, name in ISSUER_SIGNATURES:
        if re.search(pattern, lower):
            issuer = name
            break

    # ── Detect summary/employer sheet ────────────────────────────────────────
    is_summary = any(re.search(p, lower) for p in SUMMARY_SHEET_SIGNALS)

    # ── Estimate record count ─────────────────────────────────────────────────
    record_count = _estimate_record_count(text, best_type)

    # ── Notes ────────────────────────────────────────────────────────────────
    notes = []
    if confidence < 0.7 and best_type != "UNKNOWN":
        notes.append(f"Low-confidence detection ({confidence:.0%}). "
                     f"Verify document type is {best_type}.")
    if is_summary:
        notes.append("Employer/issuer summary sheet detected. "
                     "Individual employee copies may be in a separate file.")
    if record_count > 1:
        notes.append(f"Estimated {record_count} records in this PDF.")
    runner_up = _runner_up(scores, best_type)
    if runner_up and scores.get(runner_up, 0) > 0.4:
        notes.append(f"Secondary match: {runner_up} "
                     f"({scores[runner_up]:.0%}). Verify if extraction looks wrong.")

    return TaxDetectionResult(
        document_type=best_type,
        tax_year=tax_year,
        confidence=confidence,
        issuer_software=issuer,
        is_summary_sheet=is_summary,
        estimated_record_count=record_count,
        notes=notes,
    )


def _detect_tax_year(text: str) -> Optional[int]:
    """
    Extract tax year from document text.
    Tries form header year, then any 4-digit year in range 2015–2030.
    """
    # Explicit year labels near form titles
    for pattern in [
        r"(?:tax\s+year|year)\s+(\d{4})",
        r"wage\s+and\s+tax\s+statement\s+(\d{4})",
        r"form\s+w-?2\s+.*?(\d{4})",
        r"(\d{4})\s+form\s+w-?2",
        r"(?:for\s+)?(?:calendar\s+)?year\s+(\d{4})",
        r"(?:january|december)\s+\d{1,2},\s+(\d{4})",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            if 2010 <= year <= 2035:
                return year

    # Fall back: first 4-digit number in plausible range
    for m in re.finditer(r"\b(20\d{2})\b", text):
        year = int(m.group(1))
        if 2010 <= year <= 2035:
            return year

    return None


def _estimate_record_count(text: str, doc_type: str) -> int:
    """
    Estimate number of distinct records (employees/recipients) in the PDF.
    Uses SSN/TIN pattern count for tax forms.
    """
    if doc_type == "UNKNOWN":
        return 1

    # SSN pattern: NNN-NN-NNNN
    ssn_matches = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)
    if not ssn_matches:
        return 1

    # W2: each SSN appears 4x (Copy B/C/2/2) — divide by 4
    if doc_type == "W2":
        unique_ssns = len(set(ssn_matches))
        # If all occurrences are unique, it might be deduplicated already
        if unique_ssns == len(ssn_matches):
            return unique_ssns
        # Otherwise divide raw count by 4 (standard W2 copy count)
        return max(1, round(len(ssn_matches) / 4))

    # 1099s: each TIN appears 2x (Copy B and Copy C) for most forms
    if doc_type.startswith("1099"):
        return max(1, round(len(set(ssn_matches))))

    return max(1, len(set(ssn_matches)))


def _runner_up(scores: dict, winner: str) -> Optional[str]:
    """Return the second-highest scoring type."""
    remaining = {k: v for k, v in scores.items() if k != winner}
    if not remaining:
        return None
    return max(remaining, key=remaining.get)
```

---

## Master detect() function

Single entry point that runs tax detection first, financial second, and
returns a unified result. This is what `core.py` calls.

```python
# detector.py  — master detect() function

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentDetectionResult:
    # Extraction strategy
    strategy: str                    # "text" | "pdfplumber" | "ocr" | "vision"
    encoding_broken: bool
    strategy_reason: str

    # Document identity
    document_type: str               # "W2" | "BALANCE_SHEET" | etc.
    document_family: str             # "tax" | "financial_simple" | etc.
    confidence: float

    # Tax-specific (None for financial docs)
    tax_year: Optional[int] = None
    issuer_software: Optional[str] = None
    is_summary_sheet: bool = False
    estimated_record_count: int = 1

    # Notes from all detection stages
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def detect(pdf_path: str, text_sample: str) -> DocumentDetectionResult:
    """
    Master detection function. Called by core.py after text extraction.

    Args:
        pdf_path:    Path to source PDF (for pdffonts check)
        text_sample: First 3 pages of extracted text (clean or OCR'd)

    Returns DocumentDetectionResult with everything core.py needs to
    route the document to the right prompt, token budget, and schema.
    """
    from w2extract.prompts import DOCUMENT_FAMILY_MAP, DocumentFamily

    # Step 1: extraction strategy
    strategy_info = get_extraction_strategy(pdf_path)

    # Step 2: try tax detection first
    tax_result = detect_tax_document_type(text_sample)

    if tax_result.document_type != "UNKNOWN" and tax_result.confidence >= 0.5:
        family = DOCUMENT_FAMILY_MAP.get(tax_result.document_type, "tax")
        return DocumentDetectionResult(
            strategy=strategy_info["strategy"],
            encoding_broken=strategy_info["encoding_broken"],
            strategy_reason=strategy_info["reason"],
            document_type=tax_result.document_type,
            document_family=family if isinstance(family, str) else family.value,
            confidence=tax_result.confidence,
            tax_year=tax_result.tax_year,
            issuer_software=tax_result.issuer_software,
            is_summary_sheet=tax_result.is_summary_sheet,
            estimated_record_count=tax_result.estimated_record_count,
            notes=tax_result.notes,
        )

    # Step 3: try financial detection
    fin_type = detect_financial_document_type(text_sample)
    if fin_type != "FINANCIAL_UNKNOWN":
        family = DOCUMENT_FAMILY_MAP.get(fin_type, DocumentFamily.FINANCIAL_SIMPLE)
        return DocumentDetectionResult(
            strategy=strategy_info["strategy"],
            encoding_broken=strategy_info["encoding_broken"],
            strategy_reason=strategy_info["reason"],
            document_type=fin_type,
            document_family=family if isinstance(family, str) else family.value,
            confidence=0.8,
            notes=strategy_info.get("notes", []),
        )

    # Step 4: unknown
    return DocumentDetectionResult(
        strategy=strategy_info["strategy"],
        encoding_broken=strategy_info["encoding_broken"],
        strategy_reason=strategy_info["reason"],
        document_type="UNKNOWN",
        document_family="unknown",
        confidence=0.0,
        notes=[
            "Could not identify document type from text content. "
            "Check OCR quality or try --vision flag."
        ],
    )
```

---

## How this plugs into core.py

Replace the two separate calls in the current `core.py` draft with the single
`detect()` call:

```python
# core.py — updated Step 1 and Step 2

from w2extract.detector import detect

def extract_document(pdf_path, ...):

    # Step 1+2: combined — extract a text sample, run detection
    strategy_info = get_extraction_strategy(pdf_path)
    strategy = strategy_info["strategy"]

    # Get a text sample (first 3 pages) for detection
    # Use appropriate extractor based on strategy
    if strategy in ("text", "pdfplumber"):
        sample_pages = extract_with_strategy(pdf_path, strategy)[:3]
    else:
        ocr_sample = ocr_pdf(pdf_path, engine=ocr_engine, dpi=dpi,
                             gpu=gpu, fix_orientation=fix_orientation,
                             max_pages=3)
        sample_pages = ocr_sample

    text_sample = _join_pages(sample_pages)

    # Single detection call — returns everything core.py needs
    detection = detect(pdf_path, text_sample)

    # Now extract ALL pages (we only sampled for detection)
    if strategy in ("text", "pdfplumber"):
        pages = extract_with_strategy(pdf_path, strategy)
    else:
        all_ocr = ocr_pdf(pdf_path, engine=ocr_engine, dpi=dpi,
                          gpu=gpu, fix_orientation=fix_orientation)
        pages = all_ocr

    # Step 3+: boundary detection, chunking, LLM — unchanged from SPEC_PATCH_V3
    # Use detection.document_type, detection.document_family,
    # detection.is_summary_sheet, detection.estimated_record_count
    # to route to the right prompt and schema
    ...
```

---

## Detection edge cases

These need explicit handling, not left to the LLM:

### Multi-form PDFs (multiple 1099-NECs from different payers)

`estimated_record_count > 1` signals this. Each record needs separate
extraction. Pass `estimated_record_count` to the LLM prompt:

```
MULTI-RECORD: This PDF contains approximately {N} records.
Extract all {N} as an array under "recipients". Do not merge them.
```

### W2 with all 4 copies (standard employee packet)

`estimated_record_count` will return 1 (one unique SSN, appearing 4 times).
Prompt already says "output ONE record per employee — the form repeats 4x."
No special handling needed.

### ADP summary sheet mixed with employee W2s in same PDF

Rare but possible. `is_summary_sheet = True` on the summary pages;
`boundary_detector.py` should split them. The summary page gets
`is_summary_sheet: true` in metadata and is extracted separately.
The employee W2 pages get normal extraction.

### 1099-B with dozens of transactions

`estimated_record_count` returns 1 (one recipient, many transactions).
The record count heuristic is per-recipient, not per-transaction.
Use token budget and chunking for long 1099-B transaction lists, not
record count.

### K-1 from a fund that issues both 1065 and 1099-DIV

These are separate documents. Boundary detection will split them if they
are in the same PDF. If not split correctly, the runner-up note in
`TaxDetectionResult.notes` will flag the secondary match.

---

## Unit tests for detection

```python
# tests/test_detector.py

import pytest
from w2extract.detector import detect_tax_document_type, _detect_tax_year

W2_TEXT = """
Form W-2 Wage and Tax Statement 2025
Employee's social security number: 622-76-8654
Employer ID number (EIN): 87-4661053
Wages, tips, other compensation: 7680.00
Federal income tax withheld: 0.00
Social security wages: 7680.00
Medicare wages and tips: 7680.00
Statutory employee  Retirement plan  Third-party sick pay
"""

ADP_SUMMARY_TEXT = """
2025 W-2 and EARNINGS SUMMARY  ADP
COMPANY QPL
1  Total Employees
2  Total Forms Count
Balancing Form W-2/W-3 Totals to the Wage and Tax Register
For: BATCH NO. 2025/4/99686
"""

NEC_TEXT = """
Form 1099-NEC  2025
Nonemployee compensation
Box 1: 15000.00
Payer's TIN: 98-7654321
"""

MISC_TEXT = """
Form 1099-MISC  2025
Rents  Box 1: 24000.00
Fishing boat proceeds
Payer's TIN: 11-2233445
"""

INT_TEXT = """
Form 1099-INT  2025
Interest Income
Early withdrawal penalty
Tax-exempt interest
"""

K1_1065_TEXT = """
Schedule K-1 (Form 1065)  2025
Partner's Share of Income, Deductions, Credits, etc.
Guaranteed payments to partner
Partner's capital account analysis
"""

K1_1120S_TEXT = """
Schedule K-1 (Form 1120-S)  2025
Shareholder's Share of Income, Deductions, Credits, etc.
S corporation
Shareholder's pro rata share items
"""


class TestTaxDocumentDetection:

    def test_w2_detected(self):
        result = detect_tax_document_type(W2_TEXT)
        assert result.document_type == "W2"
        assert result.confidence >= 0.7
        assert result.tax_year == 2025

    def test_w2_record_count(self):
        # One SSN appearing 4 times = 1 employee
        four_copy_text = W2_TEXT * 4
        result = detect_tax_document_type(four_copy_text)
        assert result.estimated_record_count == 1

    def test_adp_summary_detected(self):
        result = detect_tax_document_type(ADP_SUMMARY_TEXT)
        assert result.document_type == "W2"
        assert result.is_summary_sheet is True
        assert result.issuer_software == "ADP"

    def test_1099_nec_vs_misc(self):
        nec = detect_tax_document_type(NEC_TEXT)
        assert nec.document_type == "1099-NEC"
        misc = detect_tax_document_type(MISC_TEXT)
        assert misc.document_type == "1099-MISC"

    def test_1099_nec_does_not_match_misc(self):
        # "Nonemployee compensation" should suppress MISC match
        result = detect_tax_document_type(NEC_TEXT)
        assert result.document_type != "1099-MISC"

    def test_1099_int(self):
        result = detect_tax_document_type(INT_TEXT)
        assert result.document_type == "1099-INT"

    def test_k1_1065_vs_1120s(self):
        result_1065 = detect_tax_document_type(K1_1065_TEXT)
        assert result_1065.document_type == "K1-1065"
        result_1120s = detect_tax_document_type(K1_1120S_TEXT)
        assert result_1120s.document_type == "K1-1120S"

    def test_unknown_returns_unknown(self):
        result = detect_tax_document_type("This is an invoice for services rendered.")
        assert result.document_type == "UNKNOWN"
        assert result.confidence < 0.5

    def test_tax_year_extraction(self):
        assert _detect_tax_year("Wage and Tax Statement 2025") == 2025
        assert _detect_tax_year("Form W-2 2024 Dept. of Treasury") == 2024
        assert _detect_tax_year("No year here") is None

    def test_financial_doc_not_detected_as_tax(self):
        financial_text = """
        Balance Sheet - PMG
        As of: 01/31/2025
        ASSETS
        Total Cash  487,081.05
        TOTAL ASSETS  1,040,135.12
        """
        result = detect_tax_document_type(financial_text)
        # Should be UNKNOWN or very low confidence — not a tax form
        assert result.document_type == "UNKNOWN" or result.confidence < 0.5
```

---

## Files affected

| File | Change |
|---|---|
| `detector.py` | Add `TAX_SIGNATURES`, `detect_tax_document_type()`, `detect()`, `DocumentDetectionResult` |
| `core.py` | Replace split detection calls with single `detect(pdf_path, text_sample)` call |
| `boundary_detector.py` | Add tax form boundary patterns (Copy B / Copy C markers for multi-employee PDFs) |
| `tests/test_detector.py` | New — unit tests above |
