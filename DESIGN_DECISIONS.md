# w2extract — Canonical Design Decisions

This document is the single source of truth for two foundational decisions
that affect every module. Read this before reading any other spec file.
When this document conflicts with README.md, FINANCIAL_STATEMENTS_SPEC_V2.md,
or SPEC_PATCH_V3.md, this document wins.

---

## Decision 1: section_total is always inline on the section object

### The rule

Every section dict has `section_total` as a direct field. No separate
`subtotals[]` list exists anywhere in any schema.

```python
# CORRECT — section_total inline
{
    "section_name": "Utilities",
    "accounts": [
        {"account_number": "5010-0000", "account_name": "ELECTRICITY", "amount": 3000.00},
        {"account_number": "5020-0000", "account_name": "WATER",       "amount": 24358.33},
        {"account_number": "5050-0000", "account_name": "TRASH/WASTE", "amount": 6133.59}
    ],
    "section_total": 33491.92
}

# WRONG — do not do this
{
    "section_name": "Utilities",
    "accounts": [...],
    "subtotals": [{"section": "Utilities", "total": 33491.92}]   # ← never
}
```

### Why

- Lower LLM failure rate. The model produces one object per section; adding a
  field to that object is structurally simpler than maintaining a parallel list
  that must be cross-referenced by name.
- Trivial verifier. `verifier._get_sections()` reads `section.get("section_total")`
  directly. No join, no name lookup, no misalignment risk.
- Consistent with how every real financial report presents totals — the total
  row appears at the end of its section, not in a separate table.

### Where this applies

Every schema that has sections: `BalanceSheet`, `IncomeStatement`,
`MultiColumnStatement`, `ReserveAllocation`. All nested section levels
(subsections within sections) follow the same rule.

### Prompt instruction (exact wording for all financial system prompts)

```
SECTION TOTALS: Place section_total as a field on the section object itself.
Do not create a separate subtotals array. Do not include total rows in the
accounts array. Example:
  {"section_name": "Cash", "accounts": [...], "section_total": 487081.05}
The verifier reads section_total from the section object and checks it
against sum(accounts). Null is acceptable if no total row is visible.
```

### Verifier implementation (canonical)

```python
# verifier.py — _get_sections() canonical implementation

def _get_sections(extracted: dict) -> list[dict]:
    """
    Walk the extracted dict and return all section dicts that have
    both an accounts[] list and a section_total field.
    Handles: flat sections[], nested subsections[], and top-level
    financial sides (assets/liabilities/equity/income/expenses).
    """
    sections = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        # This node is a section if it has accounts and section_total
        if "accounts" in node and "section_total" in node:
            sections.append(node)
        # Recurse into sections[] and subsections[]
        for key in ("sections", "subsections"):
            for child in node.get(key, []):
                _walk(child)
        # Recurse into top-level financial sides
        for key in ("assets", "liabilities", "equity",
                    "income", "expenses", "other_income",
                    "current_assets", "reserve_accounts",
                    "fixed_assets", "current_liabilities",
                    "long_term_liabilities"):
            if key in node:
                _walk(node[key])

    _walk(extracted)
    return sections
```

---

## Decision 2: FinancialMetadata extends to cover all pipeline state

### The rule

Every document type — tax and financial — uses the same `FinancialMetadata`
model. The metadata block is the single place where pipeline state,
verification results, and provenance live. No other location.

The LLM populates only `notes[]`. Everything else is set by Python pipeline
stages after the LLM call returns.

```python
# schema.py — FinancialMetadata canonical definition

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class FinancialMetadata(BaseModel):
    # ── Set by detector.py ───────────────────────────────────────────────────
    encoding_broken: bool = False
    # "text" | "pdfplumber" | "ocr" | "vision"
    extraction_strategy: Optional[str] = None
    # Software detected from headers before LLM call
    software_detected: Optional[str] = None

    # ── Set by ocr.py ────────────────────────────────────────────────────────
    # Page numbers (0-indexed) that were rotated and corrected
    pages_rotated: list[int] = []
    # Average OCR confidence across all pages (0.0–1.0). None if text-layer.
    ocr_confidence: Optional[float] = None
    # Pages below 0.6 confidence threshold
    ocr_low_confidence_pages: list[int] = []

    # ── Set by boundary_detector.py ──────────────────────────────────────────
    # Page range this document covers within the source PDF
    source_pages: Optional[tuple[int, int]] = None   # (start, end) inclusive
    # Confidence score from boundary detection (0.0–1.0)
    boundary_confidence: Optional[float] = None

    # ── Set by llm.py ────────────────────────────────────────────────────────
    # Token counts for cost/capacity tracking
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # finish_reason from the model response
    finish_reason: Optional[str] = None
    # Number of LLM calls (>1 for chunked documents)
    llm_calls: int = 1
    # Number of retry attempts that were needed
    llm_retries: int = 0

    # ── Set by verifier.py ───────────────────────────────────────────────────
    totals_verified: bool = False
    totals_mismatches: list[dict] = []
    # True if balance sheet equation holds: assets == liabilities + equity
    balance_sheet_balanced: Optional[bool] = None

    # ── Set by LLM (only this field) ─────────────────────────────────────────
    # Anything unusual the model noticed during extraction
    notes: list[str] = []

    # ── Standard document flags (LLM or detector) ────────────────────────────
    is_corrected: bool = False      # "CORRECTED" checkbox on tax forms
    is_void: bool = False           # "VOID" checkbox on tax forms
    is_summary_sheet: bool = False  # ADP/Paychex employer batch summary

    class Config:
        # Allow extra fields — future pipeline stages can add without schema bump
        extra = "allow"
```

### Why one metadata shape

- No formatter churn. Every output format (JSON, CSV, Lacerte, TXF) reads
  the same metadata fields from the same location regardless of document type.
  A W2 and a Balance Sheet both have `metadata.totals_verified`,
  `metadata.encoding_broken`, `metadata.notes`. The formatter never branches
  on document family to find metadata.

- Debuggability. When a document fails extraction, `metadata` tells the full
  story: what strategy was used, how confident the OCR was, which pages were
  rotated, whether totals verified, how many LLM calls it took, what the model
  flagged. All in one place.

- Pipeline stages are self-documenting. Each stage owns its fields and only
  writes those fields. The LLM writes only `notes[]`. The verifier writes only
  `totals_verified`, `totals_mismatches`, `balance_sheet_balanced`. Debuggers
  know exactly which stage produced each field.

### Population pattern (enforced in core.py)

```python
# core.py — metadata population sequence

def _build_metadata(
    strategy_info: dict,
    ocr_results: list[dict] | None,
    section: DocumentSection,
    llm_response_meta: dict,
    verification: VerificationResult | None,
) -> dict:
    """
    Assembles the complete metadata dict from pipeline stage outputs.
    Called after all stages complete, before final pydantic validation.
    The LLM's metadata.notes are passed through; all other fields are
    overwritten by pipeline state.
    """
    meta = {
        # detector.py
        "encoding_broken":      strategy_info.get("encoding_broken", False),
        "extraction_strategy":  strategy_info.get("strategy"),
        "software_detected":    strategy_info.get("software_detected"),

        # boundary_detector.py
        "source_pages":         (section.start_page, section.end_page),
        "boundary_confidence":  section.confidence,

        # llm.py
        "prompt_tokens":    llm_response_meta.get("prompt_tokens"),
        "completion_tokens": llm_response_meta.get("completion_tokens"),
        "finish_reason":    llm_response_meta.get("finish_reason"),
        "llm_calls":        llm_response_meta.get("llm_calls", 1),
        "llm_retries":      llm_response_meta.get("llm_retries", 0),
    }

    # ocr.py (only present if OCR was used)
    if ocr_results:
        confidences = [r["confidence"] for r in ocr_results
                       if r.get("confidence") is not None]
        meta["pages_rotated"] = [r["page"] for r in ocr_results
                                  if r.get("rotated")]
        meta["ocr_confidence"] = (sum(confidences) / len(confidences)
                                   if confidences else None)
        meta["ocr_low_confidence_pages"] = [
            r["page"] for r in ocr_results
            if r.get("confidence") is not None and r["confidence"] < 0.6
        ]

    # verifier.py (only present for financial documents)
    if verification is not None:
        meta["totals_verified"]         = verification.verified
        meta["totals_mismatches"]       = verification.mismatches
        meta["balance_sheet_balanced"]  = verification.balance_sheet_balanced
        # Append verifier notes to LLM notes — don't overwrite
        meta["notes"] = meta.get("notes", []) + verification.notes

    return meta
```

### LLM prompt instruction (add to ALL system prompts — tax and financial)

```
METADATA: In your JSON output, include a "metadata" object with only this field:
  {"metadata": {"notes": [str]}}
Use notes[] to flag anything unusual you noticed:
  - Non-standard box codes on tax forms
  - Accounts with unexpected negative balances
  - Columns that appeared garbled or misaligned
  - Totals rows that seemed inconsistent with visible line items
  - Any field you could not read clearly
Do not set any other metadata fields — they are populated by the pipeline.
```

---

## How these two decisions interact

Both decisions point in the same direction: **reduce LLM responsibility,
increase Python responsibility**.

The LLM does three things:
1. Extracts values verbatim from the document
2. Places `section_total` inline on its section (Decision 1)
3. Populates `metadata.notes[]` with observations (Decision 2)

Python does everything else:
- Verifies totals (reads `section_total` inline — Decision 1 makes this trivial)
- Computes derived fields
- Populates all other metadata fields (Decision 2 gives them a permanent home)
- Formats output

This division means the LLM can be swapped (7B for simple docs, 72B for
complex ones) without changing any pipeline logic. The contract is stable:
extract values, place totals inline, note anomalies.

---

## Files changed by these decisions

| File | Change |
|---|---|
| `schema.py` | `FinancialMetadata` is the canonical model for all doc types. Remove any doc-type-specific metadata models. |
| `verifier.py` | `_get_sections()` uses the canonical recursive walker above. No parallel subtotals list handling. |
| `prompts.py` | All system prompts (tax and financial) include both canonical prompt instructions above. |
| `core.py` | `_build_metadata()` assembles metadata from pipeline stages. LLM notes are preserved and appended to, never overwritten. |
| `formatter.py` | Reads metadata from `data["metadata"]` uniformly across all document types. No branching on document family for metadata access. |
| `README.md` | Remove any mention of `subtotals[]` list. Update schema examples to show inline `section_total`. |
| `FINANCIAL_STATEMENTS_SPEC_V2.md` | Superseded by this document for both decisions. |
| `SPEC_PATCH_V3.md` | Superseded by this document for both decisions. |

---

## Non-decisions: things deliberately left open

These are not specified here because they are implementation details
that Claude Code should resolve based on what works:

- Exact pydantic field ordering within each model
- Whether `Section.subsections` is a forward reference or a flat list with
  a `parent_section` field (either works; pick what's easier to walk)
- Whether `FinancialMetadata.source_pages` is a `tuple[int,int]` or two
  separate `source_page_start`/`source_page_end` int fields
  (pydantic v2 handles tuples fine; separate fields are more JSON-friendly)
- Token counting implementation (tiktoken, rough char estimate, or model
  API response — all acceptable)
- Whether `_build_metadata()` lives in `core.py` or a new `pipeline.py`
