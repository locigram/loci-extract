# w2extract — Spec Patch v3

Closes all open issues from v2 review. This document is a targeted patch —
read alongside README.md and FINANCIAL_STATEMENTS_SPEC_V2.md, not instead of them.

---

## Issue 1 — section_total placement: inline wins

**Decision: inline `section_total` on each section dict. Drop the separate `subtotals[]` list.**

Rationale:
- v1 schema and `verifier._get_sections()` both already use inline. Changing to
  a separate list would require the verifier to cross-reference by section name,
  adding a join that buys nothing.
- The LLM produces a section object; putting `section_total` on that same object
  is structurally obvious and less error-prone than maintaining a parallel list.
- Simpler pydantic shape (no extra model).

**Correction to FINANCIAL_SIMPLE_SYSTEM_PROMPT:**

Replace this rule:
```
# WRONG — from v2 prompt
- Section total rows: include them in a separate "subtotals" list, not in
  "accounts". The verifier will confirm they match.
```

With this:
```
# CORRECT
- Section totals: add section_total as a field on the section object itself,
  NOT in the accounts array and NOT in a separate list.
  Example:
    {
      "section_name": "Utilities",
      "accounts": [...],
      "section_total": 33491.92
    }
  The verifier reads section_total from the section object and checks it
  against sum(accounts). Do not include a subtotals[] array anywhere.
```

**Pydantic shape (add to `schema.py`):**

```python
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, field_validator
import re


# ── Shared primitives ────────────────────────────────────────────────────────

class AccountLine(BaseModel):
    account_number: Optional[str] = None
    account_name: str
    balance: Optional[float] = None   # balance sheet accounts
    amount: Optional[float] = None    # income statement accounts

    @field_validator("balance", "amount", mode="before")
    @classmethod
    def normalize_amount(cls, v):
        return _parse_amount(v)

    def value(self) -> float:
        """Return whichever of balance/amount is populated."""
        return self.balance if self.balance is not None else (self.amount or 0.0)


class Section(BaseModel):
    section_name: str
    accounts: list[AccountLine] = []
    subsections: list[Section] = []   # nested sections (e.g. Current Assets > Cash)
    section_total: Optional[float] = None

    @field_validator("section_total", mode="before")
    @classmethod
    def normalize_total(cls, v):
        return _parse_amount(v)


class SoftwareMetadata(BaseModel):
    """
    Home for all software-specific header fields.
    AppFolio, Yardi, QB all have different header structures.
    Store verbatim; callers can inspect what they know about.
    """
    raw: dict[str, Any] = {}

    # AppFolio-specific (populated when software == "AppFolio")
    properties: Optional[str] = None
    accounting_basis: Optional[str] = None
    gl_account_map: Optional[str] = None
    level_of_detail: Optional[str] = None
    include_zero_balance_accounts: Optional[bool] = None
    report_created_on: Optional[str] = None
    fund_type: Optional[str] = None

    # Yardi-specific
    book: Optional[str] = None
    entity_code: Optional[str] = None

    # QB-specific
    report_basis: Optional[str] = None   # "Accrual" | "Cash"
    report_date_range: Optional[str] = None


class FinancialEntity(BaseModel):
    name: str
    type: Optional[str] = None          # "HOA" | "LLC" | "Corp" | "Individual" | ...
    accounting_basis: Optional[str] = None
    period_start: Optional[str] = None  # ISO date
    period_end: Optional[str] = None
    prepared_by: Optional[str] = None
    software: str = "Unknown"           # "AppFolio" | "QuickBooks Desktop" | ...
    software_metadata: Optional[SoftwareMetadata] = None


class FinancialMetadata(BaseModel):
    is_corrected: bool = False
    is_void: bool = False
    is_summary_sheet: bool = False
    encoding_broken: bool = False       # set by detector, not LLM
    pages_rotated: list[int] = []       # set by OCR pipeline, not LLM
    totals_verified: bool = False       # set by verifier.py, not LLM
    totals_mismatches: list[dict] = []  # set by verifier.py
    notes: list[str] = []


# ── Balance Sheet ─────────────────────────────────────────────────────────────

class BalanceSheetAssets(BaseModel):
    sections: list[Section] = []
    total_assets: Optional[float] = None

    @field_validator("total_assets", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class BalanceSheetLiabilities(BaseModel):
    sections: list[Section] = []
    total_liabilities: Optional[float] = None

    @field_validator("total_liabilities", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class BalanceSheetEquity(BaseModel):
    sections: list[Section] = []
    total_equity_reported: Optional[float] = None   # LLM extracts verbatim
    # computed by verifier.py:
    retained_earnings_calculated: Optional[float] = None
    prior_years_retained_earnings_calculated: Optional[float] = None

    @field_validator("total_equity_reported", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class BalanceSheet(BaseModel):
    entity: FinancialEntity
    assets: BalanceSheetAssets
    liabilities: BalanceSheetLiabilities
    equity: BalanceSheetEquity
    total_liabilities_and_equity_reported: Optional[float] = None
    # computed by verifier.py:
    check_difference: Optional[float] = None
    metadata: FinancialMetadata = FinancialMetadata()

    @field_validator("total_liabilities_and_equity_reported", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


# ── Income Statement ──────────────────────────────────────────────────────────

class IncomeStatementSide(BaseModel):
    sections: list[Section] = []
    total: Optional[float] = None

    @field_validator("total", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class IncomeStatement(BaseModel):
    entity: FinancialEntity
    income: IncomeStatementSide
    expenses: IncomeStatementSide
    other_income: Optional[IncomeStatementSide] = None
    other_expenses: Optional[IncomeStatementSide] = None
    operating_income_reported: Optional[float] = None
    net_income_reported: Optional[float] = None
    # computed by verifier.py:
    net_income_calculated: Optional[float] = None
    metadata: FinancialMetadata = FinancialMetadata()

    @field_validator("operating_income_reported", "net_income_reported", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


# ── Multi-column comparison ───────────────────────────────────────────────────

class ColumnDefinition(BaseModel):
    key: str            # e.g. "jul_2024_actual"
    label: str          # e.g. "Jul 2024 Actual"
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    column_type: str = "actual"  # "actual" | "budget" | "variance_dollar" | "variance_pct"


class MultiColumnAccountLine(BaseModel):
    account_number: Optional[str] = None
    account_name: str
    section: str
    subsection: Optional[str] = None
    row_type: str = "account"   # "account" | "subtotal" | "total"
    values: dict[str, Optional[float]] = {}   # {col_key: value}

    @field_validator("values", mode="before")
    @classmethod
    def normalize_values(cls, v):
        if not isinstance(v, dict):
            return {}
        return {k: _parse_amount(val) for k, val in v.items()}


class MultiColumnStatement(BaseModel):
    entity: FinancialEntity
    columns: list[ColumnDefinition]
    line_items: list[MultiColumnAccountLine] = []
    metadata: FinancialMetadata = FinancialMetadata()


# ── General Ledger ────────────────────────────────────────────────────────────

class GLTransaction(BaseModel):
    date: Optional[str] = None
    type: Optional[str] = None
    number: Optional[str] = None
    name: Optional[str] = None
    memo: Optional[str] = None
    split: Optional[str] = None
    debit: Optional[float] = None
    credit: Optional[float] = None
    balance: Optional[float] = None
    row_type: str = "transaction"   # "transaction" | "balance_header" | "balance_footer"

    @field_validator("debit", "credit", "balance", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class GLAccount(BaseModel):
    account_number: Optional[str] = None
    account_name: str
    account_type: Optional[str] = None
    beginning_balance: Optional[float] = None
    ending_balance: Optional[float] = None
    transactions: list[GLTransaction] = []

    @field_validator("beginning_balance", "ending_balance", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class GeneralLedger(BaseModel):
    entity: FinancialEntity
    accounts: list[GLAccount] = []
    metadata: FinancialMetadata = FinancialMetadata()


# ── AR/AP Aging ───────────────────────────────────────────────────────────────

class AgingRow(BaseModel):
    name: str
    current: Optional[float] = None
    days_1_30: Optional[float] = None
    days_31_60: Optional[float] = None
    days_61_90: Optional[float] = None
    over_90: Optional[float] = None
    total: Optional[float] = None

    @field_validator("current","days_1_30","days_31_60","days_61_90","over_90","total",
                     mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class AgingReport(BaseModel):
    entity: FinancialEntity
    report_type: str = "AR"   # "AR" | "AP"
    as_of: Optional[str] = None
    aging_buckets: list[str] = ["current","1_to_30","31_to_60","61_to_90","over_90"]
    rows: list[AgingRow] = []
    totals: Optional[AgingRow] = None
    metadata: FinancialMetadata = FinancialMetadata()


# ── Reserve Allocation ────────────────────────────────────────────────────────

class ReserveComponent(BaseModel):
    account_number: Optional[str] = None
    component_name: str
    current_balance: Optional[float] = None
    annual_contribution: Optional[float] = None
    fully_funded_balance: Optional[float] = None
    percent_funded: Optional[float] = None

    @field_validator("current_balance","annual_contribution",
                     "fully_funded_balance","percent_funded", mode="before")
    @classmethod
    def nt(cls, v): return _parse_amount(v)


class ReserveAllocation(BaseModel):
    entity: FinancialEntity
    components: list[ReserveComponent] = []
    bank_accounts: list[AccountLine] = []
    # All totals computed by verifier.py:
    total_reserve_balance_calculated: Optional[float] = None
    total_bank_balance_calculated: Optional[float] = None
    due_to_from_calculated: Optional[float] = None
    metadata: FinancialMetadata = FinancialMetadata()


# ── Amount normalization ──────────────────────────────────────────────────────

def _parse_amount(v) -> Optional[float]:
    """
    Normalize all financial amount formats to float or None.
    Handles: 1234.56 / 1,234.56 / (1,234.56) / -1,234.56 / 1,234.56- / *** / ""
    """
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("", "***", "-", "—", "N/A"):
        return None
    negative = (s.startswith("(") and s.endswith(")")) or s.endswith("-")
    s = (s.removeprefix("(").removesuffix(")")
          .removesuffix("-")
          .replace(",", "")
          .replace("$", "")
          .strip())
    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return None
```

---

## Issue 2 — Confirmed: inline section_total

See Issue 1 above. Inline wins. `subtotals[]` list removed from all schemas and prompts. `verifier._get_sections()` reads `section_total` from the section object — no change needed to verifier.

---

## Issue 3 — `software_metadata` on all financial models

`SoftwareMetadata` is defined in Issue 1 above and wired into `FinancialEntity`
as `software_metadata: Optional[SoftwareMetadata] = None`.

It lives on the entity, not the document root, because the metadata describes
the entity's reporting context (what software, what accounting book, what
property filter) rather than the document structure.

All five financial pydantic models (`BalanceSheet`, `IncomeStatement`,
`MultiColumnStatement`, `GeneralLedger`, `AgingReport`, `ReserveAllocation`)
include `entity: FinancialEntity` which carries `software_metadata`.

**AppFolio extraction instruction (add to FINANCIAL_SIMPLE_SYSTEM_PROMPT):**

```
- AppFolio header fields: extract into entity.software_metadata.
  Map these header lines:
    "Properties:"                        → software_metadata.properties
    "As of: MM/DD/YYYY"                  → entity.period_end (ISO format) AND
                                           software_metadata (verbatim string)
    "Accounting Basis: Cash|Accrual"     → entity.accounting_basis AND
                                           software_metadata.accounting_basis
    "GL Account Map: ..."                → software_metadata.gl_account_map
    "Level of Detail: ..."               → software_metadata.level_of_detail
    "Include Zero Balance GL Accounts: " → software_metadata.include_zero_balance_accounts
    "Created on MM/DD/YYYY"              → software_metadata.report_created_on (ISO)
    "Fund Type: ..."                     → software_metadata.fund_type
  Set entity.software = "AppFolio".
  Store the full header as software_metadata.raw dict if any fields are
  unrecognized.
```

---

## Issue 4 — Chunked GL merge step

`chunk_for_llm()` produces `N` `TextChunk` objects → `N` LLM calls → `N`
partial `GeneralLedger` objects. The merge step is explicit below.

```python
# core.py  (new section — add after _call_llm_single)

import json
from w2extract.schema import GeneralLedger, FinancialMetadata
from w2extract.chunker import TextChunk, chunk_for_llm
from w2extract.llm import call_llm
from w2extract.prompts import get_prompt
from w2extract.verifier import verify_section_totals, compute_derived_fields


def extract_chunked(
    chunks: list[TextChunk],
    document_type: str,
    llm_client,
    model_name: str,
    max_tokens_override: int | None = None,
    temperature: float = 0,
    retries: int = 2,
) -> dict:
    """
    Call LLM once per chunk, then merge results.
    Returns merged dict ready for pydantic validation.
    """
    schema = _get_schema_hint(document_type)
    system_prompt = get_prompt(document_type, schema)

    partial_results = []
    for chunk in chunks:
        user_text = _build_chunk_prompt(chunk, document_type)
        raw_json = call_llm(
            client=llm_client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_text=user_text,
            document_type=document_type,
            max_tokens_override=max_tokens_override,
            temperature=temperature,
            retries=retries,
        )
        try:
            partial_results.append(json.loads(raw_json))
        except json.JSONDecodeError as e:
            import sys
            print(f"WARNING: Chunk {chunk.chunk_index+1}/{chunk.total_chunks} "
                  f"produced invalid JSON after retries: {e}", file=sys.stderr)
            # Continue — a partial GL is better than a failed extraction

    return _merge_chunks(partial_results, document_type)


def _build_chunk_prompt(chunk: TextChunk, document_type: str) -> str:
    header = ""
    if chunk.total_chunks > 1:
        header = (
            f"[CHUNK {chunk.chunk_index + 1} OF {chunk.total_chunks}]\n"
            f"This is one segment of a larger {document_type} document.\n"
        )
        if chunk.account_context:
            header += f"This chunk begins at: {chunk.account_context}\n"
        if chunk.chunk_index > 0:
            header += (
                "The entity/header information was in chunk 1. "
                "For this chunk, extract only the accounts/transactions present. "
                "Set entity fields to null — they will be taken from chunk 1.\n"
            )
        header += "\n"
    return header + chunk.text


def _merge_chunks(partials: list[dict], document_type: str) -> dict:
    """
    Merge N partial extraction dicts into one.
    Strategy by document type:
      - GL / Transaction:  concatenate accounts[] / transactions[]
      - Balance Sheet:     sections[] from each chunk merged by section_name
      - P&L:               same as Balance Sheet
      - Multi-column:      line_items[] concatenated
    Entity and metadata always taken from first non-null chunk.
    """
    if not partials:
        return {}
    if len(partials) == 1:
        return partials[0]

    merged = {}

    # Entity: first chunk that has a populated entity.name
    for p in partials:
        entity = p.get("entity", {})
        if entity and entity.get("name"):
            merged["entity"] = entity
            break
    if "entity" not in merged:
        merged["entity"] = partials[0].get("entity", {})

    # Metadata: merge notes and rotated_pages lists; OR boolean flags
    merged_meta: dict = {}
    all_notes: list[str] = []
    all_rotated: list[int] = []
    for p in partials:
        meta = p.get("metadata", {})
        all_notes.extend(meta.get("notes", []))
        all_rotated.extend(meta.get("pages_rotated", []))
        for flag in ("encoding_broken", "is_corrected", "is_void"):
            if meta.get(flag):
                merged_meta[flag] = True
    merged_meta["notes"] = list(set(all_notes))
    merged_meta["pages_rotated"] = sorted(set(all_rotated))
    merged["metadata"] = merged_meta

    # Document-type-specific array merging
    if document_type in ("GENERAL_LEDGER", "QB_GENERAL_LEDGER"):
        merged["accounts"] = _merge_gl_accounts(partials)

    elif document_type in ("QB_TRANSACTION_LIST",):
        merged["transactions"] = []
        for p in partials:
            merged["transactions"].extend(p.get("transactions", []))

    elif document_type in ("INCOME_STATEMENT_COMPARISON", "BUDGET_VS_ACTUAL",
                           "QB_PROFIT_LOSS"):
        # Columns from first chunk, line_items concatenated (dedup by account_number)
        merged["columns"] = partials[0].get("columns", [])
        seen = set()
        merged["line_items"] = []
        for p in partials:
            for item in p.get("line_items", []):
                key = (item.get("account_number"), item.get("account_name"))
                if key not in seen:
                    seen.add(key)
                    merged["line_items"].append(item)

    elif document_type in ("BALANCE_SHEET",):
        merged["assets"] = _merge_financial_side(partials, "assets")
        merged["liabilities"] = _merge_financial_side(partials, "liabilities")
        merged["equity"] = _merge_financial_side(partials, "equity")
        # Top-level totals from first chunk that has them
        for key in ("total_liabilities_and_equity_reported",):
            for p in partials:
                if p.get(key) is not None:
                    merged[key] = p[key]
                    break

    elif document_type in ("INCOME_STATEMENT",):
        merged["income"] = _merge_financial_side(partials, "income")
        merged["expenses"] = _merge_financial_side(partials, "expenses")
        merged["other_income"] = _merge_financial_side(partials, "other_income")
        for key in ("operating_income_reported", "net_income_reported"):
            for p in partials:
                if p.get(key) is not None:
                    merged[key] = p[key]
                    break

    else:
        # Generic: concatenate any array fields, take scalars from first chunk
        for p in partials[1:]:
            for k, v in p.items():
                if k in ("entity", "metadata"):
                    continue
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                elif k not in merged:
                    merged[k] = v

    return merged


def _merge_gl_accounts(partials: list[dict]) -> list[dict]:
    """
    Merge GL accounts across chunks.
    An account split across chunk boundaries will appear in two partials
    with the same account_number — merge its transaction lists.
    """
    accounts_by_key: dict[str, dict] = {}
    order: list[str] = []

    for p in partials:
        for acct in p.get("accounts", []):
            # Key: account_number if present, else account_name
            key = acct.get("account_number") or acct.get("account_name", "")
            if key in accounts_by_key:
                # Merge transactions
                existing = accounts_by_key[key]
                existing["transactions"] = (
                    existing.get("transactions", []) +
                    acct.get("transactions", [])
                )
                # Take ending_balance from the later chunk
                if acct.get("ending_balance") is not None:
                    existing["ending_balance"] = acct["ending_balance"]
            else:
                accounts_by_key[key] = dict(acct)
                order.append(key)

    return [accounts_by_key[k] for k in order]


def _merge_financial_side(partials: list[dict], key: str) -> dict:
    """
    Merge a balance sheet or P&L side (assets/liabilities/equity/income/expenses)
    across chunks. Sections are merged by section_name; no duplicates.
    """
    merged_side: dict = {}
    sections_by_name: dict[str, dict] = {}
    section_order: list[str] = []

    for p in partials:
        side = p.get(key, {})
        if not side:
            continue
        # Take top-level scalar (total_assets, etc.) from first occurrence
        for k, v in side.items():
            if k != "sections" and k not in merged_side:
                merged_side[k] = v
        # Merge sections
        for section in side.get("sections", []):
            sname = section.get("section_name", "")
            if sname in sections_by_name:
                # Extend accounts list; keep section_total from last occurrence
                sections_by_name[sname]["accounts"].extend(
                    section.get("accounts", [])
                )
                if section.get("section_total") is not None:
                    sections_by_name[sname]["section_total"] = section["section_total"]
            else:
                sections_by_name[sname] = dict(section)
                section_order.append(sname)

    merged_side["sections"] = [sections_by_name[n] for n in section_order]
    return merged_side


def _get_schema_hint(document_type: str) -> str:
    """
    Returns a compact schema hint string for insertion into the system prompt.
    Keeps the prompt from becoming enormous — model gets field names, not full
    pydantic definitions.
    """
    hints = {
        "BALANCE_SHEET": """
{
  "entity": {"name":str, "type":str|null, "accounting_basis":str, "period_end":"YYYY-MM-DD",
             "software":str, "software_metadata":{...}},
  "assets": {
    "sections": [{"section_name":str, "accounts":[{"account_number":str|null,
                  "account_name":str, "balance":float}], "section_total":float|null}],
    "total_assets": float|null
  },
  "liabilities": {"sections":[...], "total_liabilities":float|null},
  "equity": {"sections":[...], "total_equity_reported":float|null},
  "total_liabilities_and_equity_reported": float|null,
  "metadata": {"notes":[str]}
}""",

        "INCOME_STATEMENT": """
{
  "entity": {"name":str, "accounting_basis":str, "period_start":"YYYY-MM-DD",
             "period_end":"YYYY-MM-DD", "software":str, "software_metadata":{...}},
  "income":   {"sections":[{"section_name":str,
                "accounts":[{"account_number":str|null,"account_name":str,"amount":float}],
                "section_total":float|null}], "total":float|null},
  "expenses": {"sections":[...], "total":float|null},
  "other_income": {"sections":[...], "total":float|null} | null,
  "operating_income_reported": float|null,
  "net_income_reported": float|null,
  "metadata": {"notes":[str]}
}""",

        "INCOME_STATEMENT_COMPARISON": """
{
  "entity": {"name":str, "period_start":"YYYY-MM-DD", "period_end":"YYYY-MM-DD",
             "software":str, "software_metadata":{...}},
  "columns": [{"key":str, "label":str, "period_start":str|null, "period_end":str|null,
               "column_type":"actual"|"budget"|"variance_dollar"|"variance_pct"}],
  "line_items": [{"account_number":str|null, "account_name":str,
                  "section":str, "subsection":str|null,
                  "row_type":"account"|"subtotal"|"total",
                  "values":{"col_key":float|null, ...}}],
  "metadata": {"notes":[str]}
}""",

        "GENERAL_LEDGER": """
{
  "entity": {"name":str, "period_start":"YYYY-MM-DD", "period_end":"YYYY-MM-DD",
             "software":str, "software_metadata":{...}},
  "accounts": [
    {"account_number":str|null, "account_name":str, "account_type":str|null,
     "beginning_balance":float|null, "ending_balance":float|null,
     "transactions":[{"date":str|null, "type":str|null, "number":str|null,
                      "name":str|null, "memo":str|null, "split":str|null,
                      "debit":float|null, "credit":float|null, "balance":float|null,
                      "row_type":"transaction"}]}
  ],
  "metadata": {"notes":[str]}
}""",
    }
    return hints.get(document_type, '{"document_type": "' + document_type + '", ...}')
```

---

## Issue 5 — Test fixtures: what to use until real samples arrive

Phase 1 and Phase 2 can use synthetic fixtures. Phase 3 needs real exports.

### Synthetic encoding-broken PDF (for unit tests)

```python
# tests/fixtures/generate_broken_pdf.py
"""
Generates a PDF that mimics the PScript5/Distiller encoding-broken pattern:
text layer present but ToUnicode missing. Uses reportlab with a custom
CID font that has no ToUnicode map.

For unit testing detector.py and the OCR fallback path without needing
real client data.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os


def generate_broken_balance_sheet(output_path: str):
    """
    Generates a syntactically valid PDF with a financial layout but
    encoding that will confuse pdfminer. The text layer content is
    deliberately set to look like the garbled output of a real broken PDF.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica", 10)  # Helvetica is always clean — this is a clean fixture
    # For a truly broken fixture, you need a CID font with Identity-H
    # and no ToUnicode map. Use an external tool:
    #   gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
    #      -dPDFSETTINGS=/printer -sOutputFile=broken.pdf input.pdf
    # The -dPDFSETTINGS=/printer path triggers Distiller-style font subsetting
    # that drops ToUnicode maps, reproducing the real-world failure.

    # Clean fixture for logic testing:
    c.drawString(72, 720, "Balance Sheet - Test Entity")
    c.drawString(72, 704, "As of: 01/31/2025")
    c.drawString(72, 688, "Accounting Basis: Cash")
    c.drawString(72, 656, "ASSETS")
    c.drawString(72, 640, "Cash")
    c.drawString(200, 640, "1018-0000")
    c.drawString(350, 640, "SUNWEST BANK-OPERATING")
    c.drawString(500, 640, "158,678.65")
    c.drawString(72, 624, "Total Cash")
    c.drawString(500, 624, "158,678.65")
    c.save()


# For a truly encoding-broken fixture, use ghostscript post-processing
# or copy a real AppFolio export to tests/fixtures/appfolio_balance_sheet.pdf
# and add it to .gitignore if it contains any PII.

FIXTURE_REGISTRY = {
    "appfolio_balance_sheet":          "tests/fixtures/appfolio_balance_sheet.pdf",
    "appfolio_12mo_comparison":        "tests/fixtures/appfolio_12mo_comparison.pdf",
    "qb_desktop_profit_loss":          "tests/fixtures/qb_desktop_pl.pdf",
    "qb_desktop_general_ledger":       "tests/fixtures/qb_desktop_gl.pdf",
    "qb_online_balance_sheet":         "tests/fixtures/qb_online_bs.pdf",
    "qb_online_ar_aging":              "tests/fixtures/qb_online_ar_aging.pdf",
    "yardi_income_statement":          "tests/fixtures/yardi_income.pdf",
    "w2_digital_simple":               "tests/fixtures/w2_digital_simple.pdf",
    "w2_scanned":                      "tests/fixtures/w2_scanned.pdf",
    "1099_nec_digital":                "tests/fixtures/1099_nec.pdf",
}

def fixture_available(name: str) -> bool:
    path = FIXTURE_REGISTRY.get(name)
    return path is not None and os.path.exists(path)

def skip_if_no_fixture(name: str):
    """Use as pytest decorator: @skip_if_no_fixture("qb_desktop_profit_loss")"""
    import pytest
    return pytest.mark.skipif(
        not fixture_available(name),
        reason=f"Fixture '{name}' not available. "
               f"See tests/fixtures/README.md for acquisition instructions."
    )
```

### tests/fixtures/README.md

```markdown
# Test Fixtures

Real PDF samples needed for Phase 3 integration tests.
Never commit files containing real PII — use sanitized exports only.

## Acquisition

### QuickBooks Desktop
1. Open QuickBooks Desktop sample company: Help → Use Sample Company
2. Reports → Company & Financial → Profit & Loss Standard
3. Export: File → Save as PDF → save to tests/fixtures/qb_desktop_pl.pdf
4. Also export via: File → Print → Microsoft Print to PDF
   (creates encoding-broken version — save as qb_desktop_pl_broken.pdf)
5. Repeat for General Ledger and Balance Sheet

### QuickBooks Online
1. Use QBO test drive: https://qbo.intuit.com/redir/testdrive
2. Reports → Balance Sheet → Export → Export to PDF
3. Save to tests/fixtures/qb_online_bs.pdf

### AppFolio
1. Request demo account: https://www.appfolio.com/request-a-demo
2. Run Balance Sheet report, export to PDF
3. Save to tests/fixtures/appfolio_balance_sheet.pdf
4. Note: AppFolio exports through PScript5/Distiller on Windows —
   confirm encoding is broken with: pdffonts appfolio_balance_sheet.pdf
   Look for uni=no in the output.

## Privacy
All fixtures must be sanitized:
- Replace real entity names with "Test Entity LLC"
- Replace real account balances with round numbers
- No real SSNs, EINs, bank account numbers, or names
```

---

## Issue 6 — core.py refactor

The tax flow must continue working exactly as before. The refactor wraps the
existing single-call path and adds the financial multi-step path. No existing
tax extraction behavior changes.

```python
# core.py  (full replacement)

from __future__ import annotations
import json
from typing import Any

from w2extract.detector import get_extraction_strategy, detect_financial_document_type
from w2extract.boundary_detector import detect_boundaries, DocumentSection
from w2extract.extractor import extract_with_strategy
from w2extract.ocr import ocr_pdf
from w2extract.chunker import chunk_for_llm
from w2extract.prompts import get_prompt, DocumentFamily, DOCUMENT_FAMILY_MAP
from w2extract.llm import call_llm, get_token_budget
from w2extract.verifier import verify_section_totals, compute_derived_fields
from w2extract.schema import FinancialMetadata
import openai


def extract_document(
    pdf_path: str,
    llm_endpoint: str,
    model_name: str = "local",
    ocr_engine: str = "auto",
    gpu: bool | str = "auto",
    vision: bool = False,
    vision_model: str = "llava:34b",
    dpi: int = 300,
    fix_orientation: bool = True,
    max_tokens_override: int | None = None,
    temperature: float = 0,
    retries: int = 2,
    force_family: str | None = None,
    chunk_size_tokens: int = 6000,
    verify_totals: bool = True,
) -> dict:
    """
    Main entry point. Returns {"documents": [...]}.
    Tax documents: one LLM call, existing behavior unchanged.
    Financial documents: boundary detection → section-by-section extraction
                         → chunk if needed → merge → verify.
    """
    client = openai.OpenAI(base_url=llm_endpoint, api_key="local")

    # ── Step 1: Determine extraction strategy (text / pdfplumber / ocr / vision)
    strategy_info = get_extraction_strategy(pdf_path)
    strategy = strategy_info["strategy"]
    encoding_broken = strategy_info["encoding_broken"]

    # ── Step 2: Extract raw text per page
    if strategy in ("text", "pdfplumber"):
        pages = extract_with_strategy(pdf_path, strategy)
    else:
        # OCR or vision path
        if vision:
            pages = _extract_vision(pdf_path, client, vision_model, dpi)
        else:
            ocr_results = ocr_pdf(
                pdf_path, engine=ocr_engine, dpi=dpi, gpu=gpu,
                fix_orientation=fix_orientation
            )
            pages = [{"page": r["page"], "text": r["text"],
                      "rotated": r.get("rotated", False),
                      "confidence": r.get("confidence")} for r in ocr_results]

    # ── Step 3: Detect document boundaries within the PDF
    sections: list[DocumentSection] = detect_boundaries(pages)

    # ── Step 4: Per-section extraction
    documents = []
    for section in sections:
        section_pages = [p for p in pages
                         if section.start_page <= p["page"] <= section.end_page]
        section_text = _join_pages(section_pages)

        doc_type = section.document_type
        family = (DocumentFamily(force_family) if force_family
                  else DOCUMENT_FAMILY_MAP.get(doc_type, DocumentFamily.FINANCIAL_SIMPLE))

        # Determine if this is a tax or financial document
        is_tax = (family == DocumentFamily.TAX)

        if is_tax:
            # ── Tax path: single LLM call, existing behavior ──────────────────
            result = _extract_tax_single(
                text=section_text,
                doc_type=doc_type,
                client=client,
                model_name=model_name,
                max_tokens_override=max_tokens_override,
                temperature=temperature,
                retries=retries,
            )
        else:
            # ── Financial path: chunk → LLM → merge → verify ─────────────────
            chunks = chunk_for_llm(
                text=section_text,
                document_type=doc_type,
                max_input_tokens=chunk_size_tokens,
            )

            if len(chunks) == 1:
                schema_hint = _get_schema_hint(doc_type)
                system_prompt = get_prompt(doc_type, schema_hint)
                raw_json = call_llm(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_text=chunks[0].text,
                    document_type=doc_type,
                    max_tokens_override=max_tokens_override,
                    temperature=temperature,
                    retries=retries,
                )
                result = json.loads(raw_json)
            else:
                from w2extract.core_chunked import extract_chunked
                result = extract_chunked(
                    chunks=chunks,
                    document_type=doc_type,
                    llm_client=client,
                    model_name=model_name,
                    max_tokens_override=max_tokens_override,
                    temperature=temperature,
                    retries=retries,
                )

            # ── Post-processing: verify totals and compute derived fields ──────
            if verify_totals:
                verification = verify_section_totals(result)
                meta = result.setdefault("metadata", {})
                meta["totals_verified"]    = verification.verified
                meta["totals_mismatches"]  = verification.mismatches
                meta["notes"]              = (meta.get("notes", []) +
                                              verification.notes)

            derived = compute_derived_fields(result, doc_type)
            result.update(derived)

            # ── Propagate pipeline metadata (encoding, rotation) ──────────────
            meta = result.setdefault("metadata", {})
            meta["encoding_broken"] = encoding_broken
            rotated = [p["page"] for p in section_pages if p.get("rotated")]
            if rotated:
                meta["pages_rotated"] = rotated

        documents.append({
            "document_type": doc_type,
            "tax_year": _extract_tax_year(result),
            "section_title": section.title,
            "data": result,
        })

    return {"documents": documents}


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_tax_single(
    text: str, doc_type: str, client, model_name: str,
    max_tokens_override, temperature, retries
) -> dict:
    """
    Original single-call tax extraction path. Unchanged from pre-refactor.
    """
    from w2extract.core_chunked import _get_schema_hint
    schema_hint = _get_schema_hint(doc_type)
    system_prompt = get_prompt(doc_type, schema_hint)
    raw_json = call_llm(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=text,
        document_type=doc_type,
        max_tokens_override=max_tokens_override,
        temperature=temperature,
        retries=retries,
    )
    return json.loads(raw_json)


def _join_pages(pages: list[dict]) -> str:
    return "\n---PAGE BREAK---\n".join(p["text"] for p in pages)


def _extract_vision(pdf_path: str, client, vision_model: str, dpi: int) -> list[dict]:
    """Vision path: convert pages to PNG, send to VLM."""
    import base64
    from pdf2image import convert_from_path
    import tempfile, os, shutil

    tmpdir = tempfile.mkdtemp()
    try:
        page_images = convert_from_path(pdf_path, dpi=dpi, output_folder=tmpdir)
        results = []
        for i, img in enumerate(page_images):
            img_path = os.path.join(tmpdir, f"page_{i}.png")
            img.save(img_path)
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            response = client.chat.completions.create(
                model=vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text",
                         "text": "Extract all text from this financial document page. "
                                 "Preserve layout. Return plain text only."}
                    ]
                }],
                max_tokens=2048,
            )
            text = response.choices[0].message.content
            results.append({"page": i, "text": text, "rotated": False})
        return results
    finally:
        shutil.rmtree(tmpdir)


def _extract_tax_year(data: dict) -> int | None:
    """Best-effort tax year extraction from result dict."""
    import re
    for key in ("tax_year", "year"):
        if key in data:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    entity = data.get("entity", {})
    for key in ("period_end", "period_start"):
        val = entity.get(key, "")
        if val:
            m = re.search(r"(\d{4})", str(val))
            if m:
                return int(m.group(1))
    return None
```

---

## Canonical module layout after all patches

```
w2extract/
  __init__.py
  cli.py                  # argparse — unchanged interface
  detector.py             # get_extraction_strategy() + detect_financial_document_type()
  boundary_detector.py    # detect_boundaries() → list[DocumentSection]
  extractor.py            # extract_with_strategy() — pdfminer + pdfplumber
  ocr.py                  # ocr_pdf() — tesseract / easyocr / paddleocr + OSD rotation
  vision.py               # _extract_vision() lifted out of core if it grows
  chunker.py              # chunk_for_llm() — account-boundary / page-break / fixed
  core.py                 # extract_document() — main pipeline (refactored)
  core_chunked.py         # extract_chunked() + _merge_chunks() + helpers
  llm.py                  # call_llm() + get_token_budget()
  prompts.py              # get_prompt() + DocumentFamily + all SYSTEM_PROMPT constants
  verifier.py             # verify_section_totals() + compute_derived_fields()
  formatter.py            # to_csv() + to_lacerte() + to_txf() + to_json()
  schema.py               # all pydantic models including _parse_amount()
  
tests/
  fixtures/
    README.md             # fixture acquisition instructions
    generate_broken_pdf.py
    w2_digital_simple.pdf      # ✅ generate from QB sample company
    (others added as acquired)
  test_detector.py
  test_extractor.py
  test_verifier.py
  test_chunker.py
  test_merge.py
  test_core_tax.py        # validates tax path unchanged
  test_core_financial.py  # requires real fixtures; skipped if absent
```

---

## Change summary

| Issue | Resolution | Files changed |
|---|---|---|
| 1 + 2 | Inline `section_total` on section dict. Prompt corrected. | `prompts.py`, `schema.py` |
| 3 | `SoftwareMetadata` model on `FinancialEntity`. Prompt instructions added. | `schema.py`, `prompts.py` |
| 4 | `extract_chunked()` + `_merge_chunks()` with GL, P&L, Balance Sheet strategies. | `core_chunked.py` (new) |
| 5 | Fixture registry + acquisition README + ghostscript broken-PDF generation. | `tests/fixtures/` (new) |
| 6 | `core.py` refactored: tax path unchanged, financial path added as separate branch. | `core.py` |
