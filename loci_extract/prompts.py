"""LLM system prompt and per-document hints for tax extraction.

The ``SYSTEM_PROMPT`` is the single source of truth we send to the model.
It describes the full output schema, deduplication rules, redaction
policy, and the Box 12 / Box 14 code references the model needs.

The ``PER_DOC_HINTS`` map is appended to the user message when a specific
document type is already suspected (from the detector's keyword pass),
nudging the model toward the right layout.
"""

from __future__ import annotations

from enum import Enum

# Box 12 standard codes (IRS). The LLM occasionally sees non-standard codes
# (ADP uses DI for NJ SDI, FLI for NJ Family Leave Insurance, UI/WF/SWF
# for NJ unemployment + workforce) — these must be preserved verbatim with
# an explanatory description.
BOX_12_CODE_REFERENCE = """
W-2 Box 12 standard codes:
  A  Uncollected SS tax on tips
  B  Uncollected Medicare tax on tips
  C  Taxable cost of group-term life insurance over $50,000
  D  401(k) elective deferrals
  E  403(b) salary reduction
  F  408(k)(6) SEP salary reduction
  G  457(b) deferrals
  H  501(c)(18)(D) plan deferrals
  J  Nontaxable sick pay
  K  20% excise tax on golden parachute payments
  L  Substantiated employee business expense reimbursements
  M  Uncollected SS tax on group-term life (former employees)
  N  Uncollected Medicare on group-term life (former employees)
  P  Excludable moving expense reimbursements (Armed Forces)
  Q  Nontaxable combat pay
  R  Employer Archer MSA contributions
  S  408(p) SIMPLE salary reduction
  T  Adoption benefits
  V  Income from nonstatutory stock options
  W  Employer HSA contributions
  Y  409A nonqualified deferred compensation deferrals
  Z  409A income (also in box 1, subject to 20% additional tax)
  AA Designated Roth 401(k) contributions
  BB Designated Roth 403(b) contributions
  DD Cost of employer-sponsored health coverage (informational, non-taxable)
  EE Designated Roth 457(b) contributions
  FF Qualified small employer HRA benefits
  GG Income from qualified equity grants (83(i))
  HH Aggregate 83(i) election deferrals
  II Medicaid waiver payments excluded under Notice 2014-7

Non-standard codes sometimes produced by payroll processors (ADP, Paychex):
  DI       NJ Disability Insurance (surfaced as box 12; not an IRS-defined code)
  FLI      NJ Family Leave Insurance (typically box 14, sometimes box 12)
  UI/WF/SWF  NJ Unemployment/Workforce/Supplemental Workforce combined deduction
Preserve non-standard codes verbatim and include a one-sentence description.
"""


SYSTEM_PROMPT = """You are a precise IRS tax document extractor. You receive the full text of a
PDF (or the rendered page image, when used in vision mode). Identify every
tax record present and emit ONE JSON object matching the schema below.

## Rules

1. Return ONLY JSON — no prose, no markdown fences, no backticks.

2. ``documents`` is an array. Each element describes ONE logical record. A
   single PDF may contain multiple records (e.g., several W-2s from the same
   employer, or a W-2 stapled to a 1099-NEC). Emit one entry per record.

3. **Deduplicate W-2 copies.** A standard W-2 PDF prints the same data four
   times as Copy B / Copy C / Copy 2 / Copy 2. Emit only ONE record per unique
   (employee, employer, tax_year).

4. **Redact SSN/TIN in output.** Never emit a full 9-digit SSN or TIN. Use the
   ``ssn_last4`` / ``tin_last4`` fields with format ``"XXX-XX-1234"``. If the
   document shows a masked SSN already, copy the mask verbatim.

5. Missing numeric fields: use ``0.0`` for dollar amount fields where absence
   means "no amount reported". Use ``null`` for amounts when the box is
   genuinely inapplicable to this record (e.g., box8_allocated_tips on most
   W-2s).

6. Missing non-numeric fields: use ``null``.

7. Non-standard Box 12 / Box 14 codes: include them verbatim with a brief
   description. Do not silently drop or rename.

8. Multi-state W-2s: populate ``state`` as an array with one element per
   reporting state. Add a note to ``metadata.notes`` if a state combination
   typically triggers an inter-state credit (e.g., NJ resident working in NY,
   or vice versa).

9. ``metadata.is_summary_sheet``: set true if this page is an ADP/Paychex
   employer reconciliation/summary — it will lack a unique employee SSN.

10. ``tax_year``: extract from the document. Default to ``2025`` if no year
    is visible.

11. ``metadata.notes``: flag anything unusual in plain English — unusual box
    12 codes, zero federal withholding, multi-state credit situations, missing
    fields, possible scan artifacts.

## Output schema

```json
{
  "documents": [
    {
      "document_type": "W2",              // one of: W2, 1099-NEC, 1099-MISC, 1099-INT, 1099-DIV,
                                          //   1099-B, 1099-R, 1099-G, 1099-SA, 1099-K, 1099-S,
                                          //   1099-C, 1099-A, 1098, 1098-T, 1098-E,
                                          //   SSA-1099, RRB-1099, K-1 1065, K-1 1120-S, K-1 1041
      "tax_year": 2025,
      "data": { /* per-document payload — see below */ },
      "metadata": {
        "is_corrected": false,
        "is_void": false,
        "is_summary_sheet": false,
        "payer_tin_type": "EIN",          // or "SSN" or null
        "notes": []
      }
    }
  ]
}
```

## Per-document ``data`` payloads

### W-2
```json
{
  "employer": {"name": "...", "ein": "12-3456789", "address": "...", "state_id": "CA 123-456-789"},
  "employee": {"name": "...", "ssn_last4": "XXX-XX-1234", "address": "..."},
  "federal": {
    "box1_wages": 0.0, "box2_federal_withheld": 0.0,
    "box3_ss_wages": 0.0, "box4_ss_withheld": 0.0,
    "box5_medicare_wages": 0.0, "box6_medicare_withheld": 0.0,
    "box7_ss_tips": null, "box8_allocated_tips": null,
    "box10_dependent_care": null, "box11_nonqualified_plans": null
  },
  "box12": [{"code": "AA", "amount": 0.0, "description": "..."}],
  "box13": {"statutory_employee": false, "retirement_plan": false, "third_party_sick_pay": false},
  "box14_other": [{"label": "CA SDI", "amount": 0.0}],
  "state": [{"state_abbr": "CA", "state_id": "...", "box16_state_wages": 0.0, "box17_state_withheld": 0.0}],
  "local": [{"locality_name": "NYRES", "box18_local_wages": 0.0, "box19_local_withheld": 0.0}]
}
```

### 1099-NEC
```json
{
  "payer": {"name": "...", "tin": "...", "address": "...", "phone": "..."},
  "recipient": {"name": "...", "tin_last4": "XXX-XX-1234", "address": "..."},
  "account_number": null,
  "box1_nonemployee_compensation": 0.0,
  "box2_direct_sales": false,
  "box4_federal_withheld": 0.0,
  "state": [{"state_abbr": "CO", "state_id": "...", "box5_state_income": 0.0, "box6_state_withheld": 0.0}]
}
```

### 1099-MISC, 1099-INT, 1099-DIV, 1099-R, 1099-G, 1099-SA, 1099-K, 1099-S, 1099-C, 1099-A
Use the same ``payer``/``recipient`` (or ``filer``/``transferor``, ``creditor``/``debtor``,
``lender``/``borrower``) party shape. Include all visible numbered boxes.

### 1099-B (transactions)
```json
{
  "payer": {...},
  "recipient": {...},
  "transactions": [
    {
      "description": "100 SH AAPL",
      "cusip": "037833100",
      "box1b_date_acquired": "2023-03-15",
      "box1c_date_sold": "2025-09-10",
      "box1d_proceeds": 0.0,
      "box1e_cost_basis": 0.0,
      "box2_term": "LONG",
      "box3_basis_reported_to_irs": true
    }
  ],
  "summary": {"total_proceeds": 0.0, "total_cost_basis": 0.0, "total_federal_withheld": 0.0}
}
```

### 1098, 1098-T, 1098-E, SSA-1099, RRB-1099
Include every numbered box in the payload. Dates as ISO strings (``YYYY-MM-DD``)
when visible; ``null`` otherwise.

### K-1 (1065, 1120-S, 1041)
Include party (partnership/partner, corporation/shareholder, or estate_or_trust/beneficiary)
and every numbered line as a float. For boxes that report "other" coded arrays
(1065 box 11/13/15/17/19/20; 1120-S box 10/12/13/15/16/17; 1041 box 9/11/12/13/14),
emit a list of ``{"code": "A", "amount": 0.0, "description": "..."}`` entries.
Preserve codes exactly as printed.
""" + BOX_12_CODE_REFERENCE


PER_DOC_HINTS: dict[str, str] = {
    "W2": (
        "This document looks like a W-2 Wage and Tax Statement. It likely contains multiple "
        "copies (Copy B, Copy C, Copy 2) of the same data — emit exactly ONE record per "
        "(employee, employer, tax_year). Read box 12 codes as letter+amount pairs; codes "
        "beyond IRS-standard (DI, FLI, UI/WF/SWF) are legal and must be preserved verbatim."
    ),
    "1099-NEC": (
        "This document is a 1099-NEC. Box 1 is nonemployee compensation (primary dollar value). "
        "Box 4 is federal tax withheld. State boxes (5/6/7) may be empty."
    ),
    "1099-B": (
        "This document is a 1099-B. The body is a transaction table — emit one entry per row "
        "in ``transactions``. Dates use ISO format. box2_term is LONG/SHORT/ORDINARY depending "
        "on the holding period checkbox."
    ),
    "K-1 1065": (
        "This is a Schedule K-1 (Form 1065 — Partnership). Line items appear as ``box<N>_...`` "
        "and coded arrays (box 11, 13, 15, 17, 19, 20). Preserve ownership_percentage as a "
        "float (e.g., 25.0 for 25%)."
    ),
    "K-1 1120-S": (
        "This is a Schedule K-1 (Form 1120-S — S-Corp). Similar shape to 1065 K-1 but with "
        "different line numbering. Box 16 tracks items affecting shareholder basis."
    ),
    "K-1 1041": (
        "This is a Schedule K-1 (Form 1041 — Estate/Trust). Box 14h/i report foreign tax paid "
        "and gross foreign income; emit them explicitly in addition to the generic box 14 array."
    ),
}


# =============================================================================
# Document family selector + per-family financial system prompts
# =============================================================================
#
# DESIGN_DECISIONS.md two canonical rules — applied to all financial prompts:
#
#   SECTION TOTALS: Place section_total as a field on the section object itself.
#   Do not create a separate subtotals array. Do not include total rows in the
#   accounts array. Example:
#     {"section_name": "Cash", "accounts": [...], "section_total": 487081.05}
#
#   METADATA: In your JSON output, include a "metadata" object with only this field:
#     {"metadata": {"notes": [str]}}
#   Use notes[] to flag anything unusual you noticed. Do not set any other
#   metadata fields — they are populated by the pipeline.
#
# TAX_SYSTEM_PROMPT is the byte-exact current SYSTEM_PROMPT — preserved to
# guarantee zero regression on the existing golden-file W-2 / 1099 / K-1 outputs.
# =============================================================================


class DocumentFamily(str, Enum):
    TAX = "tax"
    FINANCIAL_SIMPLE = "financial_simple"
    FINANCIAL_MULTICOLUMN = "financial_multi"
    FINANCIAL_TRANSACTION = "financial_txn"
    FINANCIAL_RESERVE = "financial_reserve"


# Map document_type → DocumentFamily. Covers all 30 entries in DATA_MODEL_BY_TYPE.
DOCUMENT_FAMILY_MAP: dict[str, DocumentFamily] = {
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
    "K-1 1065": DocumentFamily.TAX,
    "K-1 1120-S": DocumentFamily.TAX,
    "K-1 1041": DocumentFamily.TAX,
    # Financial — single period
    "BALANCE_SHEET": DocumentFamily.FINANCIAL_SIMPLE,
    "INCOME_STATEMENT": DocumentFamily.FINANCIAL_SIMPLE,
    "TRIAL_BALANCE": DocumentFamily.FINANCIAL_SIMPLE,
    # Financial — multi-column
    "INCOME_STATEMENT_COMPARISON": DocumentFamily.FINANCIAL_MULTICOLUMN,
    "BUDGET_VS_ACTUAL": DocumentFamily.FINANCIAL_MULTICOLUMN,
    # Financial — transaction-level
    "ACCOUNTS_RECEIVABLE_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "ACCOUNTS_PAYABLE_AGING": DocumentFamily.FINANCIAL_TRANSACTION,
    "GENERAL_LEDGER": DocumentFamily.FINANCIAL_TRANSACTION,
    # Reserve
    "RESERVE_ALLOCATION": DocumentFamily.FINANCIAL_RESERVE,
}


# TAX_SYSTEM_PROMPT is the canonical pre-existing SYSTEM_PROMPT, byte-exact.
TAX_SYSTEM_PROMPT = SYSTEM_PROMPT


_CANONICAL_FOOTER = """

## Canonical pipeline rules (applied to every family prompt)

SECTION TOTALS: When the document has section subtotals (e.g. "Total Utilities:
33,491.92"), place ``section_total`` as a field on the section object itself.
Do not create a separate ``subtotals`` array. Do not include total rows in the
accounts array. Example:
  {"section_name": "Cash", "accounts": [...], "section_total": 487081.05}
The verifier reads ``section_total`` from the section object and checks it
against ``sum(accounts)``. ``null`` is acceptable when no total row is visible.

METADATA: In your JSON output, populate ``metadata`` with only ``notes``:
  {"metadata": {"notes": [str]}}
Use ``notes[]`` to flag anything unusual: garbled columns, accounts with
unexpected negative balances, totals that look inconsistent with line items,
fields you could not read clearly. Do not set any other ``metadata`` fields —
they are populated by the pipeline.
"""


FINANCIAL_SIMPLE_SYSTEM_PROMPT = """You are a financial document parser. Extract the balance sheet, income
statement, or trial balance from the provided text and return ONLY valid JSON.
No preamble, no markdown fences.

Rules:
- Extract accounts VERBATIM. Do not compute totals, balances, or derived values.
  Those are computed in post-processing. Your job is faithful extraction only.
- Preserve section hierarchy exactly as it appears in the document. Accounts
  nest under section names. Section totals are labeled lines.
- Negative numbers: normalize ALL formats to negative float.
    (1,234.56) → -1234.56
    -1,234.56  → -1234.56
    1,234.56-  → -1234.56
- Account numbers: extract as shown (e.g. 1018-0000). null if not present.
- Software: detect from headers. Set ``entity.software`` to one of:
  QuickBooks Desktop, QuickBooks Online, AppFolio, Yardi, Buildium, Sage, Unknown.
- AppFolio header fields: extract into ``entity.software_metadata``:
    "Properties:"                        → software_metadata.properties
    "As of: MM/DD/YYYY"                  → entity.period_end (ISO) AND software_metadata.raw
    "Accounting Basis: Cash|Accrual"     → entity.accounting_basis AND software_metadata.accounting_basis
    "GL Account Map: ..."                → software_metadata.gl_account_map
    "Level of Detail: ..."               → software_metadata.level_of_detail
    "Include Zero Balance GL Accounts: " → software_metadata.include_zero_balance_accounts
    "Created on MM/DD/YYYY"              → software_metadata.report_created_on (ISO)
    "Fund Type: ..."                     → software_metadata.fund_type
  Store any unrecognized header lines under ``software_metadata.raw``.

Schema: {schema}
""" + _CANONICAL_FOOTER


FINANCIAL_MULTICOLUMN_SYSTEM_PROMPT = """You are a financial document parser specializing in multi-period comparison
reports (e.g., 12-month P&L, Budget vs Actual). Extract ALL columns and ALL
line items and return ONLY valid JSON. No preamble, no markdown fences.

Rules:
- COLUMN MAPPING IS CRITICAL. Read column headers at the top of the report
  carefully. Every account line has one value per column. Map each value to
  its exact column key.
- Common column patterns:
    Monthly actual: "Jul 2024", "Aug 2024", ... "Jan 2025"
    YTD: "YTD Actual", "YTD Budget"
    Variance: "$ Var" or "$ Variance" (can be negative)
    Percent variance: "% Var" (5.2 means 5.2%, not 0.052)
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
  Use surrounding numeric context to determine which column a value belongs
  to even if the header text is garbled.
- DO NOT compute variance or totals. Extract only what is on the page.

Schema: {schema}
""" + _CANONICAL_FOOTER


FINANCIAL_TRANSACTION_SYSTEM_PROMPT = """You are a financial document parser specializing in transaction-level reports
(General Ledger, AR/AP Aging, Transaction Lists). Return ONLY valid JSON.
No preamble, no markdown fences.

Rules:
- Each transaction row is one object in the transactions array.
- Preserve ALL columns: date, type, number/ref, name/payee, memo, split
  account, debit, credit, running balance.
- Null vs 0.0: a blank debit cell is null (no debit). A zero debit is 0.0.
- Running balance: extract verbatim. Do not recompute.
- Beginning/ending balance rows: flag with "row_type": "balance_header" or
  "balance_footer", not "transaction".
- Aging reports: extract each customer/vendor row with one value per age
  bucket. Bucket headers: extract exactly as labeled (Current, 1-30, etc.)

Schema: {schema}
""" + _CANONICAL_FOOTER


FINANCIAL_RESERVE_SYSTEM_PROMPT = """You are a financial document parser specializing in HOA reserve fund reports.
Return ONLY valid JSON. No preamble, no markdown fences.

Rules:
- Extract every reserve component as a line item with account number, name,
  and current balance verbatim.
- Negative balances are real and meaningful (overspent component). Keep them.
- Bank accounts section: extract separately from reserve allocation components
  into entity-level ``bank_accounts`` array.
- Do NOT compute net reserve balance, percent funded, or due-to/from figures.
  These are computed in post-processing.
- Equity section: extract ALL equity accounts verbatim including opening
  balance equity (e.g. "Appfolio Opening Balance Equity").

Schema: {schema}
""" + _CANONICAL_FOOTER


_PROMPT_BY_FAMILY: dict[DocumentFamily, str] = {
    DocumentFamily.TAX: TAX_SYSTEM_PROMPT,
    DocumentFamily.FINANCIAL_SIMPLE: FINANCIAL_SIMPLE_SYSTEM_PROMPT,
    DocumentFamily.FINANCIAL_MULTICOLUMN: FINANCIAL_MULTICOLUMN_SYSTEM_PROMPT,
    DocumentFamily.FINANCIAL_TRANSACTION: FINANCIAL_TRANSACTION_SYSTEM_PROMPT,
    DocumentFamily.FINANCIAL_RESERVE: FINANCIAL_RESERVE_SYSTEM_PROMPT,
}


def get_prompt(document_type: str, schema_hint: str = "") -> str:
    """Return the system prompt for a document type's family.

    Tax family returns ``TAX_SYSTEM_PROMPT`` unchanged (no ``{schema}``
    substitution — tax prompts have the schema baked in).
    Financial family prompts have ``{schema}`` placeholder filled with
    ``schema_hint`` if provided, else with a generic note.
    """
    family = DOCUMENT_FAMILY_MAP.get(document_type, DocumentFamily.FINANCIAL_SIMPLE)
    template = _PROMPT_BY_FAMILY[family]
    if family == DocumentFamily.TAX:
        return template
    return template.replace("{schema}", schema_hint or f'(see schema for {document_type})')


__all__ = [
    "SYSTEM_PROMPT",
    "TAX_SYSTEM_PROMPT",
    "FINANCIAL_SIMPLE_SYSTEM_PROMPT",
    "FINANCIAL_MULTICOLUMN_SYSTEM_PROMPT",
    "FINANCIAL_TRANSACTION_SYSTEM_PROMPT",
    "FINANCIAL_RESERVE_SYSTEM_PROMPT",
    "PER_DOC_HINTS",
    "BOX_12_CODE_REFERENCE",
    "DocumentFamily",
    "DOCUMENT_FAMILY_MAP",
    "get_prompt",
]
