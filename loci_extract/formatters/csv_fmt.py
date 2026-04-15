"""CSV formatter.

Three shapes, dispatched by the document family:

- **Shape tax** — one row per document, tax-centric columns. Used for every
  `DocumentFamily.TAX` entry. Unchanged from pre-financial behavior.
- **Shape A — account rows** — used for `financial_simple`, `financial_multi`,
  `financial_reserve`. One row per account line item. Dynamic period columns
  pulled from `data.columns` (multi-column reports) or `["balance"]` /
  `["amount"]` for single-period reports. Writes section + section_total +
  top-level total rows with `row_type` in {"account", "subtotal", "total"}.
- **Shape B — transaction rows** — used for General Ledger. One row per
  transaction, with `balance_header` / `balance_footer` rows for per-account
  beginning/ending balances.
- **Shape B-aging** — used for AR/AP Aging. One row per customer/vendor + a
  totals row.

Mixed-family batches (rare — one PDF typically yields one family) fall back
to the tax shape. Callers who need per-doc CSV should extract one document
at a time.
"""

from __future__ import annotations

import csv
import io
import json

from loci_extract.prompts import DOCUMENT_FAMILY_MAP, DocumentFamily
from loci_extract.schema import Extraction

# ---------------------------------------------------------------------------
# Family classification helpers
# ---------------------------------------------------------------------------

_SHAPE_A_TYPES = {
    "BALANCE_SHEET",
    "INCOME_STATEMENT",
    "INCOME_STATEMENT_COMPARISON",
    "BUDGET_VS_ACTUAL",
    "TRIAL_BALANCE",
    "RESERVE_ALLOCATION",
    "QB_PROFIT_LOSS",
    "QB_BALANCE_SHEET",
}

_SHAPE_B_TXN_TYPES = {
    "GENERAL_LEDGER",
    "QB_GENERAL_LEDGER",
    "QB_TRANSACTION_LIST",
}

_SHAPE_B_AGING_TYPES = {
    "ACCOUNTS_RECEIVABLE_AGING",
    "ACCOUNTS_PAYABLE_AGING",
    "QB_AR_AGING",
    "QB_AP_AGING",
}


def _pick_shape(extraction: Extraction) -> str:
    """Pick CSV shape from the dominant doc type. Mixed → tax (fallback)."""
    types = {doc.document_type for doc in extraction.documents}
    if not types:
        return "tax"
    if types <= _SHAPE_A_TYPES:
        return "shape_a"
    if types <= _SHAPE_B_TXN_TYPES:
        return "shape_b_txn"
    if types <= _SHAPE_B_AGING_TYPES:
        return "shape_b_aging"
    # Any tax or mixed
    if any(DOCUMENT_FAMILY_MAP.get(t) == DocumentFamily.TAX for t in types):
        return "tax"
    return "tax"  # safe fallback


def format_extraction(extraction: Extraction) -> str:
    shape = _pick_shape(extraction)
    if shape == "shape_a":
        return _csv_account_rows(extraction)
    if shape == "shape_b_txn":
        return _csv_transactions(extraction)
    if shape == "shape_b_aging":
        return _csv_aging(extraction)
    return _csv_tax_rows(extraction)


# ===========================================================================
# Shape A — account rows (Balance Sheet / Income Statement / Trial Balance /
# Reserve Allocation / Multi-column)
# ===========================================================================

_SHAPE_A_FIXED_HEADERS = [
    "document_type",
    "entity_name",
    "software",
    "accounting_basis",
    "period_start",
    "period_end",
    "section",
    "subsection",
    "account_number",
    "account_name",
    "row_type",
]


def _get_period_columns(data: dict, document_type: str) -> list[str]:
    """Detect the dynamic period column keys for this document's data."""
    if "columns" in data and isinstance(data["columns"], list):
        keys = [c.get("key") for c in data["columns"] if isinstance(c, dict) and c.get("key")]
        if keys:
            return keys
    if document_type in ("BALANCE_SHEET", "RESERVE_ALLOCATION"):
        return ["balance"]
    if document_type == "TRIAL_BALANCE":
        return ["debit", "credit"]
    return ["amount"]


def _csv_account_rows(extraction: Extraction) -> str:
    # Union of period columns across all documents — so mixed BS + IS in one
    # extraction don't lose data. Preserves first-seen order.
    period_cols: list[str] = []
    seen: set[str] = set()
    for doc in extraction.documents:
        for col in _get_period_columns(doc.data or {}, doc.document_type):
            if col not in seen:
                seen.add(col)
                period_cols.append(col)

    headers = _SHAPE_A_FIXED_HEADERS + period_cols
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for doc in extraction.documents:
        doc_dict = doc.model_dump()
        _emit_shape_a_for_doc(writer, doc_dict, period_cols)
    return buf.getvalue()


def _emit_shape_a_for_doc(writer, doc_dict: dict, period_cols: list[str]):
    data = doc_dict.get("data", {}) or {}
    entity = data.get("entity", {}) or {}
    base = {
        "document_type": doc_dict.get("document_type", ""),
        "entity_name": entity.get("name", ""),
        "software": entity.get("software", ""),
        "accounting_basis": entity.get("accounting_basis", ""),
        "period_start": entity.get("period_start", ""),
        "period_end": entity.get("period_end", ""),
    }

    # Multi-column line_items format
    if "line_items" in data and isinstance(data["line_items"], list):
        for item in data["line_items"]:
            if not isinstance(item, dict):
                continue
            row = dict(base)
            row.update({
                "section": item.get("section", ""),
                "subsection": item.get("subsection", "") or "",
                "account_number": item.get("account_number", "") or "",
                "account_name": item.get("account_name", ""),
                "row_type": item.get("row_type", "account"),
            })
            values = item.get("values", {}) or {}
            for col in period_cols:
                v = values.get(col)
                row[col] = "" if v is None else v
            writer.writerow(row)
        # Multi-column reports don't duplicate totals outside line_items
        return

    # Section-based format (BS / IS / TB / Reserve)
    for top_name, top in _iter_top_sections(data):
        _walk_section(writer, base, top_name, top, period_cols, subsection="")


def _iter_top_sections(data: dict):
    """Yield (side_name_upper, side_dict) for top-level financial sides."""
    for key in ("assets", "liabilities", "equity",
                "income", "expenses", "other_income", "other_expenses",
                "components", "bank_accounts", "accounts"):
        if key in data:
            yield key.upper(), data[key]


def _walk_section(writer, base: dict, section_name: str, section_data,
                   period_cols: list[str], subsection: str = ""):
    """Recursively emit rows for one section or side."""
    if isinstance(section_data, list):
        for entry in section_data:
            if not isinstance(entry, dict):
                continue
            _write_account_row(writer, base, section_name, subsection, entry, period_cols)
        return

    if not isinstance(section_data, dict):
        return

    # Nested sections (flat list with section_name + accounts)
    for sub in section_data.get("sections", []) or []:
        if not isinstance(sub, dict):
            continue
        sub_name = sub.get("section_name", "")
        for acct in sub.get("accounts", []) or []:
            _write_account_row(writer, base, section_name, sub_name, acct, period_cols)
        # Recurse into subsections
        for deeper in sub.get("subsections", []) or []:
            _walk_section(writer, base, section_name, deeper, period_cols, subsection=sub_name)
        # Subtotal
        sub_total = sub.get("section_total")
        if sub_total is not None:
            _write_total_row(writer, base, section_name, sub_name, f"Total {sub_name}",
                              sub_total, "subtotal", period_cols)

    # Direct accounts at this level (no section wrapper)
    for acct in section_data.get("accounts", []) or []:
        _write_account_row(writer, base, section_name, subsection, acct, period_cols)

    # Section-level totals (flat fields like total_assets / total_liabilities)
    for total_key in (
        "section_total",
        "total_assets", "total_liabilities",
        "total_equity_reported", "total_equity",
        "total", "total_income", "total_expenses",
    ):
        val = section_data.get(total_key)
        if val is not None:
            label = total_key.replace("_", " ").title()
            _write_total_row(writer, base, section_name, subsection, label, val, "total", period_cols)


def _write_account_row(writer, base, section, subsection, account, period_cols):
    row = dict(base)
    row.update({
        "section": section,
        "subsection": subsection,
        "account_number": account.get("account_number", "") or "",
        "account_name": account.get("account_name") or account.get("component_name", ""),
        "row_type": "account",
    })
    # Populate period columns from the account dict
    for col in period_cols:
        # Common field names
        v = account.get(col)
        if v is None and col == "balance":
            v = account.get("current_balance")
        row[col] = "" if v is None else v
    writer.writerow(row)


def _write_total_row(writer, base, section, subsection, label, value, row_type, period_cols):
    row = dict(base)
    row.update({
        "section": section,
        "subsection": subsection,
        "account_number": "",
        "account_name": label,
        "row_type": row_type,
    })
    for col in period_cols:
        if col in ("balance", "amount"):
            row[col] = value
        else:
            row[col] = ""
    writer.writerow(row)


# ===========================================================================
# Shape B — transaction rows (General Ledger)
# ===========================================================================


def _csv_transactions(extraction: Extraction) -> str:
    headers = [
        "entity_name", "account_number", "account_name",
        "date", "type", "number", "name", "memo", "split",
        "debit", "credit", "balance", "row_type",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()

    for doc in extraction.documents:
        data = doc.data or {}
        entity_name = (data.get("entity") or {}).get("name", "")
        for account in data.get("accounts", []) or []:
            if not isinstance(account, dict):
                continue
            acct_num = account.get("account_number", "") or ""
            acct_name = account.get("account_name", "") or ""
            # Beginning balance
            writer.writerow({
                "entity_name": entity_name,
                "account_number": acct_num,
                "account_name": acct_name,
                "balance": account.get("beginning_balance", "") or "",
                "row_type": "balance_header",
            })
            for txn in account.get("transactions", []) or []:
                if not isinstance(txn, dict):
                    continue
                writer.writerow({
                    "entity_name": entity_name,
                    "account_number": acct_num,
                    "account_name": acct_name,
                    "date": txn.get("date", "") or "",
                    "type": txn.get("type", "") or "",
                    "number": txn.get("number", "") or "",
                    "name": txn.get("name", "") or "",
                    "memo": txn.get("memo", "") or "",
                    "split": txn.get("split", "") or "",
                    "debit": _fmt_amount(txn.get("debit")),
                    "credit": _fmt_amount(txn.get("credit")),
                    "balance": _fmt_amount(txn.get("balance")),
                    "row_type": txn.get("row_type", "transaction"),
                })
            writer.writerow({
                "entity_name": entity_name,
                "account_number": acct_num,
                "account_name": acct_name,
                "balance": account.get("ending_balance", "") or "",
                "row_type": "balance_footer",
            })

    return buf.getvalue()


def _fmt_amount(v) -> str:
    if v is None or v == "":
        return ""
    return str(v)


# ===========================================================================
# Shape B-aging — AR/AP aging reports
# ===========================================================================


def _csv_aging(extraction: Extraction) -> str:
    # Use the first document's buckets as header; they should all share
    # the same bucket list in practice.
    first = extraction.documents[0].data if extraction.documents else {}
    buckets = first.get("aging_buckets") or ["current", "1_to_30", "31_to_60", "61_to_90", "over_90"]
    headers = ["entity_name", "report_type", "name"] + list(buckets) + ["total", "row_type"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()

    for doc in extraction.documents:
        data = doc.data or {}
        entity_name = (data.get("entity") or {}).get("name", "")
        report_type = data.get("report_type", "AR")
        for row in data.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            out = {
                "entity_name": entity_name,
                "report_type": report_type,
                "name": row.get("name", ""),
                "row_type": "row",
            }
            for b in buckets:
                out[b] = _fmt_amount(row.get(b))
            out["total"] = _fmt_amount(row.get("total"))
            writer.writerow(out)
        totals = data.get("totals")
        if isinstance(totals, dict):
            out = {
                "entity_name": entity_name,
                "report_type": report_type,
                "name": "TOTAL",
                "row_type": "total",
            }
            for b in buckets:
                out[b] = _fmt_amount(totals.get(b))
            out["total"] = _fmt_amount(totals.get("total"))
            writer.writerow(out)

    return buf.getvalue()


# ===========================================================================
# Tax shape (preserved — original CSV format for W-2 / 1099 / K-1 etc.)
# ===========================================================================


_TAX_HEADER_COLUMNS = [
    "document_type",
    "tax_year",
    "is_summary_sheet",
    "is_corrected",
    "is_void",
    "party_employer_or_payer_name",
    "party_employer_or_payer_tin",
    "party_employer_or_payer_address",
    "party_recipient_name",
    "party_recipient_tin_last4",
    "party_recipient_address",
    "w2_box1_wages",
    "w2_box2_federal_withheld",
    "w2_box3_ss_wages",
    "w2_box4_ss_withheld",
    "w2_box5_medicare_wages",
    "w2_box6_medicare_withheld",
    "primary_amount_field",
    "primary_amount",
    "box12_json",
    "box14_json",
    "state_json",
    "local_json",
    "transactions_json",
    "notes_json",
]


_PRIMARY_PARTY_MAP: dict[str, tuple[str, str]] = {
    "W2": ("employer", "employee"),
    "1099-NEC": ("payer", "recipient"),
    "1099-MISC": ("payer", "recipient"),
    "1099-INT": ("payer", "recipient"),
    "1099-DIV": ("payer", "recipient"),
    "1099-B": ("payer", "recipient"),
    "1099-R": ("payer", "recipient"),
    "1099-G": ("payer", "recipient"),
    "1099-SA": ("payer", "recipient"),
    "1099-K": ("payer", "recipient"),
    "1099-S": ("filer", "transferor"),
    "1099-C": ("creditor", "debtor"),
    "1099-A": ("lender", "borrower"),
    "1098": ("recipient", "payer"),
    "1098-T": ("filer", "student"),
    "1098-E": ("recipient", "borrower"),
    "SSA-1099": ("payer", "beneficiary"),
    "RRB-1099": ("payer", "recipient"),
    "K-1 1065": ("partnership", "partner"),
    "K-1 1120-S": ("corporation", "shareholder"),
    "K-1 1041": ("estate_or_trust", "beneficiary"),
}


_PRIMARY_AMOUNT_MAP: dict[str, tuple[str, str]] = {
    "W2": ("federal.box1_wages", "Box 1 wages"),
    "1099-NEC": ("box1_nonemployee_compensation", "Box 1 nonemployee comp"),
    "1099-MISC": ("box1_rents", "Box 1 rents"),
    "1099-INT": ("box1_interest_income", "Box 1 interest"),
    "1099-DIV": ("box1a_total_ordinary_dividends", "Box 1a ordinary dividends"),
    "1099-B": ("summary.total_proceeds", "Total proceeds"),
    "1099-R": ("box1_gross_distribution", "Box 1 gross distribution"),
    "1099-G": ("box1_unemployment_compensation", "Box 1 unemployment"),
    "1099-SA": ("box1_gross_distribution", "Box 1 gross distribution"),
    "1099-K": ("box1a_gross_payment_card_transactions", "Box 1a gross payments"),
    "1099-S": ("box2_gross_proceeds", "Box 2 gross proceeds"),
    "1099-C": ("box2_amount_of_debt_discharged", "Box 2 debt discharged"),
    "1099-A": ("box2_balance_of_principal_outstanding", "Box 2 principal"),
    "1098": ("box1_mortgage_interest", "Box 1 mortgage interest"),
    "1098-T": ("box1_payments_received", "Box 1 tuition payments"),
    "1098-E": ("box1_student_loan_interest", "Box 1 student loan interest"),
    "SSA-1099": ("box5_net_benefits", "Box 5 net benefits"),
    "RRB-1099": ("box1_railroad_retirement_tier1", "Box 1 tier 1"),
    "K-1 1065": ("box1_ordinary_business_income", "Box 1 ordinary bus. income"),
    "K-1 1120-S": ("box1_ordinary_business_income", "Box 1 ordinary bus. income"),
    "K-1 1041": ("box1_interest_income", "Box 1 interest income"),
}


def _dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _row_for_tax_doc(doc_dict: dict) -> dict:
    doc_type = doc_dict.get("document_type", "")
    meta = doc_dict.get("metadata", {}) or {}
    data = doc_dict.get("data", {}) or {}
    party_a_key, party_b_key = _PRIMARY_PARTY_MAP.get(doc_type, ("", ""))
    party_a = data.get(party_a_key, {}) if party_a_key else {}
    party_b = data.get(party_b_key, {}) if party_b_key else {}

    primary_path, primary_label = _PRIMARY_AMOUNT_MAP.get(doc_type, ("", ""))
    primary_amount = _dig(data, primary_path) if primary_path else None

    return {
        "document_type": doc_type,
        "tax_year": doc_dict.get("tax_year", ""),
        "is_summary_sheet": meta.get("is_summary_sheet", False),
        "is_corrected": meta.get("is_corrected", False),
        "is_void": meta.get("is_void", False),
        "party_employer_or_payer_name": (party_a or {}).get("name", "") if isinstance(party_a, dict) else "",
        "party_employer_or_payer_tin": (party_a or {}).get("ein") or (party_a or {}).get("tin") or ""
        if isinstance(party_a, dict) else "",
        "party_employer_or_payer_address": (party_a or {}).get("address", "") if isinstance(party_a, dict) else "",
        "party_recipient_name": (party_b or {}).get("name", "") if isinstance(party_b, dict) else "",
        "party_recipient_tin_last4": (party_b or {}).get("ssn_last4") or (party_b or {}).get("tin_last4") or ""
        if isinstance(party_b, dict) else "",
        "party_recipient_address": (party_b or {}).get("address", "") if isinstance(party_b, dict) else "",
        "w2_box1_wages": _dig(data, "federal.box1_wages") or "",
        "w2_box2_federal_withheld": _dig(data, "federal.box2_federal_withheld") or "",
        "w2_box3_ss_wages": _dig(data, "federal.box3_ss_wages") or "",
        "w2_box4_ss_withheld": _dig(data, "federal.box4_ss_withheld") or "",
        "w2_box5_medicare_wages": _dig(data, "federal.box5_medicare_wages") or "",
        "w2_box6_medicare_withheld": _dig(data, "federal.box6_medicare_withheld") or "",
        "primary_amount_field": primary_label,
        "primary_amount": primary_amount if primary_amount is not None else "",
        "box12_json": json.dumps(data.get("box12", [])),
        "box14_json": json.dumps(data.get("box14_other", [])),
        "state_json": json.dumps(data.get("state", [])),
        "local_json": json.dumps(data.get("local", [])),
        "transactions_json": json.dumps(data.get("transactions", [])),
        "notes_json": json.dumps(meta.get("notes", [])),
    }


def _csv_tax_rows(extraction: Extraction) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_TAX_HEADER_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for doc in extraction.documents:
        writer.writerow(_row_for_tax_doc(doc.model_dump()))
    return buf.getvalue()


__all__ = ["format_extraction"]
