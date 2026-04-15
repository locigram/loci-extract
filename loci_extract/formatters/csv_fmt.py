"""CSV formatter — one row per document.

Multi-value fields (box12, state, local, box14, transactions) are serialized
as JSON strings inside their cells so no information is lost. Header columns
are stable: the union of interesting flat fields plus the JSON-serialized
nested fields.
"""

from __future__ import annotations

import csv
import io
import json

from loci_extract.schema import Extraction

_HEADER_COLUMNS = [
    "document_type",
    "tax_year",
    "is_summary_sheet",
    "is_corrected",
    "is_void",
    # Party block — we emit whichever "party" is most relevant for the type.
    "party_employer_or_payer_name",
    "party_employer_or_payer_tin",
    "party_employer_or_payer_address",
    "party_recipient_name",
    "party_recipient_tin_last4",
    "party_recipient_address",
    # W-2 federal boxes (other doc types leave these blank).
    "w2_box1_wages",
    "w2_box2_federal_withheld",
    "w2_box3_ss_wages",
    "w2_box4_ss_withheld",
    "w2_box5_medicare_wages",
    "w2_box6_medicare_withheld",
    # 1099 primary amount — the single "headline" number for the form.
    "primary_amount_field",
    "primary_amount",
    # JSON-serialized arrays
    "box12_json",
    "box14_json",
    "state_json",
    "local_json",
    "transactions_json",
    # Notes
    "notes_json",
]


# Which "party" fields we surface for each document type. Most 1099s use
# payer/recipient; W-2 uses employer/employee; K-1 uses partnership/partner etc.
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


# Primary single-dollar "headline" field per doc type used for the
# ``primary_amount`` column. Makes the CSV readable at a glance.
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
    """Safely fetch a dotted path from a dict, returning None if any step misses."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _row_for_doc(doc_dict: dict) -> dict:
    doc_type = doc_dict.get("document_type", "")
    meta = doc_dict.get("metadata", {}) or {}
    data = doc_dict.get("data", {}) or {}
    party_a_key, party_b_key = _PRIMARY_PARTY_MAP.get(doc_type, ("", ""))
    party_a = data.get(party_a_key, {}) if party_a_key else {}
    party_b = data.get(party_b_key, {}) if party_b_key else {}

    primary_path, primary_label = _PRIMARY_AMOUNT_MAP.get(doc_type, ("", ""))
    primary_amount = _dig(data, primary_path) if primary_path else None

    row = {
        "document_type": doc_type,
        "tax_year": doc_dict.get("tax_year", ""),
        "is_summary_sheet": meta.get("is_summary_sheet", False),
        "is_corrected": meta.get("is_corrected", False),
        "is_void": meta.get("is_void", False),
        "party_employer_or_payer_name": (party_a or {}).get("name", "") if isinstance(party_a, dict) else "",
        "party_employer_or_payer_tin": (party_a or {}).get("ein") or (party_a or {}).get("tin") or ""
        if isinstance(party_a, dict)
        else "",
        "party_employer_or_payer_address": (party_a or {}).get("address", "") if isinstance(party_a, dict) else "",
        "party_recipient_name": (party_b or {}).get("name", "") if isinstance(party_b, dict) else "",
        "party_recipient_tin_last4": (party_b or {}).get("ssn_last4") or (party_b or {}).get("tin_last4") or ""
        if isinstance(party_b, dict)
        else "",
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
    return row


def format_extraction(extraction: Extraction) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_HEADER_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for doc in extraction.documents:
        writer.writerow(_row_for_doc(doc.model_dump()))
    return buf.getvalue()


__all__ = ["format_extraction"]
