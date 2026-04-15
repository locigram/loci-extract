"""Lacerte tab-delimited import formatter.

One row per document. Tab-separated. No header row. Field order per-type
follows EXTRACT_SPEC.md. Multi-state W-2s emit one row per state (federal
boxes repeat). Box 12 and Box 14 are padded to 4 slots each. SSNs in
output are masked to last-4; user fills the first five digits before
import into Lacerte.

v1 supports W-2, 1099-NEC, 1099-INT, 1099-DIV, 1099-R. Other doc types
raise a NotImplementedError so users know the format isn't yet defined
(vs silent garbage).
"""

from __future__ import annotations

from loci_extract.schema import Extraction


def _masked_ssn(last4: str | None) -> str:
    if not last4:
        return ""
    # Lacerte expects digits + placeholders for the user to fill in.
    # Format: XXXXX + last 4
    tail = last4[-4:] if len(last4) >= 4 else last4
    return f"XXXXX{tail}"


def _pad(items: list, size: int, filler):
    return list(items) + [filler] * max(0, size - len(items))


def _fmt_amount(v) -> str:
    if v is None or v == "":
        return "0.00"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _w2_rows(doc: dict) -> list[list[str]]:
    data = doc.get("data", {})
    employer = data.get("employer", {}) or {}
    employee = data.get("employee", {}) or {}
    federal = data.get("federal", {}) or {}
    box12 = data.get("box12", []) or []
    box14 = data.get("box14_other", []) or []
    state_rows = data.get("state", []) or [{}]
    local_rows = data.get("local", []) or [{}]

    # Pad box12 → 4 (code, amount) pairs; box14 → 3 (label, amount) pairs.
    b12 = _pad(box12, 4, {"code": "", "amount": 0.0})
    b14 = _pad(box14, 3, {"label": "", "amount": 0.0})

    common_prefix = [
        _masked_ssn(employee.get("ssn_last4")),
        employer.get("name", ""),
        employer.get("ein", "") or "",
        _fmt_amount(federal.get("box1_wages")),
        _fmt_amount(federal.get("box2_federal_withheld")),
        _fmt_amount(federal.get("box3_ss_wages")),
        _fmt_amount(federal.get("box4_ss_withheld")),
        _fmt_amount(federal.get("box5_medicare_wages")),
        _fmt_amount(federal.get("box6_medicare_withheld")),
        # Box 12 slots
        b12[0].get("code", ""),
        _fmt_amount(b12[0].get("amount")),
        b12[1].get("code", ""),
        _fmt_amount(b12[1].get("amount")),
        b12[2].get("code", ""),
        _fmt_amount(b12[2].get("amount")),
        b12[3].get("code", ""),
        _fmt_amount(b12[3].get("amount")),
        # Box 14 slots
        b14[0].get("label", ""),
        _fmt_amount(b14[0].get("amount")),
        b14[1].get("label", ""),
        _fmt_amount(b14[1].get("amount")),
        b14[2].get("label", ""),
        _fmt_amount(b14[2].get("amount")),
    ]

    rows: list[list[str]] = []
    # Emit one row per state (as spec calls for). If no state, emit a single
    # blank-state row so downstream code always gets a W-2 line.
    states = state_rows or [{}]
    locals_ = local_rows or [{}]
    paired = zip(states, locals_ + [{}] * max(0, len(states) - len(locals_)), strict=False)
    for state, local in paired:
        row = list(common_prefix) + [
            state.get("state_abbr", ""),
            state.get("state_id", "") or "",
            _fmt_amount(state.get("box16_state_wages")),
            _fmt_amount(state.get("box17_state_withheld")),
            local.get("locality_name", "") or "",
            _fmt_amount(local.get("box18_local_wages")),
            _fmt_amount(local.get("box19_local_withheld")),
        ]
        rows.append(row)
    return rows


def _1099_nec_row(doc: dict) -> list[list[str]]:
    data = doc.get("data", {})
    payer = data.get("payer", {}) or {}
    recipient = data.get("recipient", {}) or {}
    # Use first state line only for NEC (one row per form).
    state = (data.get("state") or [{}])[0]
    return [[
        _masked_ssn(recipient.get("tin_last4")),
        payer.get("name", ""),
        payer.get("tin", "") or "",
        _fmt_amount(data.get("box1_nonemployee_compensation")),
        _fmt_amount(data.get("box4_federal_withheld")),
        state.get("state_abbr", "") if isinstance(state, dict) else "",
        state.get("state_id", "") if isinstance(state, dict) else "",
        _fmt_amount(state.get("box5_state_income") if isinstance(state, dict) else None),
        _fmt_amount(state.get("box6_state_withheld") if isinstance(state, dict) else None),
    ]]


def _1099_int_row(doc: dict) -> list[list[str]]:
    data = doc.get("data", {})
    payer = data.get("payer", {}) or {}
    recipient = data.get("recipient", {}) or {}
    state = (data.get("state") or [{}])[0]
    return [[
        _masked_ssn(recipient.get("tin_last4")),
        payer.get("name", ""),
        payer.get("tin", "") or "",
        _fmt_amount(data.get("box1_interest_income")),
        _fmt_amount(data.get("box2_early_withdrawal_penalty")),
        _fmt_amount(data.get("box3_us_savings_bond_interest")),
        _fmt_amount(data.get("box4_federal_withheld")),
        _fmt_amount(data.get("box8_tax_exempt_interest")),
        state.get("state_abbr", "") if isinstance(state, dict) else "",
        state.get("state_id", "") if isinstance(state, dict) else "",
        _fmt_amount(state.get("box16_state_income") if isinstance(state, dict) else None),
        _fmt_amount(state.get("box17_state_withheld") if isinstance(state, dict) else None),
    ]]


def _1099_div_row(doc: dict) -> list[list[str]]:
    data = doc.get("data", {})
    payer = data.get("payer", {}) or {}
    recipient = data.get("recipient", {}) or {}
    state = (data.get("state") or [{}])[0]
    return [[
        _masked_ssn(recipient.get("tin_last4")),
        payer.get("name", ""),
        payer.get("tin", "") or "",
        _fmt_amount(data.get("box1a_total_ordinary_dividends")),
        _fmt_amount(data.get("box1b_qualified_dividends")),
        _fmt_amount(data.get("box2a_total_capital_gain")),
        _fmt_amount(data.get("box4_federal_withheld")),
        _fmt_amount(data.get("box5_section199a_dividends")),
        state.get("state_abbr", "") if isinstance(state, dict) else "",
        state.get("state_id", "") if isinstance(state, dict) else "",
        _fmt_amount(state.get("box14_state_income") if isinstance(state, dict) else None),
        _fmt_amount(state.get("box15_state_withheld") if isinstance(state, dict) else None),
    ]]


def _1099_r_row(doc: dict) -> list[list[str]]:
    data = doc.get("data", {})
    payer = data.get("payer", {}) or {}
    recipient = data.get("recipient", {}) or {}
    return [[
        _masked_ssn(recipient.get("tin_last4")),
        payer.get("name", ""),
        payer.get("tin", "") or "",
        _fmt_amount(data.get("box1_gross_distribution")),
        _fmt_amount(data.get("box2a_taxable_amount")),
        _fmt_amount(data.get("box4_federal_withheld")),
        data.get("box7_distribution_code", "") or "",
        "Y" if data.get("box7_ira_sep_simple") else "",
        "",  # state_abbr placeholder — not a standard R field
        data.get("box15_state_id", "") or "",
        _fmt_amount(data.get("box14_state_withheld")),
        _fmt_amount(data.get("box16_state_distribution")),
    ]]


_HANDLERS = {
    "W2": _w2_rows,
    "1099-NEC": _1099_nec_row,
    "1099-INT": _1099_int_row,
    "1099-DIV": _1099_div_row,
    "1099-R": _1099_r_row,
}


def format_extraction(extraction: Extraction) -> str:
    lines: list[str] = []
    unsupported: set[str] = set()
    for doc in extraction.documents:
        doc_dict = doc.model_dump()
        doc_type = doc_dict.get("document_type", "")
        handler = _HANDLERS.get(doc_type)
        if handler is None:
            unsupported.add(doc_type)
            continue
        for row in handler(doc_dict):
            lines.append("\t".join(str(cell) for cell in row))
    if unsupported:
        raise NotImplementedError(
            f"Lacerte format not yet implemented for: {sorted(unsupported)}. "
            "Supported types: W-2, 1099-NEC, 1099-INT, 1099-DIV, 1099-R."
        )
    return "\n".join(lines) + ("\n" if lines else "")


__all__ = ["format_extraction"]
