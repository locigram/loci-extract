"""TXF (Tax Exchange Format) v42 formatter.

TXF is the import format used by TurboTax, TaxAct, UltraTax. v1 implements
W-2, 1099-INT, 1099-DIV, 1099-R — the formats with the most stable and
documented TXF codes. Other doc types raise ``NotImplementedError``.

TXF v42 structure:
    V042
    A<application name>
    D<MM/DD/YYYY>
    ^
    T<record-type-code>        # e.g., T0511 for W-2 wages
    N<numeric/string value>
    ^

Each record is terminated by ``^`` on its own line. The file begins with
a small header block (version, app, date) terminated by ``^``, followed
by one or more record blocks.

Reference: https://turbotax.intuit.com/tax-tools/txf/ (spec v42).
"""

from __future__ import annotations

from datetime import date

from loci_extract.schema import Extraction


def _header() -> list[str]:
    return [
        "V042",
        "Aloci-extract",
        f"D{date.today().strftime('%m/%d/%Y')}",
        "^",
    ]


def _record(code: str, values: list[tuple[str, str]]) -> list[str]:
    lines = [f"T{code}"]
    for kind, val in values:
        lines.append(f"{kind}{val}")
    lines.append("^")
    return lines


def _fmt_money(v) -> str:
    if v is None:
        return "0.00"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _w2_records(doc: dict) -> list[str]:
    """W-2: T0511 for box 1 wages, T0512 for box 2 federal withheld,
    T0513 SS wages, T0514 SS tax, T0515 Medicare wages, T0516 Medicare tax.
    """
    data = doc.get("data", {}) or {}
    federal = data.get("federal", {}) or {}
    employer = data.get("employer", {}) or {}
    payer_name = employer.get("name", "") or ""
    out: list[str] = []
    mapping = [
        ("0511", federal.get("box1_wages")),
        ("0512", federal.get("box2_federal_withheld")),
        ("0513", federal.get("box3_ss_wages")),
        ("0514", federal.get("box4_ss_withheld")),
        ("0515", federal.get("box5_medicare_wages")),
        ("0516", federal.get("box6_medicare_withheld")),
    ]
    for code, amount in mapping:
        out.extend(_record(code, [("P", payer_name), ("$", _fmt_money(amount))]))
    return out


def _1099_int_records(doc: dict) -> list[str]:
    """1099-INT: T0612 box 1 interest, T0613 box 2 penalty,
    T0614 box 3 savings-bond interest, T0617 box 4 fed withheld,
    T0620 box 8 tax-exempt interest."""
    data = doc.get("data", {}) or {}
    payer = data.get("payer", {}) or {}
    payer_name = payer.get("name", "") or ""
    out: list[str] = []
    mapping = [
        ("0612", data.get("box1_interest_income")),
        ("0613", data.get("box2_early_withdrawal_penalty")),
        ("0614", data.get("box3_us_savings_bond_interest")),
        ("0617", data.get("box4_federal_withheld")),
        ("0620", data.get("box8_tax_exempt_interest")),
    ]
    for code, amount in mapping:
        out.extend(_record(code, [("P", payer_name), ("$", _fmt_money(amount))]))
    return out


def _1099_div_records(doc: dict) -> list[str]:
    """1099-DIV: T0624 box 1a ord dividends, T0625 qualified dividends,
    T0627 box 2a capital gain, T0630 box 4 fed withheld."""
    data = doc.get("data", {}) or {}
    payer = data.get("payer", {}) or {}
    payer_name = payer.get("name", "") or ""
    out: list[str] = []
    mapping = [
        ("0624", data.get("box1a_total_ordinary_dividends")),
        ("0625", data.get("box1b_qualified_dividends")),
        ("0627", data.get("box2a_total_capital_gain")),
        ("0630", data.get("box4_federal_withheld")),
    ]
    for code, amount in mapping:
        out.extend(_record(code, [("P", payer_name), ("$", _fmt_money(amount))]))
    return out


def _1099_r_records(doc: dict) -> list[str]:
    """1099-R: T0521 box 1 gross distribution, T0522 box 2a taxable,
    T0523 box 4 fed withheld."""
    data = doc.get("data", {}) or {}
    payer = data.get("payer", {}) or {}
    payer_name = payer.get("name", "") or ""
    out: list[str] = []
    mapping = [
        ("0521", data.get("box1_gross_distribution")),
        ("0522", data.get("box2a_taxable_amount")),
        ("0523", data.get("box4_federal_withheld")),
    ]
    for code, amount in mapping:
        out.extend(_record(code, [("P", payer_name), ("$", _fmt_money(amount))]))
    return out


_HANDLERS = {
    "W2": _w2_records,
    "1099-INT": _1099_int_records,
    "1099-DIV": _1099_div_records,
    "1099-R": _1099_r_records,
}


def format_extraction(extraction: Extraction) -> str:
    lines = _header()
    unsupported: set[str] = set()
    for doc in extraction.documents:
        doc_dict = doc.model_dump()
        doc_type = doc_dict.get("document_type", "")
        handler = _HANDLERS.get(doc_type)
        if handler is None:
            unsupported.add(doc_type)
            continue
        lines.extend(handler(doc_dict))
    if unsupported:
        raise NotImplementedError(
            f"TXF format not yet implemented for: {sorted(unsupported)}. "
            "Supported types: W-2, 1099-INT, 1099-DIV, 1099-R."
        )
    return "\n".join(lines) + "\n"


__all__ = ["format_extraction"]
