from __future__ import annotations

import re
from collections import OrderedDict

from app.normalization import find_first_date, normalize_whitespace, parse_amount
from app.review import build_review_metadata
from app.schemas import ExtractionPayload, StructuredDocument
from app.structured.common import first_source_pages, get_text_lines, snippet_around_match

_ACCOUNT_RE = re.compile(r"^\d{4}-\d{4}$")
_BALANCE_RE = re.compile(r"^[($\-\d,\.\s)]+$")
_REPORT_TYPE_PATTERNS = {
    "balance_sheet": [r"\bbalance sheet\b"],
    "income_statement": [r"\bincome statement\b", r"\bprofit and loss\b"],
}

_SECTION_ALIASES = {
    "assets": "Assets",
    "reserve accounts": "Reserve Accounts",
    "due to/from": "Due To/From",
    "other assets": "Other Assets",
    "liabilities": "Liabilities",
    "capital": "Capital",
    "reserve allocation": "Reserve Allocation",
    "equity": "Equity",
    "current yr increase/decrease": "Current Yr Increase/Decrease",
}

_IGNORE_HEADINGS = {
    "liabilities & capital",
}



def _page_lines(raw_payload: ExtractionPayload) -> list[list[str]]:
    pages: list[list[str]] = []
    for segment in raw_payload.segments:
        if segment.type != "page":
            continue
        lines = [line.strip() for line in segment.text.splitlines() if line.strip()]
        if lines:
            pages.append(lines)
    return pages



def _detect_report_type(text: str) -> str | None:
    lowered = text.lower()
    for report_type, patterns in _REPORT_TYPE_PATTERNS.items():
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            return report_type
    return None



def _organization_name(lines: list[str]) -> str | None:
    for line in lines:
        if "properties:" in line.lower():
            candidate = line.split(":", 1)[1].strip()
            if " - " in candidate:
                candidate = candidate.split(" - ", 1)[0].strip()
            return candidate or None
    return None



def _statement_date(text: str) -> str | None:
    match = re.search(r"as of:\s*(\d{1,2}/\d{1,2}/\d{4})", text, flags=re.IGNORECASE)
    if match:
        return find_first_date(match.group(1))
    return find_first_date(text)



def _accounting_basis(text: str) -> str | None:
    match = re.search(r"accounting basis:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return normalize_whitespace(match.group(1)) if match else None



def _normalize_heading_key(text: str) -> str:
    return normalize_whitespace(text).lower().strip(":")



def _canonical_section_name(text: str) -> str | None:
    key = _normalize_heading_key(text)
    if key in _IGNORE_HEADINGS:
        return ""
    return _SECTION_ALIASES.get(key)



def _extract_page_columns(lines: list[str]) -> tuple[list[str], list[str], list[float]]:
    account_numbers: list[str] = []
    account_name_lines: list[str] = []
    balances: list[float] = []
    mode: str | None = None

    for line in lines:
        lowered = line.lower()
        if lowered == "account":
            continue
        if lowered == "number":
            mode = "accounts"
            continue
        if lowered == "account name":
            mode = "names"
            continue
        if lowered == "balance":
            mode = "balances"
            continue
        if lowered.startswith("created on") or lowered.startswith("page "):
            continue

        if mode == "accounts":
            if _ACCOUNT_RE.match(line):
                account_numbers.append(line)
            continue

        if mode == "names":
            normalized = normalize_whitespace(line)
            if normalized:
                account_name_lines.append(normalized)
            continue

        if mode == "balances" and _BALANCE_RE.match(line):
            amount = parse_amount(line)
            if amount is not None:
                balances.append(amount)

    return account_numbers, account_name_lines, balances



def _default_section_for_account(account_number: str) -> str | None:
    if account_number.startswith("1"):
        return "Assets"
    if account_number.startswith("2"):
        return "Liabilities"
    if account_number.startswith("3"):
        return "Capital"
    return None



def _build_name_entries(account_name_lines: list[str], account_count: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current_section: str | None = None

    for line in account_name_lines:
        canonical_section = _canonical_section_name(line)
        if canonical_section is not None:
            if canonical_section:
                current_section = canonical_section
            continue

        entries.append(
            {
                "account_name": line,
                "section": current_section,
                "is_total": line.lower().startswith("total ") or line.lower().startswith("total_") or line.lower().startswith("total"),
            }
        )
        if len(entries) >= account_count:
            break

    return entries[:account_count]



def _extract_line_items(page_lines: list[list[str]]) -> list[dict[str, object]]:
    line_items: list[dict[str, object]] = []

    for page_index, lines in enumerate(page_lines, start=1):
        account_numbers, account_name_lines, balances = _extract_page_columns(lines)
        if not account_numbers or not balances:
            continue

        pair_count = min(len(account_numbers), len(balances))
        if pair_count == 0:
            continue

        name_entries = _build_name_entries(account_name_lines, pair_count)
        if len(name_entries) < pair_count:
            missing_count = pair_count - len(name_entries)
            name_entries.extend(
                {
                    "account_name": f"Unknown Account {idx + 1}",
                    "section": None,
                    "is_total": False,
                }
                for idx in range(missing_count)
            )

        for idx in range(pair_count):
            account_number = account_numbers[idx]
            entry = name_entries[idx]
            account_name = str(entry.get("account_name") or "").strip()
            section = entry.get("section") or _default_section_for_account(account_number)
            line_items.append(
                {
                    "page_number": page_index,
                    "account_number": account_number,
                    "account_name": account_name,
                    "balance": balances[idx],
                    "section": section,
                    "is_total": bool(entry.get("is_total")),
                }
            )

    return line_items



def _summarize_sections(line_items: list[dict[str, object]]) -> list[dict[str, object]]:
    sections: OrderedDict[str, dict[str, object]] = OrderedDict()
    for item in line_items:
        section_name = item.get("section")
        if not section_name:
            continue
        section = sections.setdefault(
            str(section_name),
            {
                "name": str(section_name),
                "line_item_count": 0,
                "total_line_item_count": 0,
            },
        )
        section["line_item_count"] += 1
        if item.get("is_total"):
            section["total_line_item_count"] += 1
    return list(sections.values())



def build_financial_statement_document(raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument:
    del mask_pii
    text = raw_payload.raw_text
    lines = get_text_lines(raw_payload)
    pages = _page_lines(raw_payload)
    report_type = _detect_report_type(text)
    organization_name = _organization_name(lines)
    statement_date = _statement_date(text)
    accounting_basis = _accounting_basis(text)
    line_items = _extract_line_items(pages)
    sections = _summarize_sections(line_items)

    validation_errors: list[str] = []
    if report_type == "balance_sheet" and not line_items:
        validation_errors.append("no_financial_line_items_detected")

    fields = {
        "report_type": report_type,
        "organization_name": organization_name,
        "statement_date": statement_date,
        "accounting_basis": accounting_basis,
        "line_items": line_items,
        "sections": sections,
        "evidence": {
            "source_pages": first_source_pages(raw_payload, limit=3),
            "report_type": snippet_around_match(text, [r"\bbalance sheet\b", r"\bincome statement\b", r"\bprofit and loss\b"]),
            "statement_date": snippet_around_match(text, [r"as of:\s*\d{1,2}/\d{1,2}/\d{4}"]),
            "accounting_basis": snippet_around_match(text, [r"accounting basis:\s*[^\n]+"]),
            "organization_name": organization_name,
        },
    }

    review = build_review_metadata(
        required_fields={
            "report_type": report_type,
            "organization_name": organization_name,
            "statement_date": statement_date,
            "line_items": line_items,
        },
        validation_errors=validation_errors,
        raw_extra=raw_payload.extra,
        document_type="financial_statement",
    )
    return StructuredDocument(document_type="financial_statement", fields=fields, review=review)
