from __future__ import annotations

import re
from collections import Counter

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

_SECTION_TITLES = {
    "assets",
    "liabilities",
    "capital",
    "equity",
    "reserve accounts",
    "due to/from",
    "other assets",
    "reserve allocation",
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


def _is_section_heading(normalized: str) -> bool:
    normalized_lower = normalized.lower()
    if normalized_lower in _SECTION_TITLES:
        return True
    if normalized.isupper() and len(normalized.split()) <= 4 and '-' not in normalized and not any(ch.isdigit() for ch in normalized):
        return True
    return False


def _extract_line_items(page_lines: list[list[str]]) -> list[dict[str, object]]:
    account_numbers: list[str] = []
    account_names: list[str] = []
    account_sections: list[str | None] = []
    balances: list[float] = []
    current_section: str | None = None
    line_items: list[dict[str, object]] = []

    for page_index, lines in enumerate(page_lines, start=1):
        in_accounts = False
        in_names = False
        in_balances = False

        for line in lines:
            lowered = line.lower()
            if lowered == "account":
                continue
            if lowered == "number":
                in_accounts, in_names, in_balances = True, False, False
                continue
            if lowered == "account name":
                in_accounts, in_names, in_balances = False, True, False
                continue
            if lowered == "balance":
                in_accounts, in_names, in_balances = False, False, True
                continue
            if lowered.startswith("created on") or lowered.startswith("page "):
                continue

            if in_accounts and _ACCOUNT_RE.match(line):
                account_numbers.append(line)
                continue

            if in_names:
                normalized = normalize_whitespace(line)
                if not normalized:
                    continue
                if _is_section_heading(normalized):
                    current_section = normalized.title()
                    continue
                account_names.append(normalized)
                account_sections.append(current_section)
                continue

            if in_balances and _BALANCE_RE.match(line):
                amount = parse_amount(line)
                if amount is not None:
                    balances.append(amount)
                continue

        pair_count = min(len(account_numbers), len(account_names), len(balances))
        while len(line_items) < pair_count:
            idx = len(line_items)
            line_items.append(
                {
                    "page_number": page_index,
                    "account_number": account_numbers[idx],
                    "account_name": account_names[idx],
                    "balance": balances[idx],
                    "section": account_sections[idx] if idx < len(account_sections) else None,
                }
            )

    return line_items


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

    section_counter = Counter(item.get("section") for item in line_items if item.get("section"))
    sections = [
        {"name": section_name, "line_item_count": count}
        for section_name, count in section_counter.items()
    ]

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
