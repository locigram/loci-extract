from __future__ import annotations

import re

from app.normalization import parse_amount
from app.review import build_review_metadata
from app.schemas import ExtractionPayload, StructuredDocument
from app.structured.common import first_source_pages, get_form_text, snippet_around_match


def _find_amount(label: str, text: str) -> float | None:
    match = re.search(rf"{label}\s*[:#-]?\s*([\$\d,().-]+)", text, flags=re.IGNORECASE)
    return parse_amount(match.group(1)) if match else None


_1040_FORM_SIGNALS = [
    "form 1040",
    "u.s. individual income tax return",
    "filing status",
    "adjusted gross income",
    "taxable income",
    "total tax",
    "total payments",
    "amount you owe",
    "overpaid",
]


def build_tax_return_package_document(raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument:
    # Use form data pages only (1040 can span 2 pages)
    text = get_form_text(raw_payload, form_signals=_1040_FORM_SIGNALS)
    year_match = re.search(r"form\s+1040.*?(20\d{2})", text, flags=re.IGNORECASE | re.DOTALL)
    tax_year = int(year_match.group(1)) if year_match else None
    taxpayer_name_patterns = [
        r"taxpayer(?:\s+name)?\s*[:#-]?\s*([^\n]+)",
        r"your first name and middle initial\s*[:#-]?\s*([^\n]+)",
    ]
    taxpayer_name_match = re.search(taxpayer_name_patterns[0], text, flags=re.IGNORECASE)
    if not taxpayer_name_match:
        taxpayer_name_match = re.search(taxpayer_name_patterns[1], text, flags=re.IGNORECASE)
    primary_name = taxpayer_name_match.group(1).strip() if taxpayer_name_match else None

    filing_status = None
    for status in (
        "single",
        "married filing jointly",
        "married filing separately",
        "head of household",
        "qualifying surviving spouse",
    ):
        if status in text.lower():
            filing_status = status
            break

    attached_forms: list[str] = []
    for form_name in ("schedule 1", "schedule c", "schedule e"):
        if form_name in text.lower():
            attached_forms.append(form_name.replace(" ", "-"))

    page_provenance = raw_payload.extra.get("page_provenance") if isinstance(raw_payload.extra.get("page_provenance"), list) else []
    pages = [
        {"page_number": entry.get("page_number"), "form_type_hint": "1040", "section_hint": None}
        for entry in page_provenance
        if isinstance(entry, dict) and entry.get("page_number") is not None
    ]

    fields = {
        "tax_year": tax_year,
        "primary_form": "1040" if "form 1040" in text.lower() else None,
        "taxpayer": {
            "primary_name": primary_name,
            "secondary_name": None,
            "ssn_last4": None,
            "address": None,
        },
        "filing": {
            "filing_status": filing_status,
            "dependents_count": None,
            "has_spouse": filing_status in {"married filing jointly", "married filing separately", "qualifying surviving spouse"},
        },
        "summary": {
            "total_income": _find_amount("total income", text),
            "adjusted_gross_income": _find_amount("adjusted gross income", text),
            "taxable_income": _find_amount("taxable income", text),
            "total_tax": _find_amount("total tax", text),
            "total_payments": _find_amount("total payments", text),
            "refund": _find_amount("refund", text),
            "amount_owed": _find_amount("amount you owe", text),
        },
        "attached_forms": attached_forms,
        "pages": pages,
        "evidence": {
            "source_pages": first_source_pages(raw_payload),
            "taxpayer_name": snippet_around_match(text, taxpayer_name_patterns),
            "total_income": snippet_around_match(text, [r"total income\s*[:#-]?\s*([\$\d,().-]+)"]),
            "adjusted_gross_income": snippet_around_match(text, [r"adjusted gross income\s*[:#-]?\s*([\$\d,().-]+)"]),
            "taxable_income": snippet_around_match(text, [r"taxable income\s*[:#-]?\s*([\$\d,().-]+)"]),
        },
    }

    review = build_review_metadata(
        required_fields={
            "tax_year": fields["tax_year"],
            "primary_form": fields["primary_form"],
            "taxpayer.primary_name": fields["taxpayer"]["primary_name"],
        },
        validation_errors=[],
        raw_extra=raw_payload.extra,
        document_type="tax_return_package",
    )
    return StructuredDocument(document_type="tax_return_package", fields=fields, review=review)
