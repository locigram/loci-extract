from __future__ import annotations

import re

from app.normalization import find_first_date, parse_amount
from app.review import build_review_metadata
from app.schemas import ExtractionPayload, StructuredDocument
from app.structured.common import first_source_pages, get_form_lines, get_form_text, snippet_around_match

_PAYMENT_KEYWORDS = ("visa", "mastercard", "amex", "american express", "cash", "debit")


def _find_amount(label: str, text: str) -> float | None:
    match = re.search(rf"\b{label}\b\s*[:#-]?\s*([\$\d,().-]+)", text, flags=re.IGNORECASE)
    return parse_amount(match.group(1)) if match else None


def _merchant_name(lines: list[str]) -> str | None:
    for line in lines:
        lower = line.lower()
        if any(token in lower for token in ("total", "subtotal", "tax", "tip", "visa", "amex", "mastercard", "receipt")):
            continue
        return line
    return lines[0] if lines else None


_RECEIPT_FORM_SIGNALS = [
    "total",
    "subtotal",
    "tax",
    "receipt",
    "invoice",
    "amount due",
    "balance due",
]


def build_receipt_document(raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument:
    text = get_form_text(raw_payload, form_signals=_RECEIPT_FORM_SIGNALS)
    lines = get_form_lines(raw_payload, form_signals=_RECEIPT_FORM_SIGNALS)
    subtotal = _find_amount("subtotal", text)
    tax = _find_amount("tax", text)
    tip = _find_amount("tip", text)
    total = _find_amount("total", text)
    payment_method_hint = next((keyword for keyword in _PAYMENT_KEYWORDS if keyword in text.lower()), None)

    validation_errors: list[str] = []
    if subtotal is not None and total is not None and subtotal > total:
        validation_errors.append("subtotal_exceeds_total")
    if tax is not None and total is not None and subtotal is not None and subtotal + tax + (tip or 0.0) > total + 0.01:
        validation_errors.append("component_amounts_exceed_total")

    fields = {
        "merchant": {
            "name": _merchant_name(lines),
            "address": None,
            "phone": None,
        },
        "transaction": {
            "date": find_first_date(text),
            "time": None,
            "currency": "USD",
            "payment_method_hint": payment_method_hint,
            "receipt_number": None,
        },
        "amounts": {
            "subtotal": subtotal,
            "tax": tax,
            "tip": tip,
            "fees": None,
            "discount": None,
            "total": total,
        },
        "line_items": [],
        "evidence": {
            "source_pages": first_source_pages(raw_payload),
            "merchant_name": lines[0] if lines else None,
            "date": snippet_around_match(text, [r"\b\d{1,2}/\d{1,2}/\d{4}\b", r"\b\d{4}-\d{2}-\d{2}\b"]),
            "subtotal": snippet_around_match(text, [r"\bsubtotal\b\s*[:#-]?\s*([\$\d,().-]+)"]),
            "tax": snippet_around_match(text, [r"\btax\b\s*[:#-]?\s*([\$\d,().-]+)"]),
            "total": snippet_around_match(text, [r"\btotal\b\s*[:#-]?\s*([\$\d,().-]+)"]),
        },
    }
    review = build_review_metadata(
        required_fields={
            "merchant.name": fields["merchant"]["name"],
            "transaction.date": fields["transaction"]["date"],
            "amounts.total": fields["amounts"]["total"],
        },
        validation_errors=validation_errors,
        raw_extra=raw_payload.extra,
        document_type="receipt",
    )
    return StructuredDocument(document_type="receipt", fields=fields, review=review)
