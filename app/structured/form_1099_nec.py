from __future__ import annotations

import re

from app.normalization import extract_last4, mask_identifier, parse_amount
from app.review import build_review_metadata
from app.schemas import ExtractionPayload, StructuredDocument
from app.structured.common import first_source_pages, search_patterns, snippet_around_match


def _extract_value(text: str, patterns: list[str]) -> str | None:
    match = search_patterns(text, patterns)
    if not match:
        return None
    return match.group(1).strip()


def build_1099_nec_document(raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument:
    text = raw_payload.raw_text
    tax_year_match = re.search(r"1099-nec.*?(20\d{2})", text, flags=re.IGNORECASE | re.DOTALL)
    tax_year = int(tax_year_match.group(1)) if tax_year_match else None
    recipient_name_patterns = [r"recipient(?:'s)?\s+name\s*[:#-]?\s*([^\n]+)"]
    recipient_tin_patterns = [
        r"recipient(?:'s)?\s+tin\s*[:#-]?\s*([^\n]+)",
        r"recipient taxpayer identification number\s*[:#-]?\s*([^\n]+)",
    ]
    payer_name_patterns = [r"payer(?:'s)?\s+name\s*[:#-]?\s*([^\n]+)"]
    payer_tin_patterns = [
        r"payer(?:'s)?\s+tin\s*[:#-]?\s*([^\n]+)",
        r"payer identification number\s*[:#-]?\s*([^\n]+)",
    ]
    box1_patterns = [r"\b1\s+nonemployee compensation\s+([\$\d,().-]+)"]
    box4_patterns = [r"\b4\s+federal income tax withheld\s+([\$\d,().-]+)"]

    recipient_name = _extract_value(text, recipient_name_patterns)
    recipient_tin_source = _extract_value(text, recipient_tin_patterns)
    payer_name = _extract_value(text, payer_name_patterns)
    payer_tin = _extract_value(text, payer_tin_patterns)
    box1_text = _extract_value(text, box1_patterns)
    box4_text = _extract_value(text, box4_patterns)

    fields = {
        "tax_year": tax_year,
        "recipient": {
            "name": recipient_name,
            "tin_last4": extract_last4(recipient_tin_source),
            "tin_masked": mask_identifier(recipient_tin_source) if mask_pii else recipient_tin_source,
            "address": None,
        },
        "payer": {
            "name": payer_name,
            "tin": mask_identifier(payer_tin) if mask_pii else payer_tin,
            "address": None,
        },
        "boxes": {
            "1_nonemployee_compensation": parse_amount(box1_text),
            "4_federal_income_tax_withheld": parse_amount(box4_text),
            "5_state_tax_withheld": None,
            "6_state_payer_number": None,
            "7_state_income": None,
        },
        "state_entries": [],
        "evidence": {
            "source_pages": first_source_pages(raw_payload),
            "recipient_name": snippet_around_match(text, recipient_name_patterns),
            "recipient_tin": snippet_around_match(text, recipient_tin_patterns),
            "payer_name": snippet_around_match(text, payer_name_patterns),
            "box_1": snippet_around_match(text, box1_patterns),
            "box_4": snippet_around_match(text, box4_patterns),
        },
    }
    validation_errors: list[str] = []
    if box1_text and fields["boxes"]["1_nonemployee_compensation"] is None:
        validation_errors.append("unable_to_parse_box_1_nonemployee_compensation")
    review = build_review_metadata(
        required_fields={
            "tax_year": fields["tax_year"],
            "recipient.name": fields["recipient"]["name"],
            "payer.name": fields["payer"]["name"],
            "boxes.1_nonemployee_compensation": fields["boxes"]["1_nonemployee_compensation"],
        },
        validation_errors=validation_errors,
        raw_extra=raw_payload.extra,
        document_type="1099-nec",
    )
    return StructuredDocument(document_type="1099-nec", fields=fields, review=review)
