from __future__ import annotations

from app.schemas import ReviewMetadata, StructuredDocType


TAX_REVIEW_DOC_TYPES = {"w2", "1099-nec", "tax_return_package"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def build_review_metadata(
    *,
    required_fields: dict[str, object],
    validation_errors: list[str],
    raw_extra: dict[str, object],
    document_type: StructuredDocType,
) -> ReviewMetadata:
    missing_fields = [name for name, value in required_fields.items() if _is_missing(value)]
    review_reasons: list[str] = []
    warning_codes = raw_extra.get("warning_codes") or []
    page_provenance = raw_extra.get("page_provenance") if isinstance(raw_extra.get("page_provenance"), list) else []

    if missing_fields:
        review_reasons.append("missing_required_fields")
    if validation_errors:
        review_reasons.append("validation_errors")
    if warning_codes:
        review_reasons.append("source_extraction_warnings")
    if raw_extra.get("result_source") == "none":
        review_reasons.append("no_text_recovered")
    if any(entry.get("source") == "none" for entry in page_provenance if isinstance(entry, dict)):
        review_reasons.append("pages_with_no_text")
    if document_type in TAX_REVIEW_DOC_TYPES:
        if raw_extra.get("result_source") in {"ocr", "parser_fallback"}:
            review_reasons.append("ocr_backed_tax_document")
        elif any(entry.get("source") in {"ocr", "parser_fallback"} for entry in page_provenance if isinstance(entry, dict)):
            review_reasons.append("ocr_backed_tax_document")

    deduped_reasons: list[str] = []
    for reason in review_reasons:
        if reason not in deduped_reasons:
            deduped_reasons.append(reason)

    return ReviewMetadata(
        requires_human_review=bool(missing_fields or validation_errors or deduped_reasons),
        review_reasons=deduped_reasons,
        missing_fields=missing_fields,
        validation_errors=validation_errors,
    )
