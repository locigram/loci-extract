from __future__ import annotations

from app.schemas import ReviewMetadata, StructuredDocType


TAX_FORM_REVIEW_DOC_TYPES = {"w2", "1099-nec", "tax_return_package"}
OCR_SENSITIVE_DOC_TYPES = TAX_FORM_REVIEW_DOC_TYPES | {"receipt"}


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
    ocr_average_score = raw_extra.get("ocr_average_score")
    if not isinstance(ocr_average_score, (int, float)):
        ocr_average_score = raw_extra.get("ocr_score")

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

    if document_type in OCR_SENSITIVE_DOC_TYPES:
        ocr_backed = raw_extra.get("result_source") in {"ocr", "parser_fallback"} or any(
            entry.get("source") in {"ocr", "parser_fallback"} for entry in page_provenance if isinstance(entry, dict)
        )
        if ocr_backed:
            review_reasons.append("ocr_backed_tax_document" if document_type in TAX_FORM_REVIEW_DOC_TYPES else "ocr_backed_receipt")

        if isinstance(ocr_average_score, (int, float)) and ocr_average_score < 10:
            review_reasons.append(
                "low_ocr_quality_tax_document" if document_type in TAX_FORM_REVIEW_DOC_TYPES else "low_ocr_quality_receipt"
            )
        if any(
            isinstance(entry, dict)
            and entry.get("source") in {"ocr", "parser_fallback"}
            and (
                float(entry.get("ocr_score", 0.0) or 0.0) < 10
                or int(entry.get("text_length", 0) or 0) < 20
            )
            for entry in page_provenance
        ):
            review_reasons.append("weak_ocr_evidence")

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
