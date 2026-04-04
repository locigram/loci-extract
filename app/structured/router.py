from __future__ import annotations

from app.schemas import ClassificationResult, ExtractionPayload, StructuredDocument
from app.structured.common import unknown_structured_document
from app.structured.form_1099_nec import build_1099_nec_document
from app.structured.receipt import build_receipt_document
from app.structured.tax_return_package import build_tax_return_package_document
from app.structured.w2 import build_w2_document


def build_structured_document(
    classification: ClassificationResult,
    raw_payload: ExtractionPayload,
    *,
    mask_pii: bool = True,
) -> StructuredDocument:
    if classification.doc_type == "w2":
        return build_w2_document(raw_payload, mask_pii=mask_pii)
    if classification.doc_type == "1099-nec":
        return build_1099_nec_document(raw_payload, mask_pii=mask_pii)
    if classification.doc_type == "receipt":
        return build_receipt_document(raw_payload, mask_pii=mask_pii)
    if classification.doc_type == "tax_return_package":
        return build_tax_return_package_document(raw_payload, mask_pii=mask_pii)
    return unknown_structured_document()
