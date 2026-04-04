from app.normalization import extract_last4, find_first_date, mask_identifier, parse_amount
from app.review import build_review_metadata
from app.schemas import ClassificationResult, DocumentMetadata, ExtractionMethod, ExtractionPayload
from app.structured.form_1099_nec import build_1099_nec_document
from app.structured.receipt import build_receipt_document
from app.structured.router import build_structured_document
from app.structured.tax_return_package import build_tax_return_package_document
from app.structured.w2 import build_w2_document


def _payload(raw_text: str, *, source_type: str = "text", extra: dict | None = None) -> ExtractionPayload:
    return ExtractionPayload(
        document_id="doc-1",
        metadata=DocumentMetadata(filename="sample.txt", mime_type="text/plain", source_type=source_type),
        extraction=ExtractionMethod(extractor="plaintext"),
        raw_text=raw_text,
        segments=[],
        chunks=[],
        extra=extra or {},
    )


def test_parse_amount_with_commas_and_dollar_sign() -> None:
    assert parse_amount("$12,345.67") == 12345.67


def test_extract_last4() -> None:
    assert extract_last4("XXX-XX-1234") == "1234"


def test_mask_identifier() -> None:
    assert mask_identifier("123-45-6789") == "***-**-6789"


def test_find_first_date() -> None:
    assert find_first_date("Date: 03/14/2024") == "2024-03-14"


def test_missing_required_fields_require_review() -> None:
    review = build_review_metadata(
        required_fields={"employee.full_name": None},
        validation_errors=[],
        raw_extra={},
        document_type="w2",
    )
    assert review.requires_human_review is True
    assert "employee.full_name" in review.missing_fields


def test_review_marks_ocr_backed_tax_document() -> None:
    review = build_review_metadata(
        required_fields={"employee.full_name": "Jane Doe"},
        validation_errors=[],
        raw_extra={"page_provenance": [{"page_number": 1, "source": "ocr", "ocr_score": 12.0, "text_length": 8}]},
        document_type="w2",
    )
    assert review.requires_human_review is True
    assert "ocr_backed_tax_document" in review.review_reasons
    assert "weak_ocr_evidence" in review.review_reasons


def test_router_returns_unknown_fallback() -> None:
    structured = build_structured_document(ClassificationResult(doc_type="unknown"), _payload("hello world"))
    assert structured.document_type == "unknown"
    assert structured.review.requires_human_review is True


def test_build_w2_document() -> None:
    raw_text = """
Form W-2 Wage and Tax Statement 2024
Employee's social security number XXX-XX-1234
Employee name John Q Public
Employer identification number 12-3456789
Employer name Example Payroll Inc
1 Wages, tips, other compensation 85000.00
2 Federal income tax withheld 12000.00
3 Social security wages 85000.00
4 Social security tax withheld 5270.00
5 Medicare wages and tips 85000.00
6 Medicare tax withheld 1232.50
"""
    structured = build_w2_document(_payload(raw_text, extra={"page_provenance": [{"page_number": 1, "source": "parser", "has_text": True, "text_length": 120}]}))
    assert structured.document_type == "w2"
    assert structured.fields["tax_year"] == 2024
    assert structured.fields["employee"]["full_name"] == "John Q Public"
    assert structured.fields["employee"]["ssn_last4"] == "1234"
    assert structured.fields["employer"]["name"] == "Example Payroll Inc"
    assert structured.fields["boxes"]["1_wages_tips_other_comp"] == 85000.00
    assert structured.fields["evidence"]["source_pages"] == [1]
    assert 'Employee name John Q Public' in structured.fields["evidence"]["employee_name"]
    assert structured.review.requires_human_review is False


def test_build_1099_nec_document() -> None:
    raw_text = """
Form 1099-NEC 2024
Recipient's name Jane Contractor
Recipient's TIN XXX-XX-4321
Payer's name ACME Services LLC
Payer's TIN 98-7654321
1 Nonemployee compensation 25000.00
4 Federal income tax withheld 0.00
"""
    structured = build_1099_nec_document(_payload(raw_text, extra={"page_provenance": [{"page_number": 1, "source": "parser", "has_text": True, "text_length": 120}]}))
    assert structured.document_type == "1099-nec"
    assert structured.fields["recipient"]["name"] == "Jane Contractor"
    assert structured.fields["recipient"]["tin_last4"] == "4321"
    assert structured.fields["payer"]["name"] == "ACME Services LLC"
    assert structured.fields["boxes"]["1_nonemployee_compensation"] == 25000.00
    assert 'Nonemployee compensation' in structured.fields["evidence"]["box_1"]
    assert structured.review.requires_human_review is False


def test_build_receipt_document_requires_review_without_total() -> None:
    raw_text = """Coffee Shop
03/14/2024
Subtotal 10.00
Tax 0.80"""
    structured = build_receipt_document(_payload(raw_text, source_type="image", extra={"result_source": "ocr", "ocr_score": 10.0}))
    assert structured.document_type == "receipt"
    assert structured.fields["merchant"]["name"] == "Coffee Shop"
    assert structured.fields["evidence"]["date"] is not None
    assert structured.review.requires_human_review is True
    assert "amounts.total" in structured.review.missing_fields



def test_build_receipt_document_adds_receipt_specific_ocr_review_reasons() -> None:
    raw_text = """Coffee Shop
03/14/2024
Subtotal 10.00
Tax 0.80
Total 10.80"""
    structured = build_receipt_document(
        _payload(
            raw_text,
            source_type="image",
            extra={
                "result_source": "ocr",
                "ocr_score": 6.0,
                "page_provenance": [{"page_number": 1, "source": "ocr", "ocr_score": 6.0, "text_length": 12}],
            },
        )
    )
    assert structured.review.requires_human_review is True
    assert "ocr_backed_receipt" in structured.review.review_reasons
    assert "low_ocr_quality_receipt" in structured.review.review_reasons
    assert "weak_ocr_evidence" in structured.review.review_reasons


def test_build_tax_return_package_document() -> None:
    raw_text = """
Form 1040 U.S. Individual Income Tax Return 2024
Taxpayer name John and Jane Doe
Single
Total income 100000.00
Adjusted gross income 95000.00
Taxable income 80000.00
Total tax 12000.00
Refund 500.00
Schedule C
"""
    structured = build_tax_return_package_document(
        _payload(raw_text, extra={"page_provenance": [{"page_number": 1, "source": "parser", "has_text": True, "text_length": 120}]})
    )
    assert structured.document_type == "tax_return_package"
    assert structured.fields["primary_form"] == "1040"
    assert structured.fields["tax_year"] == 2024
    assert structured.fields["taxpayer"]["primary_name"] == "John and Jane Doe"
    assert structured.fields["summary"]["total_income"] == 100000.00
    assert "schedule-c" in structured.fields["attached_forms"]
    assert structured.fields["evidence"]["taxpayer_name"] is not None
    assert structured.review.requires_human_review is False
