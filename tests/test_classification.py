from app.classification.rules import classify_document


def test_classify_w2_text() -> None:
    result = classify_document(
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement 2024",
        doc_type_hint=None,
    )
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


def test_classify_1099_nec_text() -> None:
    result = classify_document(
        filename="1099.txt",
        mime_type="text/plain",
        raw_text="Form 1099-NEC Nonemployee Compensation 2024",
    )
    assert result.doc_type == "1099-nec"


def test_classify_receipt_text() -> None:
    result = classify_document(
        filename="receipt.jpg",
        mime_type="image/jpeg",
        raw_text="""Receipt
Subtotal 10.00
Tax 0.80
Total 10.80
Visa""",
    )
    assert result.doc_type == "receipt"


def test_classify_tax_return_text() -> None:
    result = classify_document(
        filename="1040.pdf",
        mime_type="application/pdf",
        raw_text="Form 1040 U.S. Individual Income Tax Return 2024",
    )
    assert result.doc_type == "tax_return_package"


def test_classify_unknown_text() -> None:
    result = classify_document(
        filename="note.txt",
        mime_type="text/plain",
        raw_text="hello world",
    )
    assert result.doc_type == "unknown"


def test_doc_type_hint_overrides_rules() -> None:
    result = classify_document(
        filename="whatever.txt",
        mime_type="text/plain",
        raw_text="hello world",
        doc_type_hint="receipt",
    )
    assert result.doc_type == "receipt"
    assert result.strategy == "hint"
