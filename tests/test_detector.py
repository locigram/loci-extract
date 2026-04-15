"""Detector keyword classification tests (no PDF I/O needed)."""

from __future__ import annotations

from loci_extract.detector import identify_doc_types


def test_identifies_w2():
    text = "Form W-2 Wage and Tax Statement 2025 ... Copy B For Employee's Records"
    assert "W2" in identify_doc_types(text)


def test_identifies_1099_nec():
    text = "Form 1099-NEC Nonemployee Compensation Box 1 $15,000.00"
    assert "1099-NEC" in identify_doc_types(text)


def test_multi_doc_pdf():
    text = (
        "Form W-2 Wage and Tax Statement ... "
        "Form 1099-NEC Nonemployee Compensation ..."
    )
    found = identify_doc_types(text)
    assert "W2" in found
    assert "1099-NEC" in found


def test_k1_variants_separate():
    text_1065 = "Schedule K-1 Form 1065 Partner's Share of Income"
    text_1120s = "Schedule K-1 Form 1120-S Shareholder's Share of Income"
    assert "K-1 1065" in identify_doc_types(text_1065)
    assert "K-1 1120-S" in identify_doc_types(text_1120s)


def test_empty_text():
    assert identify_doc_types("") == []
    assert identify_doc_types(None) == []


def test_non_tax_text():
    assert identify_doc_types("Hello world, this is just a random letter.") == []


def test_1098_t_not_confused_with_1098():
    # 1098-T has both "1098" and "1098-T" substring; 1098 must also match
    # for 1098 base form detection. Spec says they're separate types. Our
    # keyword list requires "mortgage interest statement" for 1098 base so
    # the 1098-T text shouldn't match 1098.
    text = "Form 1098-T Tuition Statement 2025 Box 1 $18,500"
    found = identify_doc_types(text)
    assert "1098-T" in found
    assert "1098" not in found
