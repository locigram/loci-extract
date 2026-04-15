"""boundary_detector.py — multi-section PDF boundary detection."""

from __future__ import annotations

from loci_extract.boundary_detector import DocumentSection, detect_boundaries


def _page(num: int, text: str) -> dict:
    return {"page": num, "text": text}


def test_two_section_pdf_balance_then_income():
    pages = [
        _page(0, "Balance Sheet\nAs of: 01/31/2025\nTotal Cash 487,081.05"),
        _page(1, "Continuation of Balance Sheet\nLiabilities & Equity"),
        _page(2, "Income Statement\nFor the period ending Jan 2025"),
        _page(3, "Continuation of Income Statement"),
    ]
    sections = detect_boundaries(pages)
    assert len(sections) == 2
    assert sections[0].document_type == "BALANCE_SHEET"
    assert sections[0].start_page == 0
    assert sections[0].end_page == 1
    assert sections[1].document_type == "INCOME_STATEMENT"
    assert sections[1].start_page == 2
    assert sections[1].end_page == 3


def test_single_section_returns_one():
    pages = [
        _page(0, "Balance Sheet\nAs of 01/31/2025"),
        _page(1, "More balance sheet stuff"),
    ]
    sections = detect_boundaries(pages)
    assert len(sections) == 1
    assert sections[0].document_type == "BALANCE_SHEET"
    assert sections[0].start_page == 0
    assert sections[0].end_page == 1


def test_no_boundaries_returns_unknown():
    pages = [
        _page(0, "Random invoice text. Customer paid 500."),
        _page(1, "Continued. No financial signals here."),
    ]
    sections = detect_boundaries(pages)
    assert len(sections) == 1
    assert sections[0].document_type == "UNKNOWN"
    assert sections[0].start_page == 0
    assert sections[0].end_page == 1


def test_empty_pages_returns_unknown_section():
    sections = detect_boundaries([])
    assert len(sections) == 1
    assert sections[0].document_type == "UNKNOWN"


def test_w2_then_1099_detected():
    pages = [
        _page(0, "Form W-2 Wage and Tax Statement\n2025\nEmployee SSN ..."),
        _page(1, "Form 1099-NEC\n2025\nNonemployee compensation"),
    ]
    sections = detect_boundaries(pages)
    assert len(sections) == 2
    assert sections[0].document_type == "W2"
    assert sections[1].document_type == "1099-NEC"


def test_section_dataclass_shape():
    s = DocumentSection(0, 5, "BALANCE_SHEET", "Balance Sheet - Acme", 0.95)
    assert s.start_page == 0
    assert s.end_page == 5
    assert s.confidence == 0.95
