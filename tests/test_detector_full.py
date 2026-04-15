"""Tax + financial detection tests per TAX_DETECTION_SPEC.md unit-test cases.

Adapted from the spec's §"Unit tests for detection" — note our schema uses
the "K-1 1065" / "K-1 1120-S" / "K-1 1041" string format, not "K1-1065"."""

from __future__ import annotations

from loci_extract.detector import (
    _detect_tax_year,
    detect_financial_document_type,
    detect_tax_document_type,
)

W2_TEXT = """
Form W-2 Wage and Tax Statement 2025
Employee's social security number: 622-76-8654
Employer ID number (EIN): 87-4661053
Wages, tips, other compensation: 7680.00
Federal income tax withheld: 0.00
Social security wages: 7680.00
Medicare wages and tips: 7680.00
Statutory employee  Retirement plan  Third-party sick pay
"""

ADP_SUMMARY_TEXT = """
2025 W-2 and EARNINGS SUMMARY  ADP
COMPANY QPL
1  Total Employees
2  Total Forms Count
Balancing Form W-2/W-3 Totals to the Wage and Tax Register
For: BATCH NO. 2025/4/99686
"""

NEC_TEXT = """
Form 1099-NEC  2025
Nonemployee compensation
Box 1: 15000.00
Payer's TIN: 98-7654321
"""

MISC_TEXT = """
Form 1099-MISC  2025
Rents  Box 1: 24000.00
Fishing boat proceeds
Payer's TIN: 11-2233445
"""

INT_TEXT = """
Form 1099-INT  2025
Interest Income
Early withdrawal penalty
Tax-exempt interest
"""

K1_1065_TEXT = """
Schedule K-1 (Form 1065)  2025
Partner's Share of Income, Deductions, Credits, etc.
Guaranteed payments to partner
Partner's capital account analysis
"""

K1_1120S_TEXT = """
Schedule K-1 (Form 1120-S)  2025
Shareholder's Share of Income, Deductions, Credits, etc.
S corporation
Shareholder's pro rata share items
"""


def test_w2_detected():
    r = detect_tax_document_type(W2_TEXT)
    assert r.document_type == "W2"
    assert r.confidence >= 0.7
    assert r.tax_year == 2025


def test_w2_record_count_one_employee_four_copies():
    # One SSN appearing 4 times = 1 employee
    r = detect_tax_document_type(W2_TEXT * 4)
    assert r.estimated_record_count == 1


def test_adp_summary_detected():
    r = detect_tax_document_type(ADP_SUMMARY_TEXT)
    assert r.document_type == "W2"
    assert r.is_summary_sheet is True
    assert r.issuer_software == "ADP"


def test_1099_nec_vs_misc():
    nec = detect_tax_document_type(NEC_TEXT)
    assert nec.document_type == "1099-NEC"
    misc = detect_tax_document_type(MISC_TEXT)
    assert misc.document_type == "1099-MISC"


def test_1099_nec_does_not_match_misc():
    # "Nonemployee compensation" should suppress MISC match
    r = detect_tax_document_type(NEC_TEXT)
    assert r.document_type != "1099-MISC"


def test_1099_int():
    r = detect_tax_document_type(INT_TEXT)
    assert r.document_type == "1099-INT"


def test_k1_1065_vs_1120s():
    r1065 = detect_tax_document_type(K1_1065_TEXT)
    assert r1065.document_type == "K-1 1065"
    r1120s = detect_tax_document_type(K1_1120S_TEXT)
    assert r1120s.document_type == "K-1 1120-S"


def test_unknown_returns_unknown():
    r = detect_tax_document_type("This is an invoice for services rendered.")
    assert r.document_type == "UNKNOWN"
    assert r.confidence < 0.5


def test_tax_year_extraction():
    assert _detect_tax_year("Wage and Tax Statement 2025") == 2025
    assert _detect_tax_year("Form W-2 2024 Dept. of Treasury") == 2024
    assert _detect_tax_year("No year here") is None


def test_financial_doc_not_detected_as_tax():
    text = """
    Balance Sheet - PMG
    As of: 01/31/2025
    ASSETS
    Total Cash  487,081.05
    TOTAL ASSETS  1,040,135.12
    """
    r = detect_tax_document_type(text)
    assert r.document_type == "UNKNOWN" or r.confidence < 0.5


def test_financial_balance_sheet():
    text = "Balance Sheet As of 01/31/2025\nTotal Assets 1,040,135.12\nCurrent Assets\nAccounts Receivable"
    assert detect_financial_document_type(text) == "BALANCE_SHEET"


def test_financial_income_statement_comparison_beats_simple():
    text = """
    Income Statement - 12 Month Comparison
    YTD Actual  YTD Budget
    Current month actual  Prior year actual
    $ Variance
    Net income
    """
    assert detect_financial_document_type(text) == "INCOME_STATEMENT_COMPARISON"


def test_financial_unknown():
    assert detect_financial_document_type("Just a memo about the weather.") == "FINANCIAL_UNKNOWN"
