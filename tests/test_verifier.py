"""verifier.py — Python-side totals verification + derived field computation."""

from __future__ import annotations

from loci_extract.verifier import (
    ROUNDING_TOLERANCE,
    VerificationResult,
    _get_sections,
    compute_derived_fields,
    verify_section_totals,
)


def test_section_with_matching_total_verifies():
    extracted = {
        "expenses": {
            "sections": [{
                "section_name": "Utilities",
                "accounts": [
                    {"account_name": "Electricity", "amount": 1000.00},
                    {"account_name": "Water",       "amount": 2000.00},
                    {"account_name": "Trash",       "amount":  500.00},
                ],
                "section_total": 3500.00,
            }],
        },
    }
    result = verify_section_totals(extracted)
    assert result.verified is True
    assert result.mismatches == []


def test_small_rounding_difference_within_tolerance():
    extracted = {
        "expenses": {
            "sections": [{
                "section_name": "Utilities",
                "accounts": [
                    {"account_name": "A", "amount": 100.00},
                    {"account_name": "B", "amount": 200.01},
                ],
                "section_total": 300.00,  # off by $0.01 — within $0.02 tolerance
            }],
        },
    }
    assert verify_section_totals(extracted).verified is True


def test_large_mismatch_fails_verification():
    extracted = {
        "expenses": {
            "sections": [{
                "section_name": "Utilities",
                "accounts": [
                    {"account_name": "A", "amount": 100.00},
                    {"account_name": "B", "amount": 200.00},
                ],
                "section_total": 500.00,  # off by $200 → mismatch
            }],
        },
    }
    result = verify_section_totals(extracted)
    assert result.verified is False
    assert len(result.mismatches) == 1
    m = result.mismatches[0]
    assert m["section"] == "Utilities"
    assert abs(m["delta"] - 200.0) < 0.01


def test_balance_sheet_equation_balanced():
    extracted = {
        "assets":      {"total_assets":      10000.00, "sections": []},
        "liabilities": {"total_liabilities":  4000.00, "sections": []},
        "equity":      {"total_equity":       6000.00, "sections": []},
    }
    result = verify_section_totals(extracted)
    assert result.balance_sheet_balanced is True
    assert result.verified is True


def test_balance_sheet_equation_unbalanced():
    extracted = {
        "assets":      {"total_assets":      10000.00, "sections": []},
        "liabilities": {"total_liabilities":  4000.00, "sections": []},
        "equity":      {"total_equity":       5950.00, "sections": []},  # off by $50
    }
    result = verify_section_totals(extracted)
    assert result.balance_sheet_balanced is False
    assert result.verified is False
    assert any(m["section"] == "balance_sheet_equation" for m in result.mismatches)
    assert any("Balance sheet does not balance" in n for n in result.notes)


def test_get_sections_recurses_into_subsections():
    extracted = {
        "assets": {
            "sections": [{
                "section_name": "Current Assets",
                "subsections": [{
                    "section_name": "Cash",
                    "accounts": [{"account_name": "Checking", "balance": 1000}],
                    "section_total": 1000,
                }],
            }],
        },
    }
    found = _get_sections(extracted)
    # Sub-section with both accounts and section_total is detected
    assert any(s.get("section_name") == "Cash" for s in found)


def test_compute_derived_balance_sheet_retained_earnings():
    extracted = {
        "equity": {
            "total_equity_reported": 1000000.00,
            "sections": [{
                "section_name": "Capital",
                "accounts": [
                    {"account_name": "Beginning Equity",  "balance": 800000.00},
                    {"account_name": "Owner Contribution","balance": 100000.00},
                ],
                "section_total": 900000.00,
            }],
        },
    }
    derived = compute_derived_fields(extracted, "BALANCE_SHEET")
    assert "retained_earnings_calculated" in derived
    # 1,000,000 reported total - 900,000 explicit = 100,000 implied retained
    assert abs(derived["retained_earnings_calculated"] - 100000.0) < 0.01


def test_compute_derived_income_statement_net_income():
    extracted = {
        "income":   {"total": 50000.00, "sections": []},
        "expenses": {"total": 35000.00, "sections": []},
    }
    derived = compute_derived_fields(extracted, "INCOME_STATEMENT")
    assert "net_income_calculated" in derived
    assert abs(derived["net_income_calculated"] - 15000.0) < 0.01


def test_compute_derived_reserve_allocation():
    extracted = {
        "components": [
            {"component_name": "Roof",     "current_balance": 50000},
            {"component_name": "Painting", "current_balance": -10000},
        ],
        "bank_accounts": [
            {"account_name": "Reserve Bank A", "balance": 30000},
            {"account_name": "Reserve Bank B", "balance": 15000},
        ],
    }
    derived = compute_derived_fields(extracted, "RESERVE_ALLOCATION")
    assert derived["total_reserve_balance_calculated"] == 40000.0
    assert derived["total_bank_balance_calculated"] == 45000.0


def test_verification_result_dataclass():
    r = VerificationResult(verified=True)
    assert r.mismatches == []
    assert r.notes == []
    assert r.balance_sheet_balanced is None


def test_rounding_tolerance_constant():
    assert float(ROUNDING_TOLERANCE) == 0.02
