"""Validate per-doc-type pydantic models round-trip correctly."""

from __future__ import annotations

import pytest

from loci_extract.schema import (
    DATA_MODEL_BY_TYPE,
    Document,
    Extraction,
)


def _w2_payload():
    return {
        "employer": {"name": "Acme", "ein": "12-3456789", "address": "123 Main"},
        "employee": {"name": "Jane", "ssn_last4": "XXX-XX-1234", "address": "456 Elm"},
        "federal": {"box1_wages": 75000.0, "box2_federal_withheld": 12000.0, "box3_ss_wages": 75000.0,
                     "box4_ss_withheld": 4650.0, "box5_medicare_wages": 75000.0, "box6_medicare_withheld": 1087.5},
        "box12": [{"code": "AA", "amount": 3833.36, "description": "Roth 401(k)"}],
        "box13": {"retirement_plan": True},
        "box14_other": [{"label": "CA SDI", "amount": 92.16}],
        "state": [{"state_abbr": "CA", "state_id": "CA-123", "box16_state_wages": 75000.0, "box17_state_withheld": 3200.0}],
        "local": [],
    }


def test_extraction_with_w2_validates():
    extraction = Extraction(documents=[
        Document(document_type="W2", tax_year=2025, data=_w2_payload())
    ])
    validated = extraction.validate_all()
    assert len(validated) == 1


def test_every_registered_type_has_default_construction():
    # Every DATA_MODEL_BY_TYPE entry should allow minimal construction with
    # required fields. We just instantiate from a tiny payload per type to
    # make sure required fields aren't surprising.
    minimal_payloads = {
        "W2": _w2_payload(),
        "1099-NEC": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-MISC": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-INT": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-DIV": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-B": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
            "transactions": [],
        },
        "1099-R": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-G": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-SA": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-K": {
            "payer": {"name": "X", "tin": "12-3456789"},
            "recipient": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-S": {
            "filer": {"name": "X", "tin": "12-3456789"},
            "transferor": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-C": {
            "creditor": {"name": "X", "tin": "12-3456789"},
            "debtor": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1099-A": {
            "lender": {"name": "X", "tin": "12-3456789"},
            "borrower": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1098": {
            "recipient": {"name": "X", "tin": "12-3456789"},
            "payer": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1098-T": {
            "filer": {"name": "U", "tin": "12-3456789"},
            "student": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "1098-E": {
            "recipient": {"name": "R", "tin": "12-3456789"},
            "borrower": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "SSA-1099": {
            "payer": {"name": "SSA"},
            "beneficiary": {"name": "Y", "ssn_last4": "XXX-XX-1234"},
        },
        "RRB-1099": {
            "payer": {"name": "RRB"},
            "recipient": {"name": "Y", "ssn_last4": "XXX-XX-1234"},
        },
        "K-1 1065": {
            "partnership": {"name": "P"},
            "partner": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "K-1 1120-S": {
            "corporation": {"name": "C"},
            "shareholder": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        "K-1 1041": {
            "estate_or_trust": {"name": "E"},
            "beneficiary": {"name": "Y", "tin_last4": "XXX-XX-1234"},
        },
        # Financial types — entity is the only required field.
        "BALANCE_SHEET": {"entity": {"name": "Acme"}},
        "INCOME_STATEMENT": {"entity": {"name": "Acme"}},
        "INCOME_STATEMENT_COMPARISON": {"entity": {"name": "Acme"}, "columns": []},
        "BUDGET_VS_ACTUAL": {"entity": {"name": "Acme"}, "columns": []},
        "TRIAL_BALANCE": {"entity": {"name": "Acme"}},
        "ACCOUNTS_RECEIVABLE_AGING": {"entity": {"name": "Acme"}},
        "ACCOUNTS_PAYABLE_AGING": {"entity": {"name": "Acme"}, "report_type": "AP"},
        "GENERAL_LEDGER": {"entity": {"name": "Acme"}},
        "RESERVE_ALLOCATION": {"entity": {"name": "Acme"}},
    }
    missing = [t for t in DATA_MODEL_BY_TYPE if t not in minimal_payloads]
    assert missing == [], f"Add minimal payloads for: {missing}"
    for doc_type, payload in minimal_payloads.items():
        model = DATA_MODEL_BY_TYPE[doc_type]
        instance = model.model_validate(payload)
        assert instance is not None


def test_wrong_data_for_type_raises():
    from pydantic import ValidationError

    bad = Document(document_type="W2", tax_year=2025, data={"employer": {"name": "X"}})  # missing required employee
    with pytest.raises(ValidationError):
        bad.validated_data()
