"""JSON / CSV / Lacerte / TXF formatter output shapes."""

from __future__ import annotations

import json

import pytest

from loci_extract.formatters import format_extraction
from loci_extract.schema import Document, Extraction


def _w2_doc(ssn="XXX-XX-1234", employer="Acme Corp", ein="12-3456789"):
    return Document(
        document_type="W2",
        tax_year=2025,
        data={
            "employer": {"name": employer, "ein": ein, "address": "123 Main"},
            "employee": {"name": "Jane Smith", "ssn_last4": ssn, "address": "456 Elm"},
            "federal": {
                "box1_wages": 75000.0, "box2_federal_withheld": 12000.0,
                "box3_ss_wages": 75000.0, "box4_ss_withheld": 4650.0,
                "box5_medicare_wages": 75000.0, "box6_medicare_withheld": 1087.5,
            },
            "box12": [{"code": "AA", "amount": 3833.36, "description": "Roth 401(k)"}],
            "box13": {"retirement_plan": True},
            "box14_other": [{"label": "CA SDI", "amount": 92.16}],
            "state": [{"state_abbr": "CA", "state_id": "CA-123", "box16_state_wages": 75000.0,
                        "box17_state_withheld": 3200.0}],
            "local": [],
        },
    )


def test_json_formatter_roundtrips():
    ext = Extraction(documents=[_w2_doc()])
    out = format_extraction(ext, "json")
    parsed = json.loads(out)
    assert parsed["documents"][0]["document_type"] == "W2"
    assert parsed["documents"][0]["data"]["federal"]["box1_wages"] == 75000.0


def test_csv_formatter_shape():
    ext = Extraction(documents=[_w2_doc()])
    out = format_extraction(ext, "csv")
    lines = out.strip().split("\n")
    assert lines[0].startswith("document_type,")
    # W-2 primary amount should be Box 1 wages label
    assert "Box 1 wages" in lines[1]
    assert "Jane Smith" in lines[1]


def test_lacerte_w2_output():
    ext = Extraction(documents=[_w2_doc()])
    out = format_extraction(ext, "lacerte")
    assert "\t" in out
    # Masked SSN prefix
    assert "XXXXX1234" in out
    # Box 1 wages in the output
    assert "75000.00" in out
    # Acme name
    assert "Acme Corp" in out


def test_lacerte_unsupported_type_raises():
    doc = Document(
        document_type="1098",
        tax_year=2025,
        data={
            "recipient": {"name": "Bank", "tin": "12-3456789"},
            "payer": {"name": "User", "tin_last4": "XXX-XX-1234"},
        },
    )
    with pytest.raises(NotImplementedError):
        format_extraction(Extraction(documents=[doc]), "lacerte")


def test_txf_w2_has_header_and_records():
    ext = Extraction(documents=[_w2_doc()])
    out = format_extraction(ext, "txf")
    assert out.startswith("V042\n")
    assert "Aloci-extract" in out
    # W-2 wages record code T0511
    assert "T0511" in out
    # Records terminate with ^
    assert "\n^\n" in out


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        format_extraction(Extraction(documents=[]), "xml")
