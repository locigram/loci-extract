"""JSON parse, retry, and SSN redaction tests using a stub client."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from loci_extract.llm import (
    extract_json_object,
    parse_extraction,
    redact_ssn_in_output,
    strip_code_fence,
)
from tests.conftest import StubLlmClient


def _w2_json():
    return json.dumps({
        "documents": [{
            "document_type": "W2",
            "tax_year": 2025,
            "data": {
                "employer": {"name": "Acme", "ein": "12-3456789", "address": "123 Main"},
                "employee": {"name": "Jane", "ssn_last4": "XXX-XX-1234", "address": "456 Elm"},
                "federal": {"box1_wages": 10000.0, "box2_federal_withheld": 1500.0,
                             "box3_ss_wages": 10000.0, "box4_ss_withheld": 620.0,
                             "box5_medicare_wages": 10000.0, "box6_medicare_withheld": 145.0},
                "box12": [],
                "box13": {},
                "box14_other": [],
                "state": [],
                "local": [],
            },
            "metadata": {"notes": []}
        }]
    })


def test_strip_code_fence_removes_json_block():
    assert strip_code_fence("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert strip_code_fence("```\n{\"a\":1}\n```") == '{"a":1}'
    assert strip_code_fence('{"a":1}') == '{"a":1}'


def test_extract_json_object_handles_preamble():
    raw = 'Sure, here is the data:\n\n{"documents": []}\n\nHope that helps!'
    assert extract_json_object(raw) == '{"documents": []}'


def test_extract_json_object_handles_nested():
    raw = '{"documents": [{"document_type": "W2", "data": {"nested": {"a": 1}}}]}'
    assert '"W2"' in extract_json_object(raw)


def test_redact_ssn_masks_full_ssn():
    data = {"ssn": "123-45-6789", "note": "SSN is 123-45-6789"}
    out = redact_ssn_in_output(data)
    assert out["ssn"] == "XXX-XX-6789"
    assert out["note"] == "SSN is XXX-XX-6789"


def test_redact_ssn_preserves_ein():
    # EIN format is 12-3456789 — should NOT match the SSN regex.
    data = {"ein": "12-3456789"}
    out = redact_ssn_in_output(data)
    assert out["ein"] == "12-3456789"


def test_parse_extraction_success():
    client = StubLlmClient([_w2_json()])
    extraction = parse_extraction(client, "Fake W-2 text", system_prompt="SYS", retry=0)
    assert len(extraction.documents) == 1
    assert extraction.documents[0].document_type == "W2"


def test_parse_extraction_retries_on_invalid_json():
    # First response is broken, second is valid.
    client = StubLlmClient(["not json at all", _w2_json()])
    extraction = parse_extraction(client, "Fake W-2 text", system_prompt="SYS", retry=1)
    assert len(extraction.documents) == 1


def test_parse_extraction_raises_after_retries_exhausted():
    client = StubLlmClient(["bad", "still bad", "still bad"])
    with pytest.raises((ValueError, ValidationError, json.JSONDecodeError, Exception)):
        parse_extraction(client, "text", system_prompt="SYS", retry=2)


def test_parse_extraction_redacts_by_default():
    # Payload contains a pre-redacted SSN + a full-SSN lookalike in notes.
    raw = json.loads(_w2_json())
    raw["documents"][0]["metadata"]["notes"] = ["Taxpayer SSN: 123-45-6789"]
    client = StubLlmClient([json.dumps(raw)])
    extraction = parse_extraction(client, "text", system_prompt="SYS", retry=0, redact=True)
    notes = extraction.documents[0].metadata.notes
    assert notes[0] == "Taxpayer SSN: XXX-XX-6789"


def test_parse_extraction_redact_off():
    raw = json.loads(_w2_json())
    raw["documents"][0]["metadata"]["notes"] = ["Taxpayer SSN: 123-45-6789"]
    client = StubLlmClient([json.dumps(raw)])
    extraction = parse_extraction(client, "text", system_prompt="SYS", retry=0, redact=False)
    assert "123-45-6789" in extraction.documents[0].metadata.notes[0]
