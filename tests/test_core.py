"""core.extract_document / extract_batch with fully stubbed LLM + file I/O."""

from __future__ import annotations

import json

import pytest

from loci_extract import core
from loci_extract.core import ExtractionOptions
from loci_extract.schema import Extraction
from tests.conftest import StubLlmClient


def _w2_json_with_dupes():
    """Emit two identical W-2 records to exercise the dedup path."""
    rec = {
        "document_type": "W2",
        "tax_year": 2025,
        "data": {
            "employer": {"name": "Acme", "ein": "12-3456789", "address": "123 Main"},
            "employee": {"name": "Jane", "ssn_last4": "XXX-XX-1234", "address": "456 Elm"},
            "federal": {"box1_wages": 10000.0, "box2_federal_withheld": 1500.0,
                         "box3_ss_wages": 10000.0, "box4_ss_withheld": 620.0,
                         "box5_medicare_wages": 10000.0, "box6_medicare_withheld": 145.0},
            "box12": [], "box13": {}, "box14_other": [], "state": [], "local": [],
        },
        "metadata": {"notes": []},
    }
    return json.dumps({"documents": [rec, rec]})


def test_extract_document_dedups_w2(monkeypatch, tmp_path):
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")

    monkeypatch.setattr(core, "detect_page_types", lambda p: {1: "text"})
    monkeypatch.setattr(core, "extract_text_pages", lambda p, pages: {1: "Form W-2 Wage and Tax Statement"})

    stub = StubLlmClient([_w2_json_with_dupes()])
    monkeypatch.setattr(core, "make_client", lambda url, api_key="local": stub)

    opts = ExtractionOptions(model_url="http://stub", model_name="stub", retry=0)
    extraction = core.extract_document(pdf, opts)
    assert len(extraction.documents) == 1


def test_extract_document_raises_on_empty_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    monkeypatch.setattr(core, "detect_page_types", lambda p: {1: "text"})
    monkeypatch.setattr(core, "extract_text_pages", lambda p, pages: {1: ""})
    monkeypatch.setattr(core, "make_client", lambda url, api_key="local": StubLlmClient([]))

    opts = ExtractionOptions(model_url="http://stub", model_name="stub")
    with pytest.raises(RuntimeError):
        core.extract_document(pdf, opts)


def test_extract_batch_continues_past_failures(monkeypatch, tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4\n%EOF\n")
    pdf_b.write_bytes(b"%PDF-1.4\n%EOF\n")

    monkeypatch.setattr(core, "detect_page_types", lambda p: {1: "text"})
    # A yields valid text, B yields nothing
    monkeypatch.setattr(core, "extract_text_pages", lambda p, pages: {1: "hello"} if "a.pdf" in str(p) else {1: ""})

    # Only one LLM call (for A)
    stub = StubLlmClient([json.dumps({"documents": []})])
    monkeypatch.setattr(core, "make_client", lambda url, api_key="local": stub)

    opts = ExtractionOptions(model_url="http://stub", model_name="stub", retry=0)
    results = core.extract_batch([pdf_a, pdf_b], opts)
    assert len(results) == 2
    # A succeeded (zero docs, empty Extraction), B failed → empty Extraction
    assert isinstance(results[0][1], Extraction)
    assert isinstance(results[1][1], Extraction)


def test_extract_document_includes_doc_type_hint(monkeypatch, tmp_path):
    """If detector finds W-2 keywords, the user prompt should include the hint."""
    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF\n")

    monkeypatch.setattr(core, "detect_page_types", lambda p: {1: "text"})
    monkeypatch.setattr(core, "extract_text_pages",
                        lambda p, pages: {1: "Form W-2 Wage and Tax Statement 2025"})

    captured = {}
    def fake_make(url, api_key="local"):
        # Return a stub and capture the text passed to chat completions
        s = StubLlmClient([json.dumps({"documents": []})])
        orig_create = s.create
        def create(**kwargs):
            captured["user_text"] = kwargs["messages"][-1]["content"]
            return orig_create(**kwargs)
        s.create = create
        return s

    monkeypatch.setattr(core, "make_client", fake_make)
    opts = ExtractionOptions(model_url="http://stub", model_name="stub", retry=0)
    # Stub extract_text_pages returned content; but then the prompt is empty
    # trimming check will pass only if we don't reject-empty. Use valid text.
    core.extract_document(pdf, opts)
    assert "W2" in captured["user_text"]
    assert "Wage and Tax Statement" in captured["user_text"]
