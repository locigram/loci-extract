"""Tests for vlm_trace population in the VLM extraction paths.

These use a fake LlmClient to avoid any network calls, and rely on PyMuPDF
to build a tiny in-memory PDF. No GPU or Ollama required.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.extractors.pdf import PdfExtractor
from app.extractors.vlm import vlm_extract_page


class _FakeClient:
    model = "fake-vlm"
    base_url = "http://fake"
    api_key = None
    timeout = 30.0

    def __init__(self, response: dict | None):
        self._response = response
        self.calls = 0

    def vision_extract_json(self, system, user_text, image, schema=None):
        self.calls += 1
        return self._response

    def complete_json(self, system, user_text, schema=None):
        return {"usable": True, "reason": "looks good", "confidence": 0.9}


def _make_pdf(text: str = "Hello world from PyMuPDF") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def test_vlm_extract_page_records_trace_on_success(monkeypatch) -> None:
    # Force single-pass so the attempt labels match the legacy schema.
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "0")
    client = _FakeClient({"raw_text": "extracted text from image", "fields": {"x": 1}})
    from PIL import Image

    img = Image.new("RGB", (100, 200), color="white")
    trace: dict = {}
    result = vlm_extract_page(client, img, doc_type="unknown", trace=trace)
    assert result is not None
    assert trace["attempt"] == "structured"
    assert trace["parsed_ok"] is True
    assert trace["response_chars"] == len("extracted text from image")
    assert trace["had_fields"] is True
    assert trace["image_size"] == [100, 200]


def test_vlm_extract_page_two_pass_records_trace(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "1")
    client = _FakeClient({
        "raw_text": "Form W-2 wages and tax statement\nEmployer: Acme",
        "doc_type": "w2",
        "confidence": 0.95,
        "fields": {"employer_name": "Acme"},
    })
    from PIL import Image

    img = Image.new("RGB", (400, 500), color="white")
    trace: dict = {}
    result = vlm_extract_page(client, img, doc_type="unknown", trace=trace)
    assert result is not None
    assert trace["attempt"] == "two_pass"
    assert trace["parsed_ok"] is True
    assert trace["pass1"]["doc_type_resolved"] == "w2"
    # w2 is in _TWO_PASS_DOC_TYPES so pass 2 should have run
    assert trace["pass2"]["ok"] is True


def test_vlm_extract_page_two_pass_unknown_skips_pass2(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "1")
    client = _FakeClient({
        "raw_text": "some generic document text",
        "doc_type": "unknown",
        "confidence": 0.5,
    })
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="white")
    trace: dict = {}
    result = vlm_extract_page(client, img, doc_type="unknown", trace=trace)
    assert result is not None
    assert result["raw_text"] == "some generic document text"
    assert trace["pass2"]["skipped"] is True


def test_vlm_extract_page_records_trace_on_none(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "0")
    client = _FakeClient(None)
    from PIL import Image

    img = Image.new("RGB", (50, 50), color="white")
    trace: dict = {}
    # The raw fallback hits httpx.post; block it by making the client fail too
    result = vlm_extract_page(client, img, trace=trace)
    # When all three stages fail to produce text, trace records the last attempt
    # tried. It may be 'plain_text_json' (if fallback also returned None) or
    # 'raw_fallback'/'none' depending on the http path. Just assert it is set.
    assert "attempt" in trace
    assert trace["image_size"] == [50, 50]
    # parsed_ok falsy if no usable result
    if result is None:
        assert trace["parsed_ok"] is False


def test_vlm_hybrid_populates_vlm_trace(monkeypatch, tmp_path: Path) -> None:
    """_extract_vlm_hybrid builds extra.vlm_trace with per-page stage info."""
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "0")
    pdf_bytes = _make_pdf()
    pdf_path = tmp_path / "hello.pdf"
    pdf_path.write_bytes(pdf_bytes)

    fake_vlm = _FakeClient({"raw_text": "vlm produced this", "fields": {}})
    fake_llm = _FakeClient(None)  # complete_json is separately mocked

    # The hybrid path imports these lazily; patch the module-level getters.
    monkeypatch.setattr("app.llm.config.get_vlm_client", lambda: fake_vlm)
    monkeypatch.setattr("app.llm.config.get_llm_client", lambda name="default": fake_llm)

    # Force the parser text through verification that says "not usable" so VLM runs
    monkeypatch.setattr(
        "app.extractors.vlm.verify_text_quality",
        lambda client, text, min_confidence=0.7: {"usable": False, "reason": "forced_vlm", "confidence": 1.0},
    )

    extractor = PdfExtractor()
    payload = extractor.extract(pdf_path, "hello.pdf", "application/pdf", ocr_strategy="vlm_hybrid")
    trace = payload.extra.get("vlm_trace")
    assert trace is not None
    assert trace["pipeline"] == "vlm_hybrid"
    assert trace["vlm_model"] == "fake-vlm"
    assert len(trace["pages"]) >= 1
    page = trace["pages"][0]
    assert page["stage_selected"] == "vlm"
    assert page["verify"]["reason"] == "forced_vlm"
    assert page["vlm"] is not None
    assert page["final_chars"] > 0
    assert trace["totals"]["vlm_used"] >= 1


def test_vlm_pure_populates_vlm_trace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCI_EXTRACT_VLM_TWO_PASS", "0")
    pdf_bytes = _make_pdf()
    pdf_path = tmp_path / "hello.pdf"
    pdf_path.write_bytes(pdf_bytes)

    fake_vlm = _FakeClient({"raw_text": "pure vlm text", "fields": {}})
    monkeypatch.setattr("app.llm.config.get_vlm_client", lambda: fake_vlm)

    extractor = PdfExtractor()
    payload = extractor.extract(pdf_path, "hello.pdf", "application/pdf", ocr_strategy="vlm")
    trace = payload.extra.get("vlm_trace")
    assert trace is not None
    assert trace["pipeline"] == "vlm"
    assert trace["verify_model"] is None
    assert trace["pages"][0]["stage_selected"] == "vlm"
    assert trace["pages"][0]["vlm"]["attempt"] == "structured"
    # Render DPI now 216 (3x matrix), up from legacy 144
    assert trace["pages"][0]["vlm"]["render_dpi"] == 216


def test_vlm_hybrid_parser_wins_no_vlm_call(monkeypatch, tmp_path: Path) -> None:
    pdf_bytes = _make_pdf("Clean readable text on the page")
    pdf_path = tmp_path / "clean.pdf"
    pdf_path.write_bytes(pdf_bytes)

    fake_vlm = _FakeClient({"raw_text": "should not be called", "fields": {}})
    fake_llm = _FakeClient(None)
    monkeypatch.setattr("app.llm.config.get_vlm_client", lambda: fake_vlm)
    monkeypatch.setattr("app.llm.config.get_llm_client", lambda name="default": fake_llm)
    monkeypatch.setattr(
        "app.extractors.vlm.verify_text_quality",
        lambda client, text, min_confidence=0.7: {"usable": True, "reason": "clean", "confidence": 0.95},
    )

    extractor = PdfExtractor()
    payload = extractor.extract(pdf_path, "clean.pdf", "application/pdf", ocr_strategy="vlm_hybrid")
    trace = payload.extra["vlm_trace"]
    assert trace["pages"][0]["stage_selected"] == "parser"
    assert trace["pages"][0]["vlm"] is None
    assert fake_vlm.calls == 0


@pytest.mark.parametrize("strategy", ["vlm", "vlm_hybrid"])
def test_vlm_paths_report_no_endpoint_when_client_missing(monkeypatch, tmp_path: Path, strategy: str) -> None:
    pdf_bytes = _make_pdf()
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(pdf_bytes)

    monkeypatch.setattr("app.llm.config.get_vlm_client", lambda: None)
    monkeypatch.setattr("app.llm.config.get_llm_client", lambda name="default": None)

    extractor = PdfExtractor()
    payload = extractor.extract(pdf_path, "x.pdf", "application/pdf", ocr_strategy=strategy)
    warning_codes = [w.code for w in payload.extraction.warnings]
    assert "vlm_not_available" in warning_codes
