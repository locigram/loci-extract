"""Tests for OCR backend abstraction, CUDA detection, ocr_backend plumbing,
layout classification, and LLM enrichment.

All tests use monkeypatch — no GPU, PaddleOCR, or LLM endpoint required.
"""

from pathlib import Path

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from app.extractors.image_ocr import ImageOcrExtractor
from app.extractors.pdf import PdfExtractor
from app.main import app
from app.ocr import _build_table_candidates_from_words
from app.ocr_backends import OcrBackendNotAvailableError, WordBox, get_backend


client = TestClient(app)


# --- get_backend resolution ---


def test_get_backend_auto_returns_tesseract_when_paddle_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.ocr_backends.paddleocr_backend._paddleocr_importable", lambda: False)
    backend = get_backend("auto")
    assert backend.name == "tesseract"


def test_get_backend_paddleocr_raises_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.ocr_backends.paddleocr_backend._paddleocr_importable", lambda: False)
    try:
        get_backend("paddleocr")
        assert False, "Should have raised OcrBackendNotAvailableError"
    except OcrBackendNotAvailableError:
        pass


def test_get_backend_tesseract_returns_tesseract(monkeypatch) -> None:
    monkeypatch.setattr("app.ocr_backends.tesseract_backend.tesseract_available", lambda: True)
    backend = get_backend("tesseract")
    assert backend.name == "tesseract"


# --- Capabilities GPU/backend keys ---


def test_capabilities_includes_gpu_and_ocr_backends_keys() -> None:
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()
    # Original keys still present
    assert "ocr" in payload
    assert "pdf" in payload
    assert "documents" in payload
    # New additive keys
    assert "gpu" in payload
    assert "cuda" in payload["gpu"]
    assert "providers" in payload["gpu"]
    assert payload["gpu"]["cuda"]["available"] is False  # CI has no CUDA
    assert "ocr_backends" in payload
    assert "tesseract" in payload["ocr_backends"]
    assert "paddleocr" in payload["ocr_backends"]
    assert "default" in payload["ocr_backends"]


def test_cuda_available_env_override(monkeypatch) -> None:
    from app.capabilities import cuda_available

    monkeypatch.setenv("LOCI_EXTRACT_FORCE_CUDA", "1")
    result = cuda_available()
    assert result["available"] is True
    assert result["provider"] == "env_override"

    monkeypatch.setenv("LOCI_EXTRACT_FORCE_CUDA", "0")
    result = cuda_available()
    assert result["available"] is False


# --- ocr_backend validation ---


def test_extract_rejects_invalid_ocr_backend() -> None:
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"ocr_backend": "invalid_backend"},
    )
    assert response.status_code == 400
    assert "ocr_backend" in response.json()["detail"]


def test_extract_accepts_valid_ocr_backends() -> None:
    for backend_name in ("auto", "tesseract", "paddleocr"):
        response = client.post(
            "/extract",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"ocr_backend": backend_name},
        )
        # Plain text doesn't use OCR, so all should succeed
        assert response.status_code == 200


# --- ocr_backend_requested round-trip ---


def test_extra_ocr_backend_requested_round_trips() -> None:
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"ocr_backend": "tesseract"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["extra"]["ocr_backend_requested"] == "tesseract"


# --- Image OCR backend stamping ---


def test_image_ocr_stamps_backend_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.extractors.image_ocr.tesseract_available", lambda: True)
    monkeypatch.setattr("app.ocr.pytesseract.image_to_string", lambda image: "test ocr text result")

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 50), color="white").save(image_path)

    payload = ImageOcrExtractor().extract(
        image_path, "sample.png", "image/png", ocr_backend="tesseract"
    )
    assert payload.extra["ocr_backend"] == "tesseract"
    assert payload.extra["ocr_backend_requested"] == "tesseract"


def test_image_ocr_paddleocr_unavailable_returns_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.ocr_backends.paddleocr_backend._paddleocr_importable", lambda: False)

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (50, 50), color="white").save(image_path)

    payload = ImageOcrExtractor().extract(
        image_path, "sample.png", "image/png", ocr_backend="paddleocr"
    )
    assert payload.extraction.status == "partial"
    warning_codes = {w.code for w in payload.extraction.warnings}
    assert "paddleocr_not_available" in warning_codes
    assert payload.extra["ocr_backend"] is None


# --- PDF extractor backend plumbing ---


def test_pdf_extractor_stamps_backend_in_extra(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.extractors.pdf.tesseract_available", lambda: True)
    monkeypatch.setattr("app.ocr.pytesseract.image_to_string", lambda image: "ocr text from pdf")

    pdf_path = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(
        pdf_path, "blank.pdf", "application/pdf", ocr_strategy="auto", ocr_backend="tesseract"
    )
    assert payload.extra["ocr_backend"] == "tesseract"
    assert payload.extra["ocr_backend_requested"] == "tesseract"


def test_pdf_extractor_ocrmypdf_provenance_preserved(monkeypatch, tmp_path: Path) -> None:
    """The ocrmypdf path must keep extra['ocr_backend'] = 'ocrmypdf' regardless of ocr_backend param."""
    monkeypatch.setattr("app.ocr_backends.tesseract_backend.tesseract_available", lambda: True)
    monkeypatch.setattr("app.extractors.pdf.tesseract_available", lambda: True)
    monkeypatch.setattr("app.extractors.pdf.ocrmypdf_available", lambda: True)

    pdf_path = tmp_path / "glyph-garbage.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "parser text")
    document.save(pdf_path)
    document.close()

    extractor = PdfExtractor()

    parser_calls = {"count": 0}

    def fake_extract_text(document, max_pages=None):
        parser_calls["count"] += 1
        if parser_calls["count"] == 1:
            return {1: "(cid:6)(cid:30)(cid:30)(cid:40)(cid:45)(cid:39)(cid:44)"}
        return {1: "Clean text after OCRmyPDF"}

    monkeypatch.setattr(extractor, "_extract_pdf_text", fake_extract_text)
    monkeypatch.setattr(extractor, "_extract_pdf_tables", lambda file_path, max_pages=None: {})

    rerouted_pdf = tmp_path / "rerouted.pdf"
    rerouted_pdf.write_bytes(pdf_path.read_bytes())
    monkeypatch.setattr(extractor, "_run_ocrmypdf_fallback", lambda file_path: rerouted_pdf)

    payload = extractor.extract(
        pdf_path, "glyph-garbage.pdf", "application/pdf", ocr_strategy="auto", ocr_backend="tesseract"
    )
    # The ocrmypdf path is parser-layer and orthogonal to the tesseract/paddle selection
    assert payload.extra["ocr_backend"] == "ocrmypdf"
    assert payload.extra["ocrmypdf_trigger_reason"] == "parser_glyph_garbage"


def test_pdf_force_image_ignores_text_layer(monkeypatch, tmp_path: Path) -> None:
    """force_image OCRs the rendered page image and ignores the text layer entirely."""
    monkeypatch.setattr("app.extractors.pdf.tesseract_available", lambda: True)
    monkeypatch.setattr("app.ocr.pytesseract.image_to_string", lambda image: "ocr from rendered image")

    pdf_path = tmp_path / "junk-overlay.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "this is junk overlay text that should be ignored")
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(
        pdf_path, "junk-overlay.pdf", "application/pdf", ocr_strategy="force_image"
    )

    assert payload.extraction.ocr_used is True
    assert payload.extraction.status == "success"
    assert payload.raw_text == "ocr from rendered image"
    assert payload.extra["ocr_strategy"] == "force_image"
    assert payload.extra["result_source"] == "ocr_force_image"
    assert payload.extra["text_layer_ignored"] is True
    assert payload.extra["ocr_attempted"] is True
    # The junk text layer is NOT in the output
    assert "junk overlay" not in payload.raw_text


def test_pdf_force_image_no_backend_returns_partial(monkeypatch, tmp_path: Path) -> None:
    """force_image with no OCR backend available returns partial."""
    monkeypatch.setattr("app.extractors.pdf.tesseract_available", lambda: False)

    pdf_path = tmp_path / "test.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(
        pdf_path, "test.pdf", "application/pdf", ocr_strategy="force_image"
    )

    assert payload.extraction.status == "partial"
    warning_codes = {w.code for w in payload.extraction.warnings}
    assert "ocr_not_available" in warning_codes


def test_extract_api_accepts_force_image() -> None:
    """The /extract endpoint accepts force_image as a valid ocr_strategy."""
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"ocr_strategy": "force_image"},
    )
    assert response.status_code == 200


# --- Phase 2: Table candidates from word boxes ---


def test_build_table_candidates_from_words_basic() -> None:
    """_build_table_candidates_from_words produces table output from word dicts."""
    words = [
        {"text": "Name", "left": 10, "top": 10, "width": 40, "height": 12, "center_y": 16.0, "conf": 0.9},
        {"text": "Role", "left": 200, "top": 10, "width": 40, "height": 12, "center_y": 16.0, "conf": 0.9},
        {"text": "Drew", "left": 10, "top": 30, "width": 40, "height": 12, "center_y": 36.0, "conf": 0.9},
        {"text": "Admin", "left": 200, "top": 30, "width": 50, "height": 12, "center_y": 36.0, "conf": 0.9},
    ]
    candidates = _build_table_candidates_from_words(words)
    assert len(candidates) == 1
    assert candidates[0]["row_count"] == 2
    assert candidates[0]["column_count"] == 2
    assert candidates[0]["detection_method"] == "ocr_word_grid"


def test_build_table_candidates_from_words_paddleocr_detection_method() -> None:
    words = [
        {"text": "A", "left": 10, "top": 10, "width": 20, "height": 10, "center_y": 15.0, "conf": 0.9},
        {"text": "B", "left": 200, "top": 10, "width": 20, "height": 10, "center_y": 15.0, "conf": 0.9},
        {"text": "C", "left": 10, "top": 30, "width": 20, "height": 10, "center_y": 35.0, "conf": 0.9},
        {"text": "D", "left": 200, "top": 30, "width": 20, "height": 10, "center_y": 35.0, "conf": 0.9},
    ]
    candidates = _build_table_candidates_from_words(words, detection_method="paddleocr_word_grid")
    assert len(candidates) == 1
    assert candidates[0]["detection_method"] == "paddleocr_word_grid"


def test_extract_table_candidates_with_backend_words() -> None:
    """_extract_table_candidates accepts WordBox list from PaddleOCR."""
    from app.ocr import _extract_table_candidates

    words = [
        WordBox(text="Name", left=10, top=10, width=40, height=12, conf=0.9),
        WordBox(text="Role", left=200, top=10, width=40, height=12, conf=0.9),
        WordBox(text="Drew", left=10, top=30, width=40, height=12, conf=0.9),
        WordBox(text="Admin", left=200, top=30, width=50, height=12, conf=0.9),
    ]
    candidates = _extract_table_candidates(Image.new("RGB", (300, 50)), backend_words=words)
    assert len(candidates) == 1
    assert candidates[0]["detection_method"] == "paddleocr_word_grid"


def test_paddleocr_single_pass_mode(monkeypatch) -> None:
    """PaddleOCR backend uses single-pass mode with paddleocr_native pass name."""
    from app.ocr import extract_best_ocr_result
    from app.ocr_backends import OcrBackendResult

    class FakePaddleBackend:
        @property
        def name(self):
            return "paddleocr"

        def is_available(self):
            return True

        def run(self, image, *, variant_name=None, rotation=0):
            return OcrBackendResult(
                text="Financial statement total assets 100000",
                confidence=0.95,
                words=[
                    WordBox(text="Financial", left=10, top=10, width=80, height=12, conf=0.95),
                    WordBox(text="statement", left=100, top=10, width=80, height=12, conf=0.95),
                    WordBox(text="total", left=10, top=30, width=40, height=12, conf=0.95),
                    WordBox(text="assets", left=100, top=30, width=50, height=12, conf=0.95),
                    WordBox(text="100000", left=200, top=30, width=60, height=12, conf=0.95),
                ],
                raw={},
            )

    monkeypatch.setattr("app.ocr.pytesseract.image_to_data", lambda *args, **kwargs: {"text": []})
    result = extract_best_ocr_result(Image.new("RGB", (300, 50)), backend=FakePaddleBackend())

    assert result["selected_pass"] == "paddleocr_native"
    assert result["selected_rotation"] == 0
    assert result["backend"] == "paddleocr"
    assert result["score"] > 0
    assert len(result["ocr_passes"]) == 1
    assert result["ocr_passes"][0]["name"] == "paddleocr_native"


# --- Phase 3: Layout classification ---


def test_layout_classifier_skipped_when_cuda_missing(monkeypatch) -> None:
    """classify_layout returns None when CUDA is not available."""
    from app.classification.layout import classify_layout

    monkeypatch.setattr("app.classification.layout._cuda_is_available", lambda: False)
    result = classify_layout(Image.new("RGB", (100, 100)))
    assert result is None


def test_layout_classifier_skipped_when_env_off(monkeypatch) -> None:
    from app.classification.layout import classify_layout

    monkeypatch.setenv("LOCI_EXTRACT_LAYOUT_CLASSIFIER", "off")
    result = classify_layout(Image.new("RGB", (100, 100)))
    assert result is None


def test_layout_classifier_promotes_when_available(monkeypatch) -> None:
    """classify_document uses layout result when confidence >= 0.75."""
    from app.classification.layout import LayoutClassification
    from app.classification.rules import classify_document

    def fake_classify_layout(image, *, page_number=1, filename=""):
        return LayoutClassification(
            doc_type="w2",
            confidence=0.85,
            strategy="layout",
            matched_signals=["title_keyword_match:1", "structured_form_layout"],
            regions=[],
        )

    monkeypatch.setattr("app.classification.layout.classify_layout", fake_classify_layout)

    result = classify_document(
        filename="scan.pdf",
        mime_type="application/pdf",
        raw_text="some unrecognizable text",
        page_image=Image.new("RGB", (100, 100)),
    )
    assert result.doc_type == "w2"
    assert result.strategy == "layout"
    assert result.confidence == 0.85


def test_layout_classifier_falls_back_to_rules(monkeypatch) -> None:
    """When layout confidence is low, fall back to rule-based classification."""
    from app.classification.layout import LayoutClassification
    from app.classification.rules import classify_document

    def fake_classify_layout(image, *, page_number=1, filename=""):
        return LayoutClassification(
            doc_type="receipt",
            confidence=0.3,
            strategy="layout",
            matched_signals=[],
            regions=[],
        )

    monkeypatch.setattr("app.classification.layout.classify_layout", fake_classify_layout)

    result = classify_document(
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement",
        page_image=Image.new("RGB", (100, 100)),
    )
    # Should fall through to rules and detect W-2
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


def test_classify_document_without_page_image_uses_rules() -> None:
    """When no page_image, layout is not attempted."""
    from app.classification.rules import classify_document

    result = classify_document(
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement",
    )
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


# --- Phase 4: LLM enrichment ---


def test_llm_enrichment_skipped_when_disabled() -> None:
    from app.structured.financial_statement import _maybe_enrich_sections_with_llm

    result = _maybe_enrich_sections_with_llm([], [], enable_llm_enrichment=False)
    assert result["attempted"] is False
    assert result["reason"] == "disabled"


def test_llm_enrichment_skipped_when_client_not_configured() -> None:
    from app.llm.config import reset_llm_client
    from app.structured.financial_statement import _maybe_enrich_sections_with_llm

    reset_llm_client()
    result = _maybe_enrich_sections_with_llm(
        [{"account_name": "Test", "section": None}],
        [],
        enable_llm_enrichment=True,
    )
    assert result["attempted"] is True
    assert result["applied"] is False
    assert result["reason"] == "llm_client_not_configured"
    reset_llm_client()


def test_llm_enrichment_graceful_on_failure(monkeypatch) -> None:
    from app.llm.client import LlmClient
    from app.llm.config import register_endpoint, reset_llm_client
    from app.structured.financial_statement import _maybe_enrich_sections_with_llm

    reset_llm_client()

    stub_client = LlmClient(base_url="http://fake", model="test-model")
    monkeypatch.setattr(stub_client, "complete_json", lambda *args, **kwargs: None)
    register_endpoint("default", stub_client)

    line_items = [{"account_name": "Unknown Account", "section": None}]
    sections: list = []
    result = _maybe_enrich_sections_with_llm(
        line_items, sections, enable_llm_enrichment=True
    )
    assert result["attempted"] is True
    assert result["applied"] is False
    assert result["reason"] == "llm_returned_none"
    assert line_items[0]["section"] is None

    reset_llm_client()


def test_llm_enrichment_applies_relabels(monkeypatch) -> None:
    from app.llm.client import LlmClient
    from app.llm.config import register_endpoint, reset_llm_client
    from app.structured.financial_statement import _maybe_enrich_sections_with_llm

    reset_llm_client()

    stub_client = LlmClient(base_url="http://fake", model="test-model")
    monkeypatch.setattr(
        stub_client,
        "complete_json",
        lambda *args, **kwargs: {
            "assignments": [
                {"account_name": "Cash in Bank", "section": "Assets"},
                {"account_name": "Accounts Payable", "section": "Liabilities"},
            ]
        },
    )
    register_endpoint("default", stub_client)

    line_items = [
        {"account_name": "Cash in Bank", "section": None, "is_total": False},
        {"account_name": "Accounts Payable", "section": None, "is_total": False},
        {"account_name": "Equipment", "section": "Assets", "is_total": False},
    ]
    sections: list = []
    result = _maybe_enrich_sections_with_llm(
        line_items, sections, enable_llm_enrichment=True
    )
    assert result["attempted"] is True
    assert result["applied"] is True
    assert result["items_relabeled"] == 2
    assert result["model"] == "test-model"
    assert line_items[0]["section"] == "Assets"
    assert line_items[1]["section"] == "Liabilities"
    assert line_items[2]["section"] == "Assets"  # untouched
    assert len(sections) >= 2

    reset_llm_client()


def test_extract_structured_enable_llm_enrichment_field() -> None:
    """The enable_llm_enrichment Form field is accepted and round-trips in extra."""
    response = client.post(
        "/extract/structured",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"enable_llm_enrichment": "true"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["extra"]["enable_llm_enrichment"] is True


def test_financial_statement_includes_llm_enrichment_in_fields() -> None:
    """Financial statement structured output includes llm_enrichment metadata."""
    response = client.post(
        "/extract/structured",
        files={"file": ("balance-sheet.txt", b"Balance Sheet\nAccount Number\nAccount Name\nAccounting Basis: Accrual", "text/plain")},
        data={"doc_type_hint": "financial_statement"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["structured"]["fields"]["llm_enrichment"]["attempted"] is False
    assert payload["structured"]["fields"]["llm_enrichment"]["reason"] == "disabled"
