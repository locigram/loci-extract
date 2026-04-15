"""Tests for extraction profiles — loading, validation, merge logic, and routing."""

from PIL import Image
from fastapi.testclient import TestClient

from app.classification.routing import classify_with_profile
from app.main import app
from app.profiles import get_profile, list_profiles, reset_cache
from app.profiles.schema import ClassifierConfig, ExtractionProfile


client = TestClient(app)


# --- Profile loading ---


def test_list_profiles_returns_built_in_names() -> None:
    reset_cache()
    names = list_profiles()
    assert "general" in names
    assert "tax" in names
    assert "financial" in names
    assert "receipt" in names


def test_get_profile_returns_valid_profile() -> None:
    reset_cache()
    profile = get_profile("tax")
    assert profile is not None
    assert profile.name == "tax"
    assert profile.classifier.strategy == "auto"
    assert profile.classifier.model_name == "hsarfraz/donut-irs-tax-docs-classifier"
    assert profile.mask_pii is False


def test_get_profile_unknown_returns_none() -> None:
    reset_cache()
    assert get_profile("nonexistent") is None


def test_general_profile_has_rules_strategy() -> None:
    reset_cache()
    profile = get_profile("general")
    assert profile is not None
    assert profile.classifier.strategy == "rules"
    assert profile.mask_pii is False


def test_receipt_profile_has_ocr_always() -> None:
    reset_cache()
    profile = get_profile("receipt")
    assert profile is not None
    assert profile.ocr_strategy == "always"


def test_financial_profile_enables_llm() -> None:
    reset_cache()
    profile = get_profile("financial")
    assert profile is not None
    assert profile.enable_llm_enrichment is True
    assert profile.classifier.strategy == "layout"


# --- Profile schema validation ---


def test_extraction_profile_defaults() -> None:
    profile = ExtractionProfile(name="test")
    assert profile.ocr_strategy == "auto"
    assert profile.ocr_backend == "auto"
    assert profile.classifier.strategy == "rules"
    assert profile.mask_pii is True
    assert profile.enable_llm_enrichment is False


# --- Profile-Form merge (API level) ---


def test_extract_with_unknown_profile_returns_404() -> None:
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"extraction_profile": "nonexistent"},
    )
    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]


def test_extract_with_profile_records_in_extra() -> None:
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"extraction_profile": "general"},
    )
    assert response.status_code == 200
    assert response.json()["extra"]["extraction_profile"] == "general"


def test_extract_without_profile_backward_compatible() -> None:
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 200
    # No extraction_profile key when not provided
    assert "extraction_profile" not in response.json()["extra"]


def test_structured_with_profile_records_in_extra() -> None:
    response = client.post(
        "/extract/structured",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"extraction_profile": "general"},
    )
    assert response.status_code == 200
    assert response.json()["extra"]["extraction_profile"] == "general"


def test_structured_profile_override_ocr_strategy() -> None:
    """Explicit Form value overrides profile default."""
    response = client.post(
        "/extract",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"extraction_profile": "receipt", "ocr_strategy": "never"},
    )
    assert response.status_code == 200
    assert response.json()["extra"]["ocr_strategy"] == "never"


# --- Classifier routing ---


def test_classify_with_profile_none_uses_rules() -> None:
    """No profile delegates to classify_document (backward compat)."""
    result = classify_with_profile(
        profile=None,
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement",
    )
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


def test_classify_with_rules_strategy() -> None:
    profile = ExtractionProfile(
        name="test-rules",
        classifier=ClassifierConfig(strategy="rules"),
    )
    result = classify_with_profile(
        profile=profile,
        filename="receipt.png",
        mime_type="image/png",
        raw_text="Coffee Shop\nSubtotal 10.00\nTax 0.80\nTotal 10.80",
    )
    assert result.doc_type == "receipt"
    assert result.strategy == "rules"


def test_classify_with_donut_strategy_falls_back_when_unavailable(monkeypatch) -> None:
    """donut-irs strategy falls back to rules when Donut model not available."""
    monkeypatch.setattr(
        "app.classification.donut_classifier._donut_available", lambda: False
    )
    profile = ExtractionProfile(
        name="test-donut",
        classifier=ClassifierConfig(
            strategy="donut-irs",
            model_name="hsarfraz/donut-irs-tax-docs-classifier",
        ),
    )
    result = classify_with_profile(
        profile=profile,
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement",
        page_image=Image.new("RGB", (100, 100)),
    )
    # Falls back to rules
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


def test_classify_with_auto_strategy_falls_through_to_rules(monkeypatch) -> None:
    """auto strategy: donut unavailable, layout unavailable -> rules."""
    monkeypatch.setattr(
        "app.classification.donut_classifier._donut_available", lambda: False
    )
    monkeypatch.setattr(
        "app.classification.layout._cuda_is_available", lambda: False
    )
    profile = ExtractionProfile(
        name="test-auto",
        classifier=ClassifierConfig(
            strategy="auto",
            model_name="hsarfraz/donut-irs-tax-docs-classifier",
        ),
    )
    result = classify_with_profile(
        profile=profile,
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement",
        page_image=Image.new("RGB", (100, 100)),
    )
    assert result.doc_type == "w2"
    assert result.strategy == "rules"


def test_classify_hint_overrides_profile_strategy() -> None:
    """doc_type_hint always wins regardless of profile strategy."""
    profile = ExtractionProfile(
        name="test-hint",
        classifier=ClassifierConfig(strategy="donut-irs"),
    )
    result = classify_with_profile(
        profile=profile,
        filename="unknown.pdf",
        mime_type="application/pdf",
        raw_text="random text",
        doc_type_hint="receipt",
    )
    assert result.doc_type == "receipt"
    assert result.strategy == "hint"


# --- Donut classifier (monkeypatched) ---


def test_donut_classifier_returns_none_when_unavailable(monkeypatch) -> None:
    from app.classification.donut_classifier import classify_with_donut

    monkeypatch.setattr(
        "app.classification.donut_classifier._donut_available", lambda: False
    )
    result = classify_with_donut(Image.new("RGB", (100, 100)))
    assert result is None


def test_donut_classifier_returns_none_for_none_image() -> None:
    from app.classification.donut_classifier import classify_with_donut

    result = classify_with_donut(None)
    assert result is None


# --- Capabilities includes profiles ---


def test_capabilities_includes_profiles() -> None:
    reset_cache()
    response = client.get("/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert "profiles" in payload
    assert "available" in payload["profiles"]
    assert "general" in payload["profiles"]["available"]
    assert "tax" in payload["profiles"]["available"]
    assert "classifier_models" in payload["profiles"]
    assert payload["profiles"]["classifier_models"]["rules"] is True
