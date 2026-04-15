from fastapi.testclient import TestClient

from app.capabilities import detect_compare_pipelines
from app.main import app


client = TestClient(app)


def test_compare_requires_pipelines() -> None:
    response = client.post(
        "/extract/compare",
        files={"file": ("hello.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 422


def test_compare_rejects_unknown_pipeline() -> None:
    response = client.post(
        "/extract/compare",
        files={"file": ("hello.txt", b"hello", "text/plain")},
        data={"pipelines": "not_a_pipeline"},
    )
    assert response.status_code == 400
    assert "Unknown pipeline" in response.json()["detail"]


def test_compare_parser_on_text_file() -> None:
    response = client.post(
        "/extract/compare",
        files={"file": ("hello.txt", b"hello world\nsecond line", "text/plain")},
        data={"pipelines": "parser"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "hello.txt"
    assert "parser" in body["results"]
    result = body["results"]["parser"]
    assert result["ok"] is True
    assert result["raw_text_len"] > 0
    assert result["ocr_strategy"] == "never"
    assert "classification" in result


def test_compare_unavailable_pipeline_reported_not_failed(monkeypatch) -> None:
    # Force tesseract to look absent so ocr_tesseract is unavailable.
    import app.capabilities as caps

    monkeypatch.setattr(caps, "tesseract_available", lambda: False)
    monkeypatch.setattr(caps, "_vlm_endpoint_available", lambda: False)
    monkeypatch.setattr(caps, "_paddleocr_importable", lambda: False)

    response = client.post(
        "/extract/compare",
        files={"file": ("hello.txt", b"hello", "text/plain")},
        data={"pipelines": ["parser", "ocr_tesseract", "vlm_hybrid"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]["parser"]["ok"] is True
    assert body["results"]["ocr_tesseract"]["ok"] is False
    assert body["results"]["ocr_tesseract"]["available"] is False
    assert body["results"]["vlm_hybrid"]["ok"] is False


def test_compare_dedupes_pipelines() -> None:
    response = client.post(
        "/extract/compare",
        files={"file": ("hello.txt", b"hello", "text/plain")},
        data={"pipelines": ["parser", "parser"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pipelines"] == ["parser"]
    assert list(body["results"]) == ["parser"]


def test_capabilities_lists_compare_section() -> None:
    response = client.get("/capabilities")
    assert response.status_code == 200
    caps = response.json()
    assert "compare" in caps
    assert "parser" in caps["compare"]["available_pipelines"]


def test_detect_compare_pipelines_parser_always_available(monkeypatch) -> None:
    import app.capabilities as caps

    monkeypatch.setattr(caps, "tesseract_available", lambda: False)
    monkeypatch.setattr(caps, "_vlm_endpoint_available", lambda: False)
    monkeypatch.setattr(caps, "_paddleocr_importable", lambda: False)
    out = detect_compare_pipelines()
    assert "parser" in out["available_pipelines"]
    assert out["backends"]["tesseract"] is False
    assert out["backends"]["vlm"] is False
