"""FastAPI endpoint smoke tests with monkey-patched core."""

from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _stub_extraction_dict():
    return {
        "documents": [{
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
        }]
    }


@pytest.fixture
def client(monkeypatch):
    # Import here so pytest can skip cleanly if fastapi isn't installed.
    from loci_extract.api import server
    from loci_extract.schema import Extraction

    def fake_extract(path, opts, progress_callback=None):
        return Extraction.model_validate(_stub_extraction_dict())

    monkeypatch.setattr(server, "extract_document", fake_extract)
    return TestClient(server.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_capabilities(client):
    r = client.get("/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert "ocr_engines" in body
    assert "llm" in body


def test_extract_json(client):
    files = {"file": ("test.pdf", b"%PDF-1.4\n%EOF\n", "application/pdf")}
    r = client.post("/extract", files=files, data={"format": "json"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["documents"][0]["document_type"] == "W2"


def test_extract_csv(client):
    files = {"file": ("test.pdf", b"%PDF\n", "application/pdf")}
    r = client.post("/extract", files=files, data={"format": "csv"})
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "document_type" in r.text


def test_extract_batch(client):
    files = [
        ("files", ("a.pdf", b"%PDF\n", "application/pdf")),
        ("files", ("b.pdf", b"%PDF\n", "application/pdf")),
    ]
    r = client.post("/extract/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["filename"] == "a.pdf"


def test_auth_required_when_key_set(monkeypatch):
    monkeypatch.setenv("LOCI_EXTRACT_API_KEY", "secret-key")
    # Reload server module so it picks up the env var at import time.
    import importlib

    from loci_extract.api import server as s
    importlib.reload(s)

    def fake_extract(path, opts, progress_callback=None):
        from loci_extract.schema import Extraction
        return Extraction.model_validate(_stub_extraction_dict())

    monkeypatch.setattr(s, "extract_document", fake_extract)
    c = TestClient(s.app)
    r = c.get("/capabilities")
    assert r.status_code == 401

    r2 = c.get("/capabilities", headers={"Authorization": "Bearer secret-key"})
    assert r2.status_code == 200
    # /healthz always open
    r3 = c.get("/healthz")
    assert r3.status_code == 200

    # Clean up: unset env and reload again so later tests aren't affected
    monkeypatch.delenv("LOCI_EXTRACT_API_KEY", raising=False)
    importlib.reload(s)
