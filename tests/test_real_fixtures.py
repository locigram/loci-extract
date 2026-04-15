"""Integration tests against real (sanitized) PDF fixtures.

Each test is guarded with ``skip_if_no_fixture()`` so runs without the
fixtures on disk skip cleanly instead of failing. Live LLM tests are
additionally gated by the ``LOCI_EXTRACT_LIVE_LLM`` env var so `pytest -q`
on CI doesn't make network calls.
"""

from __future__ import annotations

import os

import pytest

from loci_extract.detector import get_extraction_strategy
from tests.fixtures import FIXTURE_REGISTRY, skip_if_no_fixture

# ---------------------------------------------------------------------------
# Phase 3a — encoding detection
# ---------------------------------------------------------------------------


@skip_if_no_fixture("appfolio_jan25_owner")
def test_appfolio_pscript5_is_encoding_broken():
    """The canonical AppFolio print-to-PDF fixture uses Identity-H CID
    encoding with no ToUnicode map (PScript5/Distiller workflow) — the
    detector must flag it and route to OCR. If this test ever fails, the
    encoding detector has silently regressed and text-only extraction will
    produce glyph garbage."""
    path = FIXTURE_REGISTRY["appfolio_jan25_owner"]
    result = get_extraction_strategy(path)
    assert result["encoding_broken"] is True, (
        f"Expected PScript5 fixture to be flagged encoding_broken. Got: {result}"
    )
    assert result["strategy"] == "ocr"
    assert "Identity-H" in result["reason"] or "ToUnicode" in result["reason"]


@skip_if_no_fixture("appfolio_income_statement")
def test_appfolio_macroman_is_clean_text_layer():
    """The alternate AppFolio export path (MacRoman single-byte encoding)
    is text-recoverable even with uni=no in pdffonts output. The detector
    must NOT falsely flag it as encoding-broken."""
    path = FIXTURE_REGISTRY["appfolio_income_statement"]
    result = get_extraction_strategy(path)
    assert result["encoding_broken"] is False, (
        f"Expected MacRoman fixture to parse cleanly. Got: {result}"
    )
    # Either "text" (pdfminer direct) or "pdfplumber" (coordinate-aware).
    assert result["strategy"] in ("text", "pdfplumber")


# ---------------------------------------------------------------------------
# Phase 3b — live LLM end-to-end (gated on LOCI_EXTRACT_LIVE_LLM=1)
# ---------------------------------------------------------------------------

live_llm_required = pytest.mark.skipif(
    os.getenv("LOCI_EXTRACT_LIVE_LLM") != "1",
    reason=(
        "Set LOCI_EXTRACT_LIVE_LLM=1 to run end-to-end tests against the "
        "configured LLM endpoint. Skipped by default to keep `pytest -q` offline."
    ),
)


@live_llm_required
@skip_if_no_fixture("appfolio_jan25_owner")
def test_appfolio_balance_sheet_end_to_end():
    """Full extraction against the canonical AppFolio fixture via the live
    LLM endpoint. This is the integration smoke check for the financial
    pipeline: OCR routing → boundary detection → chunked financial extract →
    totals verification → CSV output.

    Asserts ONLY structural expectations — the LLM output content can drift
    with model updates, so we don't lock values, just shapes."""
    from loci_extract import ExtractionOptions, extract_document

    path = FIXTURE_REGISTRY["appfolio_jan25_owner"]
    opts = ExtractionOptions(
        model_url=os.getenv("LOCI_EXTRACT_MODEL_URL", "http://10.10.100.20:9020/v1"),
        model_name=os.getenv("LOCI_EXTRACT_MODEL_NAME", "qwen3-vl-32b"),
        retry=1,
    )
    extraction = extract_document(path, opts)
    assert len(extraction.documents) >= 1, "Expected at least one document extracted"
    for doc in extraction.documents:
        # Metadata should reflect the encoding-broken routing
        assert doc.metadata.encoding_broken is True or doc.metadata.extraction_strategy == "ocr"
        # AppFolio signature
        software = (doc.data or {}).get("entity", {}).get("software", "")
        assert "appfolio" in software.lower() or doc.document_type in {
            "BALANCE_SHEET", "INCOME_STATEMENT", "INCOME_STATEMENT_COMPARISON",
        }
