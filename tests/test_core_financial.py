"""End-to-end financial extraction tests with a stubbed LLM client.

One test per financial doc type. Each uses a trivial PDF generated with
reportlab, a stubbed `call_llm_raw` that returns canned JSON, and asserts
the full pipeline produces the right schema + metadata enrichment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loci_extract import ExtractionOptions, core, extract_document
from loci_extract import core_chunked as cc
from loci_extract.formatters import format_extraction


def _make_pdf(tmp_path: Path, name: str, lines: list[str]) -> Path:
    """Build a minimal PDF with the given text. Just enough to trigger the
    detector and reach the LLM call."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.pdfgen import canvas

    path = tmp_path / name
    c = canvas.Canvas(str(path))
    c.setFont("Helvetica", 10)
    y = 720
    for line in lines:
        c.drawString(50, y, line)
        y -= 16
    c.save()
    return path


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    """Auto-apply: stub out LLM client + page-text gathering so tests run
    offline. Individual tests override ``core_chunked.call_llm_raw`` with
    their per-doc canned JSON via ``_canned_llm()``. The detector text-sample
    is set by each test via ``_set_page_text()``.
    """

    class _StubClient:
        class _chat:
            class _completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("Use call_llm_raw patch in tests; no live LLM")

            completions = _completions()
        chat = _chat()

    monkeypatch.setattr(core, "make_client", lambda url, api_key="local": _StubClient())
    return monkeypatch


def _set_page_text(monkeypatch, text: str):
    """Short-circuit core._gather_pages so reportlab PDFs that pdfminer
    can't easily parse still exercise the financial branch. The text drives
    detector keyword matching (BALANCE SHEET, GENERAL LEDGER, etc.)."""
    monkeypatch.setattr(core, "_gather_pages",
                         lambda pdf_path, opts, client, progress: {1: text})


def _canned_llm(monkeypatch, canned_json: str):
    """Replace `core_chunked.call_llm_raw` with a function that returns canned JSON."""

    def fake(**kwargs):
        return {
            "raw": canned_json,
            "prompt_tokens": 100,
            "completion_tokens": 500,
            "finish_reason": "stop",
            "llm_calls": 1,
            "llm_retries": 0,
        }

    monkeypatch.setattr(cc, "call_llm_raw", fake)


def _opts() -> ExtractionOptions:
    return ExtractionOptions(model_url="http://stub", model_name="stub", retry=0)


# ---------------------------------------------------------------------------
# BALANCE_SHEET
# ---------------------------------------------------------------------------


def test_balance_sheet_extracts_and_verifies(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "bs.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Balance Sheet - Acme HOA\n"
        "As of: 01/31/2025\n"
        "ASSETS\nTotal Assets 1,040,135.12\n"
        "Current Assets\nAccounts Receivable\n"
        "LIABILITIES\nLiabilities & Equity\nTotal Liabilities 42,311.84\n"
        "EQUITY\nTotal Equity 997,823.28"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Acme HOA", "period_end": "2025-01-31",
                   "accounting_basis": "Cash", "software": "AppFolio"},
        "assets": {
            "sections": [{
                "section_name": "Cash",
                "accounts": [
                    {"account_number": "1018-0000", "account_name": "Operating", "balance": 158678.65},
                    {"account_number": "1021-0000", "account_name": "Reserve", "balance": 328402.40},
                ],
                "section_total": 487081.05,
            }],
            "total_assets": 1040135.12,
        },
        "liabilities": {
            "sections": [{
                "section_name": "Current",
                "accounts": [
                    {"account_number": "2025-0000", "account_name": "Prepaid Assessments",
                     "balance": 42311.84},
                ],
                "section_total": 42311.84,
            }],
            "total_liabilities": 42311.84,
        },
        "equity": {
            "sections": [{
                "section_name": "Capital",
                "accounts": [
                    {"account_number": "3998-0000", "account_name": "Beginning Equity",
                     "balance": -252184.23},
                ],
                "section_total": -252184.23,
            }],
            "total_equity_reported": 997823.28,
        },
        "total_liabilities_and_equity_reported": 1040135.12,
        "metadata": {"notes": ["clean extraction"]},
    }))
    ext = extract_document(pdf, _opts())
    assert len(ext.documents) == 1
    doc = ext.documents[0]
    assert doc.document_type == "BALANCE_SHEET"
    # Totals verifier ran; canned data actually balances
    # (L+E = 42,311.84 + 997,823.28 = 1,040,135.12 = total_assets)
    assert doc.metadata.totals_verified is True
    assert doc.metadata.balance_sheet_balanced is True
    # Derived retained_earnings_calculated present
    assert "retained_earnings_calculated" in doc.data["equity"]
    # Note from LLM preserved
    assert "clean extraction" in doc.metadata.notes


# ---------------------------------------------------------------------------
# INCOME_STATEMENT
# ---------------------------------------------------------------------------


def test_income_statement_with_derived_net_income(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "is.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Income Statement - Test Entity\n"
        "For the period ending 01/31/2025\n"
        "Total Income 50,000.00\nTotal Expenses 35,000.00\n"
        "Net income\nOperating income"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Test Entity", "period_start": "2025-01-01",
                   "period_end": "2025-01-31", "software": "AppFolio"},
        "income": {"sections": [], "total": 50000.00},
        "expenses": {"sections": [], "total": 35000.00},
        "operating_income_reported": 15000.00,
        "net_income_reported": 15000.00,
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "INCOME_STATEMENT"
    assert doc.data.get("net_income_calculated") == 15000.00
    assert doc.metadata.llm_calls == 1


# ---------------------------------------------------------------------------
# INCOME_STATEMENT_COMPARISON (multi-column)
# ---------------------------------------------------------------------------


def test_income_statement_comparison_multicolumn(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "isc.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Income Statement 12-Month Comparison\n"
        "YTD Actual  YTD Budget  $ Variance  Annual Budget\n"
        "Current month actual  Prior year actual"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "PMG", "software": "AppFolio"},
        "columns": [
            {"key": "ytd_actual", "label": "YTD Actual", "column_type": "actual"},
            {"key": "ytd_budget", "label": "YTD Budget", "column_type": "budget"},
        ],
        "line_items": [
            {"account_number": "4000-0000", "account_name": "Assessment",
             "section": "Income", "row_type": "account",
             "values": {"ytd_actual": 712116.00, "ytd_budget": 714000.00}},
        ],
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "INCOME_STATEMENT_COMPARISON"
    assert len(doc.data["line_items"]) == 1
    assert doc.data["line_items"][0]["values"]["ytd_actual"] == 712116.00


# ---------------------------------------------------------------------------
# TRIAL_BALANCE
# ---------------------------------------------------------------------------


def test_trial_balance(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "tb.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Trial Balance\nTotal Debits 245,000.00\nTotal Credits 245,000.00\nDebit Credit"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Acme Corp", "period_end": "2025-01-31",
                   "software": "QuickBooks Desktop"},
        "accounts": [
            {"account_number": "1000", "account_name": "Checking",
             "debit": 45230.00, "credit": 0.00},
            {"account_number": "2000", "account_name": "Accounts Payable",
             "debit": 0.00, "credit": 12400.00},
        ],
        "total_debits": 45230.00,
        "total_credits": 12400.00,
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "TRIAL_BALANCE"
    assert len(doc.data["accounts"]) == 2


# ---------------------------------------------------------------------------
# RESERVE_ALLOCATION
# ---------------------------------------------------------------------------


def test_reserve_allocation(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "ra.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Reserve Allocation\nReserve fund\nComponent balance\n"
        "Tile/Shake roof\nAsphalt replacement\nContingency"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Niguel Villas HOA", "software": "AppFolio"},
        "components": [
            {"account_number": "3015-0000", "component_name": "Tile/Shake Roof",
             "current_balance": 694334.24},
            {"account_number": "3011-0000", "component_name": "Painting",
             "current_balance": -147545.64},
        ],
        "bank_accounts": [
            {"account_number": "1021-0000", "account_name": "Reserve Bank",
             "balance": 328402.40},
        ],
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "RESERVE_ALLOCATION"
    assert doc.data["total_reserve_balance_calculated"] == 546788.60
    assert doc.data["total_bank_balance_calculated"] == 328402.40


# ---------------------------------------------------------------------------
# GENERAL_LEDGER
# ---------------------------------------------------------------------------


def test_general_ledger(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "gl.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "General Ledger\nFrom 01/01/2025 To 01/31/2025\n"
        "Date Type Number Memo Debit Credit Balance\nTransaction detail"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Acme Corp", "period_start": "2025-01-01",
                   "period_end": "2025-01-31", "software": "QuickBooks Desktop"},
        "accounts": [{
            "account_number": "1000",
            "account_name": "Checking",
            "beginning_balance": 32000.00,
            "transactions": [
                {"date": "2025-01-05", "type": "Check", "number": "1042",
                 "name": "Office Depot", "debit": 245.00, "credit": None,
                 "balance": 31755.00},
                {"date": "2025-01-10", "type": "Deposit", "number": None,
                 "name": "Customer ABC", "debit": None, "credit": 5000.00,
                 "balance": 36755.00},
            ],
            "ending_balance": 36755.00,
        }],
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "GENERAL_LEDGER"
    assert len(doc.data["accounts"][0]["transactions"]) == 2


# ---------------------------------------------------------------------------
# AR_AGING
# ---------------------------------------------------------------------------


def test_ar_aging(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "ar.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Accounts Receivable Aging\n"
        "Customer current 1-30 31-60 61-90 over 90"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Acme Corp", "software": "QuickBooks Desktop"},
        "report_type": "AR",
        "as_of": "2025-01-31",
        "rows": [
            {"name": "Client A", "current": 5000, "days_31_60": 1200, "total": 6200},
        ],
        "totals": {"name": "TOTAL", "current": 45000, "days_31_60": 3100, "total": 57600},
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    doc = ext.documents[0]
    assert doc.document_type == "ACCOUNTS_RECEIVABLE_AGING"
    # Validate via the per-type pydantic model
    validated = doc.validated_data()
    assert validated.rows[0].name == "Client A"
    assert validated.totals.total == 57600


# ---------------------------------------------------------------------------
# CSV output shapes
# ---------------------------------------------------------------------------


def test_csv_shape_a_for_balance_sheet(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "bs2.pdf", ["stub"])
    _set_page_text(monkeypatch, (
        "Balance Sheet\nTotal Assets 1,000\nCurrent Assets\nAccounts Receivable"
    ))
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "Simple Co", "period_end": "2025-01-31",
                   "software": "AppFolio"},
        "assets": {
            "sections": [{
                "section_name": "Cash",
                "accounts": [
                    {"account_number": "1000", "account_name": "Operating", "balance": 500},
                    {"account_number": "1001", "account_name": "Savings",   "balance": 500},
                ],
                "section_total": 1000,
            }],
            "total_assets": 1000,
        },
        "liabilities": {"sections": [], "total_liabilities": 0},
        "equity": {"sections": [], "total_equity_reported": 1000},
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    out = format_extraction(ext, "csv")
    lines = out.strip().split("\n")
    # Header is Shape A
    assert lines[0].startswith("document_type,entity_name,software,")
    assert "balance" in lines[0]
    # Two account rows + one subtotal + one total
    assert "Operating" in out
    assert "Savings" in out
    # Subtotal rendered with "Total Cash"
    assert "Total Cash" in out


def test_parallel_chunks_calls_all_chunks(monkeypatch, tmp_path):
    """When a doc is chunked, max_parallel > 1 must still dispatch every chunk
    and the final Extraction must aggregate all partials."""
    pdf = _make_pdf(tmp_path, "gl.pdf", ["stub"])
    # Force GL detection so the chunker kicks in with account-boundary split
    _set_page_text(monkeypatch, (
        "General Ledger\nFrom 01/01/2025 To 01/31/2025\n"
        "Date Type Number Memo Debit Credit Balance\nTransaction detail\n\n"
        + "\n\n".join(f"{i:04d}-0000 Account {i}\n" + ("Lorem ipsum dolor " * 30)
                        for i in range(6))
    ))

    call_counter = {"n": 0}

    def slow_fake_llm(**kw):
        call_counter["n"] += 1
        return {
            "raw": json.dumps({
                "entity": {"name": "Test Entity", "software": "QuickBooks Desktop"},
                "accounts": [{"account_name": f"Account chunk {call_counter['n']}",
                              "beginning_balance": 0.0, "ending_balance": 0.0,
                              "transactions": []}],
                "metadata": {"notes": []},
            }),
            "prompt_tokens": 100, "completion_tokens": 500,
            "finish_reason": "stop", "llm_calls": 1, "llm_retries": 0,
        }
    monkeypatch.setattr(cc, "call_llm_raw", slow_fake_llm)

    opts = ExtractionOptions(
        model_url="http://stub", model_name="stub", retry=0,
        chunk_size_tokens=150,  # force chunking
        max_parallel=4,
    )
    ext = extract_document(pdf, opts)
    # Every chunk produced a partial; merged GL has >= 1 account
    assert call_counter["n"] >= 2, f"expected multiple chunks, got {call_counter['n']}"
    doc = ext.documents[0]
    assert doc.document_type == "GENERAL_LEDGER"
    # Account list merged from N partials
    assert len(doc.data["accounts"]) == call_counter["n"]


def test_lacerte_raises_for_financial(monkeypatch, tmp_path):
    pdf = _make_pdf(tmp_path, "bs3.pdf", ["stub"])
    _set_page_text(monkeypatch, "Balance Sheet\nTotal Assets 1\nCurrent Assets\nAccounts Receivable")
    _canned_llm(monkeypatch, json.dumps({
        "entity": {"name": "X"},
        "metadata": {"notes": []},
    }))
    ext = extract_document(pdf, _opts())
    with pytest.raises(NotImplementedError):
        format_extraction(ext, "lacerte")
    with pytest.raises(NotImplementedError):
        format_extraction(ext, "txf")
