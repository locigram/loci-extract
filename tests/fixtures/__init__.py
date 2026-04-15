"""Fixture registry + pytest skip helper for real-PDF integration tests.

Real fixtures live in ``tests/fixtures/*.pdf`` but are guarded with
``skip_if_no_fixture()`` so CI / contributors without the PDFs see clean
skips rather than collection errors. When the fixture isn't on disk the
test is marked skipped with a clear reason and the acquisition note.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent

FIXTURE_REGISTRY: dict[str, Path] = {
    # Tax
    "sample_w2": FIXTURES_DIR / "sample_w2.pdf",
    # AppFolio (PScript5/Distiller — encoding-broken)
    "appfolio_jan25_owner": FIXTURES_DIR / "01 - Jan 2025 Financials Owner.pdf",
    # AppFolio (MacRoman — clean text layer, second export path)
    "appfolio_income_statement": FIXTURES_DIR / "appfolio income statment.pdf",
    # QuickBooks Desktop XLSX exports
    "qb_balance_sheet": FIXTURES_DIR / "A2. Unadjusted BS 2025.xlsx",
    "qb_profit_loss": FIXTURES_DIR / "A1. Unadjusted P&L 2025.xlsx",
    "qb_general_ledger": FIXTURES_DIR / "SAMPLE GL.xlsx",
}


def fixture_path(name: str) -> Path | None:
    path = FIXTURE_REGISTRY.get(name)
    if path is None or not path.exists():
        return None
    return path


def skip_if_no_fixture(name: str):
    """pytest decorator — skips the test when the named fixture isn't on disk."""
    path = FIXTURE_REGISTRY.get(name)
    reason = (
        f"Fixture {name!r} not available at {path}. "
        f"Commit it to tests/fixtures/ or see tests/fixtures/README.md "
        f"for acquisition instructions."
    )
    return pytest.mark.skipif(
        path is None or not path.exists(),
        reason=reason,
    )


__all__ = ["FIXTURE_REGISTRY", "fixture_path", "skip_if_no_fixture"]
