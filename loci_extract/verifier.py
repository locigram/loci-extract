"""Python-side totals verification + derived-field computation.

Per DESIGN_DECISIONS.md (canonical _get_sections walker) and
FINANCIAL_STATEMENTS_SPEC_V2.md §"Totals Verification in Python".

The LLM extracts accounts and labeled subtotal rows verbatim. This module:
  - Verifies sum(section.accounts) ≈ section.section_total (within $0.02)
  - Verifies the balance-sheet equation (assets == liabilities + equity)
  - Computes derived fields (retained_earnings_calculated, etc.)

Operates on the dict returned by the LLM, BEFORE pydantic validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

ROUNDING_TOLERANCE = Decimal("0.02")


@dataclass
class VerificationResult:
    verified: bool
    mismatches: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    balance_sheet_balanced: bool | None = None


def _to_decimal(v) -> Decimal:
    if v is None:
        return Decimal(0)
    try:
        return Decimal(str(v))
    except Exception:
        return Decimal(0)


def _account_value(account: dict) -> Decimal:
    """Read whichever of `balance` / `amount` is populated."""
    if not isinstance(account, dict):
        return Decimal(0)
    val = account.get("balance")
    if val is None:
        val = account.get("amount")
    return _to_decimal(val)


def _get_sections(extracted: dict) -> list[dict]:
    """Canonical recursive walker per DESIGN_DECISIONS Decision 1.

    Returns every dict that has both an ``accounts`` list and a
    ``section_total`` field, drilling through ``sections``/``subsections``
    children and the top-level financial sides
    (assets/liabilities/equity/income/expenses/...).
    """
    sections: list[dict] = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        if "accounts" in node and "section_total" in node:
            sections.append(node)
        for key in ("sections", "subsections"):
            for child in node.get(key, []) or []:
                _walk(child)
        for key in (
            "assets", "liabilities", "equity",
            "income", "expenses", "other_income", "other_expenses",
            "current_assets", "reserve_accounts",
            "fixed_assets", "current_liabilities",
            "long_term_liabilities",
        ):
            if key in node:
                _walk(node[key])

    _walk(extracted)
    return sections


def _sum_accounts(accounts) -> Decimal:
    if not isinstance(accounts, list):
        return Decimal(0)
    return sum((_account_value(a) for a in accounts), Decimal(0))


def verify_section_totals(extracted: dict) -> VerificationResult:
    """Compare each section's reported total against sum(accounts).

    Also runs top-level total checks (total_assets, total_liabilities, etc.)
    and the balance-sheet equation when applicable.
    """
    mismatches: list[dict] = []
    notes: list[str] = []

    for section in _get_sections(extracted):
        accounts = section.get("accounts", [])
        reported = section.get("section_total")
        if reported is None:
            continue
        computed = _sum_accounts(accounts)
        delta = abs(computed - _to_decimal(reported))
        if delta > ROUNDING_TOLERANCE:
            mismatches.append({
                "section": section.get("section_name", "unknown"),
                "computed": float(computed),
                "reported": float(reported),
                "delta": float(delta),
            })

    # Balance sheet equation: total_assets == total_liabilities + total_equity
    bs_balanced: bool | None = None
    assets_total = _find_total(extracted, "assets", "total_assets")
    liab_total = _find_total(extracted, "liabilities", "total_liabilities")
    equity_total = _find_total(extracted, "equity", "total_equity_reported")
    if equity_total is None:
        equity_total = _find_total(extracted, "equity", "total_equity")
    if all(v is not None for v in (assets_total, liab_total, equity_total)):
        lhs = _to_decimal(assets_total)
        rhs = _to_decimal(liab_total) + _to_decimal(equity_total)
        delta = abs(lhs - rhs)
        bs_balanced = delta <= ROUNDING_TOLERANCE
        if not bs_balanced:
            mismatches.append({
                "section": "balance_sheet_equation",
                "computed": float(rhs),
                "reported": float(lhs),
                "delta": float(delta),
            })
            notes.append(
                f"Balance sheet does not balance: Assets={float(lhs):.2f}, "
                f"Liabilities+Equity={float(rhs):.2f}, Delta={float(delta):.2f}"
            )

    return VerificationResult(
        verified=len(mismatches) == 0,
        mismatches=mismatches,
        notes=notes,
        balance_sheet_balanced=bs_balanced,
    )


def _find_total(extracted: dict, side_key: str, total_key: str):
    """Look for a top-level total either at the root or one level into the side dict."""
    if total_key in extracted:
        return extracted[total_key]
    side = extracted.get(side_key)
    if isinstance(side, dict) and total_key in side:
        return side[total_key]
    return None


def compute_derived_fields(extracted: dict, document_type: str) -> dict:
    """Compute fields that must NOT come from the LLM (DESIGN_DECISIONS).

    **Mutates ``extracted`` in-place** to set nested per-doc-type derived
    fields (e.g., ``extracted["equity"]["retained_earnings_calculated"]``).
    Also returns a flat dict of top-level derived fields suitable for
    ``merged.update(...)``.

    Per-doc-type:
      - BALANCE_SHEET: equity.retained_earnings_calculated (nested)
      - INCOME_STATEMENT / INCOME_STATEMENT_COMPARISON: net_income_calculated (top-level)
      - RESERVE_ALLOCATION: total_reserve_balance_calculated,
        total_bank_balance_calculated (top-level)
    """
    derived: dict = {}

    if document_type == "BALANCE_SHEET":
        equity = extracted.setdefault("equity", {})
        explicit_sections = equity.get("sections", []) or []
        explicit_sum = sum(
            (_account_value(a) for s in explicit_sections for a in s.get("accounts", []) or []),
            Decimal(0),
        )
        total_equity = _to_decimal(
            equity.get("total_equity_reported") or equity.get("total_equity") or 0
        )
        equity["retained_earnings_calculated"] = float(total_equity - explicit_sum)

    if document_type in ("INCOME_STATEMENT", "INCOME_STATEMENT_COMPARISON"):
        income_total = _find_total(extracted, "income", "total")
        expense_total = _find_total(extracted, "expenses", "total")
        if income_total is None:
            income_total = extracted.get("total_income")
        if expense_total is None:
            expense_total = extracted.get("total_expenses")
        if income_total is not None and expense_total is not None:
            derived["net_income_calculated"] = float(
                _to_decimal(income_total) - _to_decimal(expense_total)
            )

    if document_type == "RESERVE_ALLOCATION":
        components = extracted.get("components", []) or []
        total_balance = sum(
            (_to_decimal((c or {}).get("current_balance")) for c in components),
            Decimal(0),
        )
        derived["total_reserve_balance_calculated"] = float(total_balance)
        bank_accounts = extracted.get("bank_accounts", []) or []
        total_bank = sum(
            (_account_value(a) for a in bank_accounts),
            Decimal(0),
        )
        derived["total_bank_balance_calculated"] = float(total_bank)

    return derived


__all__ = [
    "ROUNDING_TOLERANCE",
    "VerificationResult",
    "verify_section_totals",
    "compute_derived_fields",
]
