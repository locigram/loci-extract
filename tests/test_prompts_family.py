"""prompts.py — DocumentFamily + DOCUMENT_FAMILY_MAP + get_prompt + canonical rules."""

from __future__ import annotations

from loci_extract.prompts import (
    DOCUMENT_FAMILY_MAP,
    SYSTEM_PROMPT,
    TAX_SYSTEM_PROMPT,
    DocumentFamily,
    get_prompt,
)
from loci_extract.schema import DATA_MODEL_BY_TYPE


def test_tax_system_prompt_byte_exact_alias():
    """SYSTEM_PROMPT must equal TAX_SYSTEM_PROMPT — preserves the existing
    tax extraction contract verbatim. Any divergence here will regress the
    golden-file test."""
    assert SYSTEM_PROMPT == TAX_SYSTEM_PROMPT


def test_get_prompt_w2_returns_tax_prompt():
    p = get_prompt("W2")
    assert "Deduplicate" in p
    assert "Box 12" in p


def test_get_prompt_balance_sheet_uses_simple_family():
    p = get_prompt("BALANCE_SHEET")
    assert "balance sheet" in p.lower() or "income statement" in p.lower()
    assert "section_total" in p
    # Inline section_total rule (no separate subtotals[] list)
    assert "subtotals" not in p.lower() or "do not create a separate" in p.lower()


def test_get_prompt_general_ledger_uses_transaction_family():
    p = get_prompt("GENERAL_LEDGER")
    assert "transaction" in p.lower()


def test_get_prompt_unknown_falls_back_to_simple():
    p = get_prompt("DEFINITELY_NOT_A_REAL_TYPE")
    # Falls back to FINANCIAL_SIMPLE
    assert "section_total" in p


def test_every_doc_type_has_family_mapping():
    for doc_type in DATA_MODEL_BY_TYPE:
        assert doc_type in DOCUMENT_FAMILY_MAP, f"{doc_type} missing from DOCUMENT_FAMILY_MAP"


def test_document_family_values_are_valid_enum():
    for doc_type, family in DOCUMENT_FAMILY_MAP.items():
        assert isinstance(family, DocumentFamily), f"{doc_type} maps to non-enum {family!r}"


def test_canonical_metadata_rule_in_financial_prompts():
    """Per DESIGN_DECISIONS, every financial family prompt must instruct
    the LLM to populate only metadata.notes[]."""
    for doc_type in ("BALANCE_SHEET", "INCOME_STATEMENT_COMPARISON", "GENERAL_LEDGER", "RESERVE_ALLOCATION"):
        p = get_prompt(doc_type)
        assert "METADATA" in p and "notes" in p, f"{doc_type} prompt missing METADATA/notes rule"


def test_canonical_section_totals_rule_in_financial_prompts():
    for doc_type in ("BALANCE_SHEET", "INCOME_STATEMENT_COMPARISON", "RESERVE_ALLOCATION"):
        p = get_prompt(doc_type)
        assert "SECTION TOTALS" in p, f"{doc_type} prompt missing SECTION TOTALS rule"
        assert "section_total" in p


def test_all_5_families_resolve():
    for fam in DocumentFamily:
        # Find any doc type mapped to this family
        matches = [dt for dt, f in DOCUMENT_FAMILY_MAP.items() if f == fam]
        assert matches, f"No doc type maps to {fam}"
