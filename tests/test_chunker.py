"""chunker.py — 3-tier text chunking."""

from __future__ import annotations

from loci_extract.chunker import CHARS_PER_TOKEN, TextChunk, chunk_for_llm


def test_short_text_single_chunk():
    text = "Hello world, this is short text."
    chunks = chunk_for_llm(text, "BALANCE_SHEET", max_input_tokens=6000)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].text == text


def test_page_break_split():
    # Two long pages separated by page break marker. With small token budget,
    # they should split into 2 chunks.
    page = "X" * 5000
    text = f"{page}\n--- PAGE BREAK ---\n{page}"
    chunks = chunk_for_llm(text, "BALANCE_SHEET", max_input_tokens=2000)
    assert len(chunks) >= 2
    assert all(c.total_chunks == len(chunks) for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_gl_account_boundary_split():
    # Construct text with multiple account-section headers separated by blanks.
    accounts = "\n\n".join(
        f"{i:04d}-0000 SUNWEST BANK ACCT\n" + ("Transaction line " * 30)
        for i in range(10)
    )
    chunks = chunk_for_llm(accounts, "GENERAL_LEDGER", max_input_tokens=500)
    assert len(chunks) > 1
    # Account context should be populated for at least some chunks
    assert any(c.account_context for c in chunks)


def test_fixed_size_fallback():
    # No page breaks, no account boundaries → falls through to fixed-size with overlap.
    text = "lorem ipsum " * 5000
    chunks = chunk_for_llm(text, "BALANCE_SHEET", max_input_tokens=1000)
    assert len(chunks) >= 2
    assert all(c.total_chunks == len(chunks) for c in chunks)
    # All chunks have a non-empty text
    assert all(c.text.strip() for c in chunks)


def test_total_chunks_consistent():
    text = ("a" * 1000 + "\n--- PAGE BREAK ---\n") * 5
    chunks = chunk_for_llm(text, "BALANCE_SHEET", max_input_tokens=500)
    total = len(chunks)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.total_chunks == total


def test_text_chunk_dataclass():
    c = TextChunk(0, 1, "hello", None)
    assert c.chunk_index == 0
    assert c.account_context is None


def test_chars_per_token_constant():
    assert CHARS_PER_TOKEN > 0
