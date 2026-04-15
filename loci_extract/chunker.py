"""Text chunking for LLM input — splits long documents into manageable pieces.

3-tier strategy:
  1. Schema-boundary split (account sections for GL — blank line + account-like prefix)
  2. Per-page split on ``--- PAGE BREAK ---`` markers inserted by the extractor
  3. Fixed-size with 10% overlap as last resort

Per FINANCIAL_STATEMENTS_SPEC_V2.md §"GL Chunking — Robust Strategy"
and SPEC_PATCH_V3.md §Issue 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Rough chars-per-token estimate for financial text (mostly numbers + short words).
# Used to convert max_input_tokens → max_chars without a real tokenizer dep.
CHARS_PER_TOKEN = 3.5


@dataclass
class TextChunk:
    chunk_index: int        # 0-indexed position in the chunk list
    total_chunks: int       # how many chunks the text was split into
    text: str
    account_context: str | None  # account name/number this chunk starts with


def chunk_for_llm(
    text: str,
    document_type: str,
    max_input_tokens: int = 6000,
) -> list[TextChunk]:
    """Split text for LLM processing.

    Strategy by doc type:
      - GENERAL_LEDGER / QB_GENERAL_LEDGER: schema-boundary split (per account)
      - Other: per-page split, then fixed-size fallback

    Short text (<= max_chars) returns a single chunk.
    """
    max_chars = int(max_input_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return [TextChunk(0, 1, text, None)]

    if document_type in ("GENERAL_LEDGER", "QB_GENERAL_LEDGER"):
        chunks = _chunk_by_account_boundary(text, max_chars)
        if chunks:
            return chunks

    chunks = _chunk_by_page_break(text, max_chars)
    if chunks:
        return chunks

    return _chunk_fixed(text, max_chars, overlap=0.10)


def _chunk_by_account_boundary(text: str, max_chars: int) -> list[TextChunk]:
    """Split on detected account-section starts.

    Heuristic: a blank line followed by a line starting with an account-like
    prefix (4-12 alphanumeric/dash chars + space + capitalized name).
    Matches NNNN-NNNN, NNNN, and alphanumeric account number formats.
    Returns empty list if not enough boundaries to chunk meaningfully.
    """
    # Match a blank line followed by an account-like prefix:
    # "NNNN-NNNN SUNWEST BANK", "1000 Cash", or "ACC123 Petty Cash".
    # Allow dashes inside the prefix so QuickBooks-style "NNNN-NNNN" matches.
    section_starts = [m.start() for m in re.finditer(r"\n\n(?=[\w-]{4,15}\s+[A-Z])", text)]
    if len(section_starts) < 2:
        return []

    chunks: list[TextChunk] = []
    chunk_start = 0
    chunk_text = ""
    account_context: str | None = None

    for boundary in section_starts:
        segment = text[chunk_start:boundary]
        if len(chunk_text) + len(segment) > max_chars and chunk_text:
            chunks.append(TextChunk(
                chunk_index=len(chunks),
                total_chunks=0,
                text=chunk_text.strip(),
                account_context=account_context,
            ))
            chunk_text = segment
            chunk_start = boundary
            first_line = segment.strip().split("\n")[0] if segment.strip() else ""
            account_context = first_line[:80]
        else:
            chunk_text += segment
            if account_context is None:
                first_line = segment.strip().split("\n")[0] if segment.strip() else ""
                account_context = first_line[:80]

    remaining = text[chunk_start:]
    if remaining.strip():
        chunk_text += remaining
        chunks.append(TextChunk(
            chunk_index=len(chunks),
            total_chunks=0,
            text=chunk_text.strip(),
            account_context=account_context,
        ))

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    return chunks


def _chunk_by_page_break(text: str, max_chars: int) -> list[TextChunk]:
    """Split on ``--- PAGE BREAK ---`` markers, batching pages until max_chars."""
    pages = re.split(r"\n---\s*PAGE BREAK\s*---\n", text)
    if len(pages) < 2:
        return []

    chunks: list[TextChunk] = []
    current = ""
    for page in pages:
        if len(current) + len(page) > max_chars and current:
            chunks.append(TextChunk(len(chunks), 0, current.strip(), None))
            current = page
        else:
            current = current + "\n" + page if current else page
    if current.strip():
        chunks.append(TextChunk(len(chunks), 0, current.strip(), None))

    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    return chunks


def _chunk_fixed(text: str, max_chars: int, overlap: float) -> list[TextChunk]:
    """Fixed-size chunks with overlap. Last resort when no semantic boundaries exist."""
    step = max(1, int(max_chars * (1 - overlap)))
    raw_chunks = [text[i:i + max_chars] for i in range(0, len(text), step)]
    total = len(raw_chunks)
    return [TextChunk(i, total, chunk, None) for i, chunk in enumerate(raw_chunks)]


__all__ = ["TextChunk", "CHARS_PER_TOKEN", "chunk_for_llm"]
