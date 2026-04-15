from __future__ import annotations

import re
from typing import Iterable

from app.schemas import ExtractionPayload, StructuredDocument


def get_text_lines(raw_payload: ExtractionPayload) -> list[str]:
    lines = [line.strip() for line in raw_payload.raw_text.splitlines()]
    return [line for line in lines if line]


def search_patterns(text: str, patterns: Iterable[str], *, flags: int = re.IGNORECASE) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            return match
    return None


def snippet_around_match(text: str, patterns: Iterable[str], *, width: int = 80) -> str | None:
    match = search_patterns(text, patterns)
    if not match:
        return None
    start = max(match.start() - width, 0)
    end = min(match.end() + width, len(text))
    return " ".join(text[start:end].split())


def first_source_pages(raw_payload: ExtractionPayload, *, limit: int = 2) -> list[int]:
    pages: list[int] = []
    for segment in raw_payload.segments:
        page_number = segment.metadata.get("page_number")
        if isinstance(page_number, int) and page_number not in pages:
            pages.append(page_number)
        if len(pages) >= limit:
            return pages

    page_provenance = raw_payload.extra.get("page_provenance")
    if isinstance(page_provenance, list):
        for entry in page_provenance:
            if not isinstance(entry, dict):
                continue
            page_number = entry.get("page_number")
            if isinstance(page_number, int) and page_number not in pages:
                pages.append(page_number)
            if len(pages) >= limit:
                break
    return pages


_INSTRUCTION_SIGNALS = [
    "general instructions",
    "specific instructions",
    "instructions for employee",
    "instructions for recipient",
    "instructions for payer",
    "instructions for employer",
    "instructions for forms",
    "how to complete",
    "filing instructions",
    "department of the treasury",
    "privacy act and paperwork reduction act notice",
    "this information is being furnished to the internal revenue service",
    "see the instructions on the back of copy",
]

_COPY_LABEL_SIGNALS = [
    "copy a for social security administration",
    "copy b to be filed with employee",
    "copy c for employer",
    "copy d for employer",
    "copy 1 for state",
    "copy 2 to be filed with employee",
    "copy b for recipient",
    "copy a for internal revenue service",
    "for your records",
]


def _is_instruction_page(text: str) -> bool:
    """Detect if a page is instructions/copy labels rather than form data."""
    lowered = text.lower()
    instruction_hits = sum(1 for sig in _INSTRUCTION_SIGNALS if sig in lowered)
    if instruction_hits >= 2:
        return True
    # Check if the page is mostly a copy label (very short + copy signal)
    stripped = text.strip()
    if len(stripped) < 200:
        if any(sig in lowered for sig in _COPY_LABEL_SIGNALS):
            return True
    return False


def _is_form_data_page(text: str, form_signals: list[str] | None = None) -> bool:
    """Detect if a page contains actual form data (not just instructions)."""
    if not text or len(text.strip()) < 20:
        return False
    if _is_instruction_page(text):
        return False
    # If caller provides form-specific signals, check for them
    if form_signals:
        lowered = text.lower()
        return any(sig in lowered for sig in form_signals)
    return True


def get_form_pages(
    raw_payload: ExtractionPayload,
    *,
    form_signals: list[str] | None = None,
    max_form_pages: int | None = None,
) -> list[str]:
    """Extract text from form data pages only, skipping instruction/copy pages.

    For multi-form PDFs (e.g. 5 W-2s in one PDF), returns text from all form pages.
    For single-form PDFs with instructions, returns only the form page(s).

    Args:
        form_signals: keywords that indicate a page has actual form data
            (e.g. ["wage", "employer", "employee"] for W-2)
        max_form_pages: cap the number of form pages returned (None = no limit)
    """
    page_segments = [
        seg for seg in raw_payload.segments
        if seg.type == "page" and isinstance(seg.metadata.get("page_number"), int)
    ]

    if not page_segments:
        return [raw_payload.raw_text] if raw_payload.raw_text.strip() else []

    page_segments.sort(key=lambda s: s.metadata.get("page_number", 0))

    form_pages: list[str] = []
    for seg in page_segments:
        if _is_form_data_page(seg.text, form_signals=form_signals):
            form_pages.append(seg.text)
            if max_form_pages and len(form_pages) >= max_form_pages:
                break

    # Fallback: if no form pages detected, use first page
    if not form_pages and page_segments:
        form_pages = [page_segments[0].text]

    return form_pages


def get_form_text(
    raw_payload: ExtractionPayload,
    *,
    form_signals: list[str] | None = None,
    max_form_pages: int | None = None,
) -> str:
    """Get concatenated text from form pages only."""
    pages = get_form_pages(raw_payload, form_signals=form_signals, max_form_pages=max_form_pages)
    return "\n\n".join(pages)


def get_form_lines(
    raw_payload: ExtractionPayload,
    *,
    form_signals: list[str] | None = None,
    max_form_pages: int | None = None,
) -> list[str]:
    """Get text lines from form pages only."""
    text = get_form_text(raw_payload, form_signals=form_signals, max_form_pages=max_form_pages)
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def unknown_structured_document() -> StructuredDocument:
    return StructuredDocument(
        document_type="unknown",
        fields={},
        review={
            "requires_human_review": True,
            "review_reasons": ["unknown_document_type"],
            "missing_fields": [],
            "validation_errors": [],
        },
    )
