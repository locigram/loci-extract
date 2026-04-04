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
