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
