from __future__ import annotations

from typing import Protocol

from app.schemas import ExtractionPayload, StructuredDocument


class StructuredExtractor(Protocol):
    def __call__(self, raw_payload: ExtractionPayload, *, mask_pii: bool = True) -> StructuredDocument: ...
