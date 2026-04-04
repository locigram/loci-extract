from __future__ import annotations

from typing import Protocol

from app.schemas import ClassificationResult


class DocumentClassifier(Protocol):
    def __call__(
        self,
        *,
        filename: str,
        mime_type: str,
        raw_text: str,
        doc_type_hint: str | None = None,
    ) -> ClassificationResult: ...
