from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas import ExtractionPayload


class BaseExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def supports(self, filename: str, mime_type: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
        ocr_backend: str = "auto",
    ) -> ExtractionPayload:
        raise NotImplementedError
