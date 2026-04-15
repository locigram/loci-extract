from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import (
    DocumentMetadata,
    ExtractionMethod,
    ExtractionPayload,
    ExtractionWarning,
    TextSegment,
)


class PlainTextExtractor(BaseExtractor):
    name = "plaintext"

    def supports(self, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return mime_type.startswith("text/") or lower.endswith((".txt", ".md", ".csv", ".json"))

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
        ocr_backend: str = "auto",
    ) -> ExtractionPayload:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        document_id = str(uuid4())
        cleaned_text = text.strip()
        segments = [TextSegment(type="section", index=1, label="body", text=text)] if cleaned_text else []
        warnings: list[ExtractionWarning] = []
        status = "success"
        if not segments:
            status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="text_no_content_detected",
                    message="The text file was readable but contained no non-whitespace content.",
                )
            )
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="text"),
            extraction=ExtractionMethod(extractor=self.name, status=status, warnings=warnings),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
