from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, TextSegment


class PlainTextExtractor(BaseExtractor):
    name = "plaintext"

    def supports(self, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return mime_type.startswith("text/") or lower.endswith((".txt", ".md", ".csv", ".json"))

    def extract(self, file_path: Path, filename: str, mime_type: str) -> ExtractionPayload:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        document_id = str(uuid4())
        segments = [TextSegment(type="section", index=1, label="body", text=text)]
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="text"),
            extraction=ExtractionMethod(extractor=self.name),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
