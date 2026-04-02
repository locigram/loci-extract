from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from docx import Document

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, TextSegment


class DocxExtractor(BaseExtractor):
    name = "python-docx"

    def supports(self, filename: str, mime_type: str) -> bool:
        return filename.lower().endswith(".docx") or mime_type == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
    ) -> ExtractionPayload:
        doc = Document(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        document_id = str(uuid4())
        segments = [TextSegment(type="paragraph", index=i + 1, label=None, text=p) for i, p in enumerate(paragraphs)]
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="docx"),
            extraction=ExtractionMethod(extractor=self.name),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
