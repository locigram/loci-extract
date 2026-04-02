from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytesseract
from PIL import Image

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, TextSegment


class ImageOcrExtractor(BaseExtractor):
    name = "tesseract"

    def supports(self, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")) or mime_type.startswith(
            "image/"
        )

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
    ) -> ExtractionPayload:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image).strip()
        document_id = str(uuid4())
        segments = [TextSegment(type="page", index=1, label="image-1", text=text)] if text else []
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
            extraction=ExtractionMethod(extractor=self.name, ocr_used=True),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
