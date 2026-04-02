from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytesseract
from PIL import Image

from app.capabilities import tesseract_available
from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import (
    DocumentMetadata,
    ExtractionMethod,
    ExtractionPayload,
    ExtractionWarning,
    TextSegment,
)


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
        document_id = str(uuid4())
        warnings: list[ExtractionWarning] = []
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_available": tesseract_available(),
            "ocr_backend": "tesseract" if tesseract_available() else None,
        }

        if ocr_strategy == "never":
            warnings.append(
                ExtractionWarning(
                    code="ocr_disabled",
                    message="OCR was disabled for this request (ocr_strategy=never).",
                )
            )
            return ExtractionPayload(
                document_id=document_id,
                metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
                extraction=ExtractionMethod(
                    extractor=self.name,
                    status="partial",
                    warnings=warnings,
                ),
                raw_text="",
                segments=[],
                chunks=[],
                extra=extra,
            )

        if not tesseract_available():
            warnings.append(
                ExtractionWarning(
                    code="tesseract_not_available",
                    message="Image OCR requires the tesseract binary, which is not currently installed.",
                )
            )
            return ExtractionPayload(
                document_id=document_id,
                metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
                extraction=ExtractionMethod(
                    extractor=self.name,
                    status="partial",
                    warnings=warnings,
                ),
                raw_text="",
                segments=[],
                chunks=[],
                extra=extra,
            )

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image).strip()
        segments = [TextSegment(type="page", index=1, label="image-1", text=text)] if text else []
        status = "success" if text else "partial"
        if not text:
            warnings.append(
                ExtractionWarning(
                    code="ocr_no_text_detected",
                    message="OCR completed but no text was detected in the image.",
                )
            )
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
            extraction=ExtractionMethod(
                extractor=self.name,
                ocr_used=True,
                status=status,
                warnings=warnings,
            ),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )
