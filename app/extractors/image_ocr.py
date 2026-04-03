from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytesseract
from PIL import Image, ImageOps

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

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, dict[str, object]]:
        original_mode = image.mode
        processed = ImageOps.exif_transpose(image)
        processed = processed.convert("L")
        processed = ImageOps.autocontrast(processed)
        processed = processed.point(lambda px: 255 if px > 180 else 0)
        processed = processed.resize((processed.width * 2, processed.height * 2))
        metadata = {
            "original_mode": original_mode,
            "processed_mode": processed.mode,
            "preprocessing": [
                "exif_transpose",
                "grayscale",
                "autocontrast",
                "threshold",
                "upscale_2x",
            ],
        }
        return processed, metadata

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
        ocr_is_available = tesseract_available()
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_available": ocr_is_available,
            "ocr_backend": "tesseract" if ocr_is_available else None,
            "ocr_attempted": False,
            "result_source": "none",
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

        if not ocr_is_available:
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
        processed_image, preprocessing_metadata = self._preprocess_image(image)
        extra.update(preprocessing_metadata)
        extra["ocr_attempted"] = True
        text = pytesseract.image_to_string(processed_image).strip()
        segments = [
            TextSegment(
                type="page",
                index=1,
                label="image-1",
                text=text,
                metadata={"source": "ocr", "page_number": 1},
            )
        ] if text else []
        status = "success" if text else "partial"
        if text:
            extra["result_source"] = "ocr"
        else:
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
