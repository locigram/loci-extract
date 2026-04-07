from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.capabilities import tesseract_available
from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.ocr import extract_best_ocr_result
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
        ocr_result = extract_best_ocr_result(image)
        extra.update(
            {
                "ocr_attempted": True,
                "selected_ocr_pass": ocr_result["selected_pass"],
                "selected_ocr_rotation": ocr_result.get("selected_rotation", 0),
                "ocr_score": ocr_result["score"],
                "processed_mode": ocr_result["processed_mode"],
                "preprocessing": ocr_result["preprocessing"],
                "ocr_passes": ocr_result["ocr_passes"],
            }
        )

        text = ocr_result["text"]
        page_provenance = [
            {
                "page_number": 1,
                "source": "ocr" if text else "none",
                "has_text": bool(text),
                "text_length": len(text),
                "ocr_score": ocr_result["score"],
                "selected_ocr_pass": ocr_result["selected_pass"],
                "selected_ocr_rotation": ocr_result.get("selected_rotation", 0),
            }
        ]
        extra["page_provenance"] = page_provenance

        segments = [
            TextSegment(
                type="page",
                index=1,
                label="image-1",
                text=text,
                metadata={
                    "source": "ocr",
                    "page_number": 1,
                    "ocr_score": ocr_result["score"],
                    "selected_ocr_pass": ocr_result["selected_pass"],
                    "selected_ocr_rotation": ocr_result.get("selected_rotation", 0),
                },
            )
        ] if text else []
        status = "success" if text else "partial"
        if text:
            extra["result_source"] = "ocr"
            if ocr_result["score"] < 10:
                warnings.append(
                    ExtractionWarning(
                        code="ocr_low_quality",
                        message="OCR detected text in the image, but the selected result scored as low quality.",
                    )
                )
                status = "partial"
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
