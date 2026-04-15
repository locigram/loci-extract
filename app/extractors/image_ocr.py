from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.capabilities import tesseract_available
from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.ocr import extract_best_ocr_result
from app.ocr_backends import OcrBackendNotAvailableError, get_backend
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
        ocr_backend: str = "auto",
    ) -> ExtractionPayload:
        document_id = str(uuid4())
        warnings: list[ExtractionWarning] = []

        # VLM strategy: send image directly to VLM, skip OCR entirely
        if ocr_strategy == "vlm":
            return self._extract_with_vlm(file_path, filename, mime_type, document_id)

        ocr_is_available = tesseract_available()
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_backend_requested": ocr_backend,
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

        # For non-default backends, try to resolve; for auto/tesseract, use local availability check
        if ocr_backend not in ("auto", "tesseract"):
            try:
                resolved_backend = get_backend(ocr_backend)
                if not resolved_backend.is_available():
                    raise OcrBackendNotAvailableError(f"{ocr_backend} not available")
            except OcrBackendNotAvailableError:
                extra["ocr_backend"] = None
                extra["ocr_available"] = False
                warnings.append(
                    ExtractionWarning(
                        code="paddleocr_not_available",
                        message=f"OCR backend '{ocr_backend}' is not available.",
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
            extra["ocr_backend"] = resolved_backend.name
            extra["ocr_available"] = True
        else:
            # auto/tesseract: use existing local availability check (preserves monkeypatch targets)
            resolved_backend = None
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
        if resolved_backend is not None:
            ocr_result = extract_best_ocr_result(image, backend=resolved_backend)
        else:
            ocr_result = extract_best_ocr_result(image)
        extra["ocr_backend"] = ocr_result.get("backend", extra.get("ocr_backend"))
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

    def _extract_with_vlm(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        document_id: str,
    ) -> ExtractionPayload:
        """Send the image directly to a VLM for extraction. No OCR binary used."""
        from app.llm.config import get_vlm_client

        warnings: list[ExtractionWarning] = []
        client = get_vlm_client()

        if client is None:
            warnings.append(
                ExtractionWarning(
                    code="vlm_not_available",
                    message="VLM extraction requested but no VLM endpoint is configured.",
                )
            )
            return ExtractionPayload(
                document_id=document_id,
                metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
                extraction=ExtractionMethod(extractor=self.name, status="partial", warnings=warnings),
                raw_text="",
                segments=[],
                chunks=[],
                extra={"ocr_strategy": "vlm", "result_source": "none", "vlm_used": True},
            )

        image = Image.open(file_path)

        # Pre-process: resize large images to max 1568px on longest side
        max_dim = 1568
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        from app.extractors.vlm import vlm_extract_page
        result = vlm_extract_page(client, image, doc_type="unknown")

        text = str(result.get("raw_text", "")).strip() if result else ""
        status = "success" if text else "partial"
        if not text:
            warnings.append(
                ExtractionWarning(code="vlm_no_text_detected", message="VLM processed the image but extracted no text.")
            )

        segments = [
            TextSegment(
                type="page", index=1, label="image-1", text=text,
                metadata={"source": "vlm", "page_number": 1},
            )
        ] if text else []

        extra: dict[str, object] = {
            "ocr_strategy": "vlm",
            "ocr_backend": "vlm",
            "result_source": "vlm" if text else "none",
            "vlm_used": True,
            "vlm_model": client.model,
            "text_layer_ignored": True,
            "page_provenance": [{
                "page_number": 1,
                "source": "vlm" if text else "none",
                "has_text": bool(text),
                "text_length": len(text),
            }],
        }
        if result and result.get("fields"):
            extra["vlm_fields"] = result["fields"]

        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="image"),
            extraction=ExtractionMethod(extractor=self.name, ocr_used=False, status=status, warnings=warnings),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )
