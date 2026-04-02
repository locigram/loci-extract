from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fitz

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, ExtractionWarning, TextSegment


class PdfExtractor(BaseExtractor):
    name = "pymupdf"

    _OCR_STRATEGY_MESSAGES = {
        "never": "OCR was disabled for this request (ocr_strategy=never).",
        "auto": "OCR fallback was requested automatically, but no OCR backend is configured yet.",
        "always": "OCR was explicitly requested, but no OCR backend is configured yet.",
    }

    def supports(self, filename: str, mime_type: str) -> bool:
        return filename.lower().endswith(".pdf") or mime_type == "application/pdf"

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
    ) -> ExtractionPayload:
        document = fitz.open(file_path)
        document_id = str(uuid4())
        pages: list[str] = []
        segments: list[TextSegment] = []

        for idx, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(text)
                segments.append(TextSegment(type="page", index=idx, label=f"page-{idx}", text=text))

        warnings: list[ExtractionWarning] = []
        extraction_status = "success"
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_available": False,
            "ocr_backend": None,
        }
        if not pages:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="pdf_no_text_layer",
                    message="No extractable PDF text layer was found in this PDF.",
                )
            )
            warnings.append(
                ExtractionWarning(
                    code="ocr_not_available",
                    message=self._OCR_STRATEGY_MESSAGES.get(
                        ocr_strategy,
                        "OCR fallback was requested, but no OCR backend is configured yet.",
                    ),
                )
            )

        raw_text = "\n\n".join(pages)
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(
                filename=filename,
                mime_type=mime_type,
                source_type="pdf",
                page_count=len(document),
            ),
            extraction=ExtractionMethod(
                extractor=self.name,
                status=extraction_status,
                warnings=warnings,
            ),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )
