from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import fitz
import pytesseract
from PIL import Image

from app.capabilities import tesseract_available
from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, ExtractionWarning, TextSegment


DEFAULT_MAX_PDF_PAGES = 200


class PdfExtractor(BaseExtractor):
    name = "pymupdf"

    _OCR_STRATEGY_MESSAGES = {
        "never": "OCR was disabled for this request (ocr_strategy=never).",
        "auto": "OCR fallback was requested automatically, but no OCR backend is configured yet.",
        "always": "OCR was explicitly requested, but no OCR backend is configured yet.",
    }

    def supports(self, filename: str, mime_type: str) -> bool:
        return filename.lower().endswith(".pdf") or mime_type == "application/pdf"

    def _max_pdf_pages(self) -> int:
        raw = os.getenv("LOCI_EXTRACT_MAX_PDF_PAGES", str(DEFAULT_MAX_PDF_PAGES)).strip()
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_MAX_PDF_PAGES
        return value if value > 0 else DEFAULT_MAX_PDF_PAGES

    def _extract_pdf_text(
        self, document: fitz.Document, *, max_pages: int | None = None
    ) -> dict[int, str]:
        parsed_pages: dict[int, str] = {}
        for idx, page in enumerate(document, start=1):
            if max_pages is not None and idx > max_pages:
                break
            text = page.get_text("text").strip()
            if text:
                parsed_pages[idx] = text
        return parsed_pages

    def _extract_pdf_ocr(
        self, document: fitz.Document, *, max_pages: int | None = None
    ) -> dict[int, str]:
        ocr_pages: dict[int, str] = {}
        for idx, page in enumerate(document, start=1):
            if max_pages is not None and idx > max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image).strip()
            if text:
                ocr_pages[idx] = text
        return ocr_pages

    def _build_page_outputs(
        self,
        *,
        processed_page_count: int,
        parser_pages: dict[int, str],
        ocr_pages: dict[int, str] | None = None,
        prefer_ocr: bool = False,
        ocr_attempted: bool = False,
    ) -> tuple[list[str], list[TextSegment], list[dict[str, object]]]:
        pages: list[str] = []
        segments: list[TextSegment] = []
        provenance: list[dict[str, object]] = []
        ocr_pages = ocr_pages or {}

        for page_number in range(1, processed_page_count + 1):
            parser_text = parser_pages.get(page_number)
            ocr_text = ocr_pages.get(page_number)

            if prefer_ocr and ocr_text:
                source = "ocr"
                text = ocr_text
            elif prefer_ocr and parser_text and ocr_attempted:
                source = "parser_fallback"
                text = parser_text
            elif parser_text:
                source = "parser"
                text = parser_text
            elif ocr_text:
                source = "ocr"
                text = ocr_text
            else:
                source = "none"
                text = None

            provenance.append(
                {
                    "page_number": page_number,
                    "source": source,
                    "has_text": bool(text),
                    "text_length": len(text) if text else 0,
                }
            )

            if not text:
                continue

            pages.append(text)
            segments.append(
                TextSegment(
                    type="page",
                    index=page_number,
                    label=f"page-{page_number}",
                    text=text,
                    metadata={"source": source, "page_number": page_number},
                )
            )

        return pages, segments, provenance

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
        total_pages = len(document)
        max_pages = self._max_pdf_pages()
        processed_page_count = min(total_pages, max_pages)
        page_limit_applied = total_pages > max_pages
        parser_pages = self._extract_pdf_text(document, max_pages=max_pages)
        pages, segments, page_provenance = self._build_page_outputs(
            processed_page_count=processed_page_count,
            parser_pages=parser_pages,
        )

        warnings: list[ExtractionWarning] = []
        extraction_status = "success"
        ocr_used = False
        ocr_is_available = tesseract_available()
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_available": ocr_is_available,
            "ocr_backend": "tesseract" if ocr_is_available else None,
            "ocr_attempted": False,
            "result_source": "parser" if pages else "none",
            "page_limit_applied": page_limit_applied,
            "processed_page_count": processed_page_count,
            "max_pdf_pages": max_pages,
            "page_provenance": page_provenance,
        }

        if page_limit_applied:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="pdf_page_limit_applied",
                    message=(
                        f"PDF has {total_pages} pages, but only the first {max_pages} pages "
                        "were processed due to the configured page limit."
                    ),
                )
            )

        should_attempt_ocr = ocr_strategy == "always" or (ocr_strategy == "auto" and not pages)

        if not pages:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="pdf_no_text_layer",
                    message="No extractable PDF text layer was found in this PDF.",
                )
            )

        if ocr_strategy == "never" and not pages:
            warnings.append(
                ExtractionWarning(
                    code="ocr_disabled",
                    message=self._OCR_STRATEGY_MESSAGES["never"],
                )
            )
        elif should_attempt_ocr:
            if not ocr_is_available:
                warnings.append(
                    ExtractionWarning(
                        code="ocr_not_available",
                        message=self._OCR_STRATEGY_MESSAGES.get(
                            ocr_strategy,
                            "OCR fallback was requested, but no OCR backend is configured yet.",
                        ),
                    )
                )
            else:
                extra["ocr_attempted"] = True
                ocr_pages = self._extract_pdf_ocr(document, max_pages=max_pages)
                ocr_used = True
                pages, segments, page_provenance = self._build_page_outputs(
                    processed_page_count=processed_page_count,
                    parser_pages=parser_pages,
                    ocr_pages=ocr_pages,
                    prefer_ocr=ocr_strategy == "always" or not pages,
                    ocr_attempted=True,
                )
                extra["page_provenance"] = page_provenance

                if any(entry["source"] == "ocr" for entry in page_provenance):
                    extraction_status = "partial" if page_limit_applied else "success"
                    extra["result_source"] = "ocr"
                elif any(entry["source"] == "parser_fallback" for entry in page_provenance):
                    extraction_status = "partial" if page_limit_applied else "success"
                    extra["result_source"] = "parser_fallback"
                    warnings.append(
                        ExtractionWarning(
                            code="ocr_no_text_detected",
                            message="OCR was attempted for this PDF but detected no text on one or more pages, so parser-extracted PDF text was kept for those pages.",
                        )
                    )
                else:
                    extra["result_source"] = "none"
                    warnings.append(
                        ExtractionWarning(
                            code="ocr_no_text_detected",
                            message="OCR fallback ran but no text was detected in the PDF pages.",
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
                ocr_used=ocr_used,
                status=extraction_status,
                warnings=warnings,
            ),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )
