from __future__ import annotations

import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import fitz
import pdfplumber
from PIL import Image

from app.capabilities import ocrmypdf_available, tesseract_available
from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.ocr import extract_best_ocr_result
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

    def _normalize_table_rows(self, rows: list[list[object] | None]) -> list[list[str]]:
        normalized_rows: list[list[str]] = []
        for row in rows:
            if not row:
                continue
            values = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(values):
                normalized_rows.append(values)
        return normalized_rows

    def _control_character_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        control_chars = sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t")
        return control_chars / max(len(text), 1)

    def _looks_like_glyph_garbage(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False
        if cleaned.count("(cid:") >= 3:
            return True
        return self._control_character_ratio(cleaned) >= 0.12

    def _parser_garbage_trigger_reason(self, parser_pages: dict[int, str]) -> str | None:
        if not parser_pages:
            return None
        combined = "\n".join(parser_pages.values())
        if combined.count("(cid:") >= 3:
            return "parser_glyph_garbage"
        if self._control_character_ratio(combined) >= 0.12:
            return "parser_glyph_garbage"
        return None

    def _run_ocrmypdf_fallback(self, file_path: Path) -> Path:
        output_dir = Path(tempfile.mkdtemp(prefix="loci-ocrmypdf-"))
        output_path = output_dir / f"{file_path.stem}.ocr.pdf"
        subprocess.run(
            [
                "ocrmypdf",
                "--force-ocr",
                "--rotate-pages",
                "--deskew",
                "--clean",
                str(file_path),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output_path

    def _format_table_text(self, rows: list[list[str]]) -> str:
        return "\n".join(" | ".join(cell for cell in row if cell) for row in rows if any(row)).strip()

    def _extract_pdf_tables(self, file_path: Path, *, max_pages: int | None = None) -> dict[int, list[dict[str, object]]]:
        extracted_tables: dict[int, list[dict[str, object]]] = {}
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    if max_pages is not None and idx > max_pages:
                        break
                    page_tables: list[dict[str, object]] = []
                    for table_index, table in enumerate(page.extract_tables() or [], start=1):
                        rows = self._normalize_table_rows(table)
                        if len(rows) < 2:
                            continue
                        column_count = max(len(row) for row in rows)
                        if column_count < 2:
                            continue
                        table_text = self._format_table_text(rows)
                        if not table_text:
                            continue
                        page_tables.append(
                            {
                                "text": table_text,
                                "page_table_index": table_index,
                                "row_count": len(rows),
                                "column_count": column_count,
                                "detection_method": "pdfplumber",
                            }
                        )
                    if page_tables:
                        extracted_tables[idx] = page_tables
        except Exception:
            return {}
        return extracted_tables

    def _extract_pdf_ocr(
        self, document: fitz.Document, *, max_pages: int | None = None
    ) -> dict[int, dict[str, object]]:
        ocr_pages: dict[int, dict[str, object]] = {}
        for idx, page in enumerate(document, start=1):
            if max_pages is not None and idx > max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            ocr_result = extract_best_ocr_result(image)
            ocr_result["page_rotation"] = page.rotation
            ocr_pages[idx] = ocr_result
        return ocr_pages

    def _build_page_outputs(
        self,
        *,
        processed_page_count: int,
        parser_pages: dict[int, str],
        ocr_pages: dict[int, dict[str, object]] | None = None,
        prefer_ocr: bool = False,
        ocr_attempted: bool = False,
        parser_source: str = "parser",
    ) -> tuple[list[str], list[TextSegment], list[dict[str, object]]]:
        pages: list[str] = []
        segments: list[TextSegment] = []
        provenance: list[dict[str, object]] = []
        ocr_pages = ocr_pages or {}

        for page_number in range(1, processed_page_count + 1):
            parser_text = parser_pages.get(page_number)
            ocr_result = ocr_pages.get(page_number, {})
            ocr_text = str(ocr_result.get("text") or "").strip() if isinstance(ocr_result, dict) else ""
            ocr_score = float(ocr_result.get("score") or 0.0) if isinstance(ocr_result, dict) else 0.0
            selected_pass = ocr_result.get("selected_pass") if isinstance(ocr_result, dict) else None
            selected_rotation = int(ocr_result.get("selected_rotation") or 0) if isinstance(ocr_result, dict) else 0

            if prefer_ocr and ocr_text:
                source = "ocr"
                text = ocr_text
            elif prefer_ocr and parser_text and ocr_attempted:
                source = "parser_fallback"
                text = parser_text
            elif parser_text:
                source = parser_source
                text = parser_text
            elif ocr_text:
                source = "ocr"
                text = ocr_text
            else:
                source = "none"
                text = None

            provenance_entry: dict[str, object] = {
                "page_number": page_number,
                "source": source,
                "has_text": bool(text),
                "text_length": len(text) if text else 0,
            }
            if ocr_attempted:
                provenance_entry["ocr_score"] = ocr_score
                provenance_entry["selected_ocr_pass"] = selected_pass
                provenance_entry["selected_ocr_rotation"] = selected_rotation
            provenance.append(provenance_entry)

            if not text:
                continue

            segment_metadata = {"source": source, "page_number": page_number}
            if ocr_attempted:
                segment_metadata["ocr_score"] = ocr_score
                segment_metadata["selected_ocr_pass"] = selected_pass
                segment_metadata["selected_ocr_rotation"] = selected_rotation
            pages.append(text)
            segments.append(
                TextSegment(
                    type="page",
                    index=page_number,
                    label=f"page-{page_number}",
                    text=text,
                    metadata=segment_metadata,
                )
            )

        return pages, segments, provenance

    def _build_table_segments(
        self,
        *,
        processed_page_count: int,
        pdf_tables: dict[int, list[dict[str, object]]] | None = None,
        ocr_pages: dict[int, dict[str, object]] | None = None,
        page_provenance: list[dict[str, object]] | None = None,
    ) -> list[TextSegment]:
        table_segments: list[TextSegment] = []
        pdf_tables = pdf_tables or {}
        ocr_pages = ocr_pages or {}
        page_source = {
            int(entry.get("page_number")): str(entry.get("source") or "none")
            for entry in (page_provenance or [])
            if isinstance(entry, dict) and isinstance(entry.get("page_number"), int)
        }

        table_segment_index = 0
        for page_number in range(1, processed_page_count + 1):
            page_tables = list(pdf_tables.get(page_number, []))
            if not page_tables and page_source.get(page_number) == "ocr":
                ocr_result = ocr_pages.get(page_number, {})
                for table_index, table_candidate in enumerate(ocr_result.get("table_candidates", []), start=1):
                    page_tables.append(
                        {
                            "text": str(table_candidate.get("text") or "").strip(),
                            "page_table_index": table_index,
                            "row_count": int(table_candidate.get("row_count") or 0),
                            "column_count": int(table_candidate.get("column_count") or 0),
                            "detection_method": str(table_candidate.get("detection_method") or "ocr_word_grid"),
                            "selected_ocr_pass": ocr_result.get("selected_pass"),
                            "selected_ocr_rotation": int(ocr_result.get("selected_rotation") or 0),
                            "ocr_score": float(ocr_result.get("score") or 0.0),
                        }
                    )

            for table in page_tables:
                table_text = str(table.get("text") or "").strip()
                if not table_text:
                    continue
                table_segment_index += 1
                metadata = {
                    "page_number": page_number,
                    "page_table_index": int(table.get("page_table_index") or table_segment_index),
                    "row_count": int(table.get("row_count") or 0),
                    "column_count": int(table.get("column_count") or 0),
                    "detection_method": table.get("detection_method") or "unknown",
                }
                if table.get("selected_ocr_pass"):
                    metadata["selected_ocr_pass"] = table["selected_ocr_pass"]
                if "selected_ocr_rotation" in table:
                    metadata["selected_ocr_rotation"] = int(table.get("selected_ocr_rotation") or 0)
                if "ocr_score" in table:
                    metadata["ocr_score"] = float(table.get("ocr_score") or 0.0)
                table_segments.append(
                    TextSegment(
                        type="table",
                        index=table_segment_index,
                        label=f"page-{page_number}-table-{metadata['page_table_index']}",
                        text=table_text,
                        metadata=metadata,
                    )
                )
        return table_segments

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
        pdf_tables = self._extract_pdf_tables(file_path, max_pages=max_pages)
        parser_garbage_trigger_reason = self._parser_garbage_trigger_reason(parser_pages)
        pages, page_segments, page_provenance = self._build_page_outputs(
            processed_page_count=processed_page_count,
            parser_pages=parser_pages,
        )
        table_segments = self._build_table_segments(
            processed_page_count=processed_page_count,
            pdf_tables=pdf_tables,
            page_provenance=page_provenance,
        )
        segments = page_segments + table_segments

        warnings: list[ExtractionWarning] = []
        extraction_status = "success"
        ocr_used = False
        tesseract_is_available = tesseract_available()
        ocrmypdf_is_available = ocrmypdf_available()
        ocr_is_available = tesseract_is_available or ocrmypdf_is_available
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_available": ocr_is_available,
            "ocr_backend": "tesseract" if tesseract_is_available else ("ocrmypdf" if ocrmypdf_is_available else None),
            "ocr_attempted": False,
            "result_source": "parser" if pages else "none",
            "page_limit_applied": page_limit_applied,
            "processed_page_count": processed_page_count,
            "max_pdf_pages": max_pages,
            "page_provenance": page_provenance,
        }

        if parser_garbage_trigger_reason:
            extra["parser_quality_issue"] = parser_garbage_trigger_reason

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

        if ocr_strategy == "auto" and parser_garbage_trigger_reason and ocrmypdf_is_available:
            try:
                extra["ocr_attempted"] = True
                extra["ocr_backend"] = "ocrmypdf"
                extra["ocrmypdf_trigger_reason"] = parser_garbage_trigger_reason
                ocrmypdf_path = self._run_ocrmypdf_fallback(file_path)
                with fitz.open(ocrmypdf_path) as ocrmypdf_document:
                    parser_pages = self._extract_pdf_text(ocrmypdf_document, max_pages=max_pages)
                pdf_tables = self._extract_pdf_tables(ocrmypdf_path, max_pages=max_pages)
                pages, page_segments, page_provenance = self._build_page_outputs(
                    processed_page_count=processed_page_count,
                    parser_pages=parser_pages,
                    parser_source="ocrmypdf",
                )
                table_segments = self._build_table_segments(
                    processed_page_count=processed_page_count,
                    pdf_tables=pdf_tables,
                    page_provenance=page_provenance,
                )
                segments = page_segments + table_segments
                extra["page_provenance"] = page_provenance
                extra["result_source"] = "ocrmypdf" if pages else "none"
                extra["ocrmypdf_applied"] = True
                ocr_used = True
                parser_garbage_trigger_reason = None
            except subprocess.CalledProcessError as exc:
                warnings.append(
                    ExtractionWarning(
                        code="ocrmypdf_failed",
                        message=f"OCRmyPDF fallback failed: {exc.stderr.strip() or exc.stdout.strip() or 'unknown error'}",
                    )
                )
                extraction_status = "partial"

        should_attempt_ocr = ocr_strategy == "always" or (ocr_strategy == "auto" and (not pages or parser_garbage_trigger_reason is not None))

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
            if not tesseract_is_available:
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
                extra["ocr_backend"] = "tesseract"
                ocr_pages = self._extract_pdf_ocr(document, max_pages=max_pages)
                ocr_used = True
                pages, page_segments, page_provenance = self._build_page_outputs(
                    processed_page_count=processed_page_count,
                    parser_pages=parser_pages,
                    ocr_pages=ocr_pages,
                    prefer_ocr=ocr_strategy == "always" or not pages,
                    ocr_attempted=True,
                )
                table_segments = self._build_table_segments(
                    processed_page_count=processed_page_count,
                    pdf_tables=pdf_tables,
                    ocr_pages=ocr_pages,
                    page_provenance=page_provenance,
                )
                segments = page_segments + table_segments
                extra["page_provenance"] = page_provenance
                ocr_scores = [entry.get("ocr_score", 0.0) for entry in page_provenance if isinstance(entry.get("ocr_score"), (int, float))]
                extra["ocr_average_score"] = round(sum(ocr_scores) / len(ocr_scores), 2) if ocr_scores else 0.0
                extra["ocr_passes_by_page"] = {
                    str(page_number): ocr_result.get("ocr_passes", []) for page_number, ocr_result in ocr_pages.items()
                }

                if any(entry["source"] == "ocr" for entry in page_provenance):
                    extraction_status = "partial" if page_limit_applied else "success"
                    extra["result_source"] = "ocr"
                    if extra["ocr_average_score"] < 10:
                        warnings.append(
                            ExtractionWarning(
                                code="ocr_low_quality",
                                message="OCR recovered text for this PDF, but the selected OCR results scored as low quality.",
                            )
                        )
                        extraction_status = "partial"
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
