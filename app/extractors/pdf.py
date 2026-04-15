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
from app.ocr_backends import OcrBackendNotAvailableError, get_backend
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, ExtractionWarning, TextSegment


DEFAULT_MAX_PDF_PAGES = 200

# VLM render settings — tuned for Qwen3-VL on tax forms. Higher DPI catches small
# box numbers and fine print that 144 DPI / 1568px was squashing. Tax documents
# benefit most; general PDFs cost roughly 2x more pixels per page.
_VLM_RENDER_DPI = 216
_VLM_RENDER_MATRIX_SCALE = 3  # fitz.Matrix(3, 3) ≈ 216 DPI (72 * 3)
_VLM_MAX_IMAGE_DIM = 2560


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
        self, document: fitz.Document, *, max_pages: int | None = None, ocr_backend_instance=None
    ) -> dict[int, dict[str, object]]:
        ocr_pages: dict[int, dict[str, object]] = {}
        for idx, page in enumerate(document, start=1):
            if max_pages is not None and idx > max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pix.tobytes("png")))
            if ocr_backend_instance is not None:
                ocr_result = extract_best_ocr_result(image, backend=ocr_backend_instance)
            else:
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

    def _extract_force_image(
        self,
        *,
        document: fitz.Document,
        file_path: Path,
        document_id: str,
        filename: str,
        mime_type: str,
        total_pages: int,
        max_pages: int,
        processed_page_count: int,
        page_limit_applied: bool,
        ocr_backend: str = "auto",
    ) -> ExtractionPayload:
        """OCR every page from rendered images, completely ignoring the PDF text layer.

        Used when ocr_strategy=force_image to handle PDFs with garbage text overlays.
        """
        warnings: list[ExtractionWarning] = []
        extraction_status = "success"

        # Resolve backend
        resolved_ocr_backend = None
        if ocr_backend not in ("auto", "tesseract"):
            try:
                resolved_ocr_backend = get_backend(ocr_backend)
                if not resolved_ocr_backend.is_available():
                    resolved_ocr_backend = None
            except OcrBackendNotAvailableError:
                pass
            backend_available = resolved_ocr_backend is not None
        else:
            backend_available = tesseract_available()

        if not backend_available:
            warnings.append(
                ExtractionWarning(
                    code="ocr_not_available",
                    message="force_image requires an OCR backend, but none is available.",
                )
            )
            return ExtractionPayload(
                document_id=document_id,
                metadata=DocumentMetadata(
                    filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages,
                ),
                extraction=ExtractionMethod(extractor=self.name, status="partial", warnings=warnings),
                raw_text="",
                segments=[],
                chunks=[],
                extra={
                    "ocr_strategy": "force_image",
                    "ocr_backend_requested": ocr_backend,
                    "ocr_available": False,
                    "ocr_backend": None,
                    "ocr_attempted": False,
                    "result_source": "none",
                    "page_limit_applied": page_limit_applied,
                    "processed_page_count": processed_page_count,
                    "max_pdf_pages": max_pages,
                    "page_provenance": [],
                },
            )

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

        # OCR every page from rendered image — no parser text used
        ocr_pages = self._extract_pdf_ocr(document, max_pages=max_pages, ocr_backend_instance=resolved_ocr_backend)
        backend_name = resolved_ocr_backend.name if resolved_ocr_backend else "tesseract"

        # Build outputs using OCR only (empty parser_pages = all text comes from OCR)
        pages, page_segments, page_provenance = self._build_page_outputs(
            processed_page_count=processed_page_count,
            parser_pages={},
            ocr_pages=ocr_pages,
            prefer_ocr=True,
            ocr_attempted=True,
        )
        table_segments = self._build_table_segments(
            processed_page_count=processed_page_count,
            pdf_tables={},
            ocr_pages=ocr_pages,
            page_provenance=page_provenance,
        )
        segments = page_segments + table_segments

        ocr_scores = [
            entry.get("ocr_score", 0.0)
            for entry in page_provenance
            if isinstance(entry.get("ocr_score"), (int, float))
        ]
        avg_score = round(sum(ocr_scores) / len(ocr_scores), 2) if ocr_scores else 0.0

        result_source = "ocr_force_image"
        if not pages:
            extraction_status = "partial"
            result_source = "none"
            warnings.append(
                ExtractionWarning(
                    code="ocr_no_text_detected",
                    message="force_image OCR ran on all pages but no text was detected.",
                )
            )
        elif avg_score < 10:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="ocr_low_quality",
                    message="force_image OCR recovered text, but quality scored low.",
                )
            )

        raw_text = "\n\n".join(pages)
        extra: dict[str, object] = {
            "ocr_strategy": "force_image",
            "ocr_backend_requested": ocr_backend,
            "ocr_available": True,
            "ocr_backend": backend_name,
            "ocr_attempted": True,
            "result_source": result_source,
            "page_limit_applied": page_limit_applied,
            "processed_page_count": processed_page_count,
            "max_pdf_pages": max_pages,
            "page_provenance": page_provenance,
            "ocr_average_score": avg_score,
            "ocr_passes_by_page": {
                str(pn): ocr_result.get("ocr_passes", []) for pn, ocr_result in ocr_pages.items()
            },
            "text_layer_ignored": True,
        }

        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(
                filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages,
            ),
            extraction=ExtractionMethod(
                extractor=self.name, ocr_used=True, status=extraction_status, warnings=warnings,
            ),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )

    def _extract_vlm_hybrid(
        self,
        *,
        document: fitz.Document,
        document_id: str,
        filename: str,
        mime_type: str,
        total_pages: int,
        max_pages: int,
        processed_page_count: int,
        page_limit_applied: bool,
    ) -> ExtractionPayload:
        """Hybrid VLM pipeline: parser text → LLM verification → VLM fallback per page.

        Stage 1: Read text layer (instant)
        Stage 2: Verify text quality via fast LLM (1-2s)
        Stage 3: VLM image extraction for bad/missing pages (30-60s)
        """
        from app.llm.config import get_llm_client, get_vlm_client
        from app.extractors.vlm import verify_text_quality, vlm_extract_page
        import logging as _logging
        import time as _time
        _log = _logging.getLogger("loci.extractors.vlm_hybrid")

        warnings: list[ExtractionWarning] = []
        extraction_status = "success"
        pipeline_start = _time.perf_counter()

        vlm_client = get_vlm_client()
        llm_client = get_llm_client()  # Fast 8B model for verification

        if vlm_client is None:
            warnings.append(ExtractionWarning(
                code="vlm_not_available",
                message="VLM hybrid requested but no VLM endpoint is configured.",
            ))
            return ExtractionPayload(
                document_id=document_id,
                metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages),
                extraction=ExtractionMethod(extractor=self.name, status="partial", warnings=warnings),
                raw_text="", segments=[], chunks=[],
                extra={"ocr_strategy": "vlm_hybrid", "result_source": "none"},
            )

        if page_limit_applied:
            extraction_status = "partial"
            warnings.append(ExtractionWarning(
                code="pdf_page_limit_applied",
                message=f"PDF has {total_pages} pages, but only the first {max_pages} were processed.",
            ))

        pages: list[str] = []
        segments: list[TextSegment] = []
        page_provenance: list[dict[str, object]] = []
        vlm_fields_by_page: dict[int, dict] = {}
        stats = {"parser_used": 0, "vlm_used": 0, "verify_calls": 0}
        trace_pages: list[dict[str, object]] = []

        for idx, page in enumerate(document, start=1):
            if idx > processed_page_count:
                break

            page_start = _time.perf_counter()
            page_text = ""
            source = "none"
            verification: dict | None = None
            verify_ms: int | None = None
            vlm_trace: dict[str, object] | None = None

            # Stage 1: Try parser text
            parser_text = page.get_text("text").strip()
            parser_chars = len(parser_text)
            parser_garbage = False

            if parser_text and len(parser_text) >= 10:
                # Quick heuristic check first (free)
                if self._looks_like_glyph_garbage(parser_text):
                    _log.info("Page %d: parser text is glyph garbage, going to VLM", idx)
                    verification = {"usable": False, "reason": "glyph_garbage_heuristic", "confidence": 1.0}
                    parser_garbage = True
                elif llm_client:
                    # Stage 2: LLM verification
                    _log.info("Page %d: verifying parser text quality (%d chars)...", idx, len(parser_text))
                    verify_start = _time.perf_counter()
                    verification = verify_text_quality(llm_client, parser_text)
                    verify_ms = int((_time.perf_counter() - verify_start) * 1000)
                    stats["verify_calls"] += 1
                    _log.info("Page %d: verification result: usable=%s reason=%s", idx, verification["usable"], verification["reason"])
                else:
                    # No LLM for verification — trust heuristic pass
                    verification = {"usable": True, "reason": "no_llm_for_verify", "confidence": 0.5}

                if verification and verification["usable"]:
                    page_text = parser_text
                    source = "parser"
                    stats["parser_used"] += 1
                    _log.info("Page %d: using parser text (%d chars)", idx, len(page_text))

            # Stage 3: VLM fallback for bad/missing text
            vlm_ms: int | None = None
            if not page_text:
                _log.info("Page %d: rendering image for VLM extraction...", idx)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(_VLM_RENDER_MATRIX_SCALE, _VLM_RENDER_MATRIX_SCALE),
                    alpha=False,
                )
                image = Image.open(BytesIO(pix.tobytes("png")))

                # Resize for VLM
                max_dim = _VLM_MAX_IMAGE_DIM
                w, h = image.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

                vlm_trace = {"render_dpi": _VLM_RENDER_DPI}
                vlm_start = _time.perf_counter()
                result = vlm_extract_page(vlm_client, image, doc_type="unknown", trace=vlm_trace)
                vlm_ms = int((_time.perf_counter() - vlm_start) * 1000)
                vlm_trace["ms"] = vlm_ms
                if result:
                    page_text = str(result.get("raw_text", "")).strip()
                    if result.get("fields"):
                        vlm_fields_by_page[idx] = result["fields"]

                source = "vlm" if page_text else "none"
                if page_text:
                    stats["vlm_used"] += 1
                    _log.info("Page %d: VLM extracted %d chars", idx, len(page_text))
                else:
                    _log.warning("Page %d: VLM returned no text", idx)

            pages.append(page_text)
            prov_entry: dict[str, object] = {
                "page_number": idx,
                "source": source,
                "has_text": bool(page_text),
                "text_length": len(page_text),
            }
            if verification:
                prov_entry["text_verification"] = verification
            page_provenance.append(prov_entry)

            page_ms = int((_time.perf_counter() - page_start) * 1000)
            trace_entry: dict[str, object] = {
                "page": idx,
                "parser_chars": parser_chars,
                "parser_looked_like_garbage": parser_garbage,
                "verify": ({**verification, "ms": verify_ms} if verification else None),
                "stage_selected": source,
                "vlm": vlm_trace,
                "final_chars": len(page_text),
                "ms": page_ms,
            }
            trace_pages.append(trace_entry)

            if page_text:
                segments.append(TextSegment(
                    type="page", index=idx, label=f"page-{idx}", text=page_text,
                    metadata={"source": source, "page_number": idx},
                ))

        raw_text = "\n\n".join(p for p in pages if p)
        if not raw_text:
            extraction_status = "partial"
            warnings.append(ExtractionWarning(
                code="vlm_hybrid_no_text",
                message="Hybrid pipeline found no usable text on any page.",
            ))

        result_source = "parser" if stats["parser_used"] == processed_page_count else (
            "vlm" if stats["vlm_used"] == processed_page_count else (
                "hybrid" if stats["parser_used"] > 0 and stats["vlm_used"] > 0 else "none"
            )
        )

        total_ms = int((_time.perf_counter() - pipeline_start) * 1000)
        extra: dict[str, object] = {
            "ocr_strategy": "vlm_hybrid",
            "ocr_backend": "vlm",
            "result_source": result_source,
            "vlm_used": stats["vlm_used"] > 0,
            "vlm_model": vlm_client.model,
            "vlm_endpoint": vlm_client.base_url,
            "hybrid_stats": stats,
            "page_limit_applied": page_limit_applied,
            "processed_page_count": processed_page_count,
            "max_pdf_pages": max_pages,
            "page_provenance": page_provenance,
            "vlm_trace": {
                "pipeline": "vlm_hybrid",
                "vlm_model": vlm_client.model,
                "vlm_endpoint": vlm_client.base_url,
                "verify_model": llm_client.model if llm_client else None,
                "totals": {**stats, "total_ms": total_ms},
                "pages": trace_pages,
            },
        }
        if vlm_fields_by_page:
            extra["vlm_fields_by_page"] = vlm_fields_by_page

        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages),
            extraction=ExtractionMethod(extractor=self.name, ocr_used=False, status=extraction_status, warnings=warnings),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
        )

    def _extract_vlm(
        self,
        *,
        document: fitz.Document,
        document_id: str,
        filename: str,
        mime_type: str,
        total_pages: int,
        max_pages: int,
        processed_page_count: int,
        page_limit_applied: bool,
    ) -> ExtractionPayload:
        """Send rendered page images to a VLM for text extraction. Ignores text layer entirely."""
        from app.llm.config import get_vlm_client

        warnings: list[ExtractionWarning] = []
        extraction_status = "success"
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
                metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages),
                extraction=ExtractionMethod(extractor=self.name, status="partial", warnings=warnings),
                raw_text="",
                segments=[],
                chunks=[],
                extra={"ocr_strategy": "vlm", "result_source": "none", "text_layer_ignored": True, "vlm_used": True},
            )

        if page_limit_applied:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="pdf_page_limit_applied",
                    message=f"PDF has {total_pages} pages, but only the first {max_pages} were processed.",
                )
            )

        pages: list[str] = []
        segments: list[TextSegment] = []
        page_provenance: list[dict[str, object]] = []
        vlm_fields_by_page: dict[int, dict] = {}
        trace_pages: list[dict[str, object]] = []

        import logging as _logging
        import time as _time
        _vlm_log = _logging.getLogger("loci.extractors.vlm")
        pipeline_start = _time.perf_counter()

        from app.extractors.vlm import vlm_extract_page

        for idx, page in enumerate(document, start=1):
            if idx > processed_page_count:
                break

            page_start = _time.perf_counter()
            # Render page as image — 3x matrix ≈ 216 DPI for tax form fine print
            pix = page.get_pixmap(
                matrix=fitz.Matrix(_VLM_RENDER_MATRIX_SCALE, _VLM_RENDER_MATRIX_SCALE),
                alpha=False,
            )
            image = Image.open(BytesIO(pix.tobytes("png")))

            # Pre-process: resize to max dim on longest side — 2560 catches small
            # numerals on W-2 Box 12 / 1040 line items that 1568 was squashing.
            max_dim = _VLM_MAX_IMAGE_DIM
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            _vlm_log.info("Page %d: %dx%d pixels, sending to VLM...", idx, image.size[0], image.size[1])

            vlm_trace: dict[str, object] = {"render_dpi": _VLM_RENDER_DPI}
            vlm_start = _time.perf_counter()
            result = vlm_extract_page(client, image, doc_type="unknown", trace=vlm_trace)
            vlm_trace["ms"] = int((_time.perf_counter() - vlm_start) * 1000)

            page_text = ""
            if result:
                page_text = str(result.get("raw_text", "")).strip()
                if result.get("fields"):
                    vlm_fields_by_page[idx] = result["fields"]

            source = "vlm" if page_text else "none"
            pages.append(page_text)
            page_provenance.append({
                "page_number": idx,
                "source": source,
                "has_text": bool(page_text),
                "text_length": len(page_text),
            })
            trace_pages.append({
                "page": idx,
                "parser_chars": 0,
                "parser_looked_like_garbage": False,
                "verify": None,
                "stage_selected": source,
                "vlm": vlm_trace,
                "final_chars": len(page_text),
                "ms": int((_time.perf_counter() - page_start) * 1000),
            })

            if page_text:
                segments.append(TextSegment(
                    type="page",
                    index=idx,
                    label=f"page-{idx}",
                    text=page_text,
                    metadata={"source": "vlm", "page_number": idx},
                ))

        raw_text = "\n\n".join(p for p in pages if p)
        if not raw_text:
            extraction_status = "partial"
            warnings.append(
                ExtractionWarning(code="vlm_no_text_detected", message="VLM processed all pages but extracted no text.")
            )

        total_ms = int((_time.perf_counter() - pipeline_start) * 1000)
        extra: dict[str, object] = {
            "ocr_strategy": "vlm",
            "ocr_backend": "vlm",
            "ocr_attempted": False,
            "result_source": "vlm" if raw_text else "none",
            "text_layer_ignored": True,
            "vlm_used": True,
            "vlm_model": client.model,
            "vlm_endpoint": client.base_url,
            "vlm_max_image_dim": _VLM_MAX_IMAGE_DIM,
            "page_limit_applied": page_limit_applied,
            "processed_page_count": processed_page_count,
            "max_pdf_pages": max_pages,
            "page_provenance": page_provenance,
            "vlm_trace": {
                "pipeline": "vlm",
                "vlm_model": client.model,
                "vlm_endpoint": client.base_url,
                "verify_model": None,
                "totals": {
                    "parser_used": 0,
                    "vlm_used": sum(1 for t in trace_pages if t["stage_selected"] == "vlm"),
                    "verify_calls": 0,
                    "total_ms": total_ms,
                },
                "pages": trace_pages,
            },
        }
        if vlm_fields_by_page:
            extra["vlm_fields_by_page"] = vlm_fields_by_page

        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="pdf", page_count=total_pages),
            extraction=ExtractionMethod(extractor=self.name, ocr_used=False, status=extraction_status, warnings=warnings),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
            extra=extra,
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
        document = fitz.open(file_path)
        document_id = str(uuid4())
        total_pages = len(document)
        max_pages = self._max_pdf_pages()
        processed_page_count = min(total_pages, max_pages)
        page_limit_applied = total_pages > max_pages

        # force_image: skip text layer entirely, OCR rendered page images
        if ocr_strategy == "force_image":
            return self._extract_force_image(
                document=document,
                file_path=file_path,
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
                total_pages=total_pages,
                max_pages=max_pages,
                processed_page_count=processed_page_count,
                page_limit_applied=page_limit_applied,
                ocr_backend=ocr_backend,
            )

        # vlm: send rendered page images to VLM, ignore text layer entirely
        if ocr_strategy == "vlm":
            return self._extract_vlm(
                document=document,
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
                total_pages=total_pages,
                max_pages=max_pages,
                processed_page_count=processed_page_count,
                page_limit_applied=page_limit_applied,
            )

        # vlm_hybrid: parser text → LLM verify → VLM fallback per page
        if ocr_strategy == "vlm_hybrid":
            return self._extract_vlm_hybrid(
                document=document,
                document_id=document_id,
                filename=filename,
                mime_type=mime_type,
                total_pages=total_pages,
                max_pages=max_pages,
                processed_page_count=processed_page_count,
                page_limit_applied=page_limit_applied,
            )

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

        # Resolve OCR backend: for auto/tesseract, use local availability check
        # (preserves existing monkeypatch targets). For explicit non-default backends,
        # use the backend abstraction.
        resolved_ocr_backend = None
        if ocr_backend not in ("auto", "tesseract"):
            try:
                resolved_ocr_backend = get_backend(ocr_backend)
                if not resolved_ocr_backend.is_available():
                    resolved_ocr_backend = None
            except OcrBackendNotAvailableError:
                pass
            ocr_is_available = (resolved_ocr_backend is not None) or ocrmypdf_is_available
        else:
            ocr_is_available = tesseract_is_available or ocrmypdf_is_available

        resolved_backend_name = resolved_ocr_backend.name if resolved_ocr_backend else ("tesseract" if tesseract_is_available else ("ocrmypdf" if ocrmypdf_is_available else None))
        extra: dict[str, object] = {
            "ocr_strategy": ocr_strategy,
            "ocr_backend_requested": ocr_backend,
            "ocr_available": ocr_is_available,
            "ocr_backend": resolved_backend_name,
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
                extra["ocr_backend"] = "ocrmypdf"  # ocrmypdf is parser-layer, orthogonal to tesseract/paddle
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
            # For auto/tesseract, use local availability; for explicit backends, use resolved_ocr_backend
            if ocr_backend not in ("auto", "tesseract"):
                backend_available = resolved_ocr_backend is not None
            else:
                backend_available = tesseract_is_available
            if not backend_available:
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
                extra["ocr_backend"] = resolved_ocr_backend.name if resolved_ocr_backend else "tesseract"
                ocr_pages = self._extract_pdf_ocr(document, max_pages=max_pages, ocr_backend_instance=resolved_ocr_backend)
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
