from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import (
    DocumentMetadata,
    ExtractionMethod,
    ExtractionPayload,
    ExtractionWarning,
    TextSegment,
)


class DocxExtractor(BaseExtractor):
    name = "python-docx"

    def supports(self, filename: str, mime_type: str) -> bool:
        return filename.lower().endswith(".docx") or mime_type == (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def _iter_block_items(self, document: DocxDocument):
        body = document.element.body
        for child in body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, document)
            elif isinstance(child, CT_Tbl):
                yield Table(child, document)

    def _table_to_text(self, table: Table) -> str:
        rows: list[str] = []
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells]
            if any(values):
                rows.append(" | ".join(values))
        return "\n".join(rows).strip()

    def _paragraph_segment_kind(self, paragraph: Paragraph) -> tuple[str, dict[str, object]]:
        style_name = (getattr(getattr(paragraph, "style", None), "name", "") or "").strip()
        style_lower = style_name.lower()
        metadata: dict[str, object] = {}
        if style_name:
            metadata["style_name"] = style_name

        if style_lower.startswith("heading"):
            level = None
            parts = style_name.split()
            if len(parts) > 1 and parts[-1].isdigit():
                level = int(parts[-1])
                metadata["heading_level"] = level
            metadata["structure"] = "heading"
            return "section", metadata

        if any(token in style_lower for token in ("list", "bullet", "number")):
            metadata["structure"] = "list_item"
            return "paragraph", metadata

        metadata["structure"] = "paragraph"
        return "paragraph", metadata

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
        ocr_backend: str = "auto",
    ) -> ExtractionPayload:
        doc = Document(str(file_path))
        document_id = str(uuid4())
        segments: list[TextSegment] = []
        text_parts: list[str] = []
        paragraph_idx = 0
        table_idx = 0
        segment_idx = 0

        for block in self._iter_block_items(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                paragraph_idx += 1
                segment_idx += 1
                segment_type, metadata = self._paragraph_segment_kind(block)
                label_prefix = "heading" if segment_type == "section" else "paragraph"
                segments.append(
                    TextSegment(
                        type=segment_type,
                        index=segment_idx,
                        label=f"{label_prefix}-{paragraph_idx}",
                        text=text,
                        metadata=metadata,
                    )
                )
                text_parts.append(text)
            elif isinstance(block, Table):
                table_text = self._table_to_text(block)
                if not table_text:
                    continue
                table_idx += 1
                segment_idx += 1
                segments.append(
                    TextSegment(
                        type="table",
                        index=segment_idx,
                        label=f"table-{table_idx}",
                        text=table_text,
                        metadata={"table_index": table_idx},
                    )
                )
                text_parts.append(table_text)

        text = "\n\n".join(text_parts)
        warnings: list[ExtractionWarning] = []
        status = "success"
        if not segments:
            status = "partial"
            warnings.append(
                ExtractionWarning(
                    code="docx_no_text_detected",
                    message="No extractable paragraph or table text was detected in this DOCX document.",
                )
            )
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(filename=filename, mime_type=mime_type, source_type="docx"),
            extraction=ExtractionMethod(extractor=self.name, status=status, warnings=warnings),
            raw_text=text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
