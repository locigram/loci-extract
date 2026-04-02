from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from openpyxl import load_workbook

from app.chunking import build_chunks
from app.extractors.base import BaseExtractor
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, TextSegment


class XlsxExtractor(BaseExtractor):
    name = "openpyxl"

    def supports(self, filename: str, mime_type: str) -> bool:
        return filename.lower().endswith(".xlsx") or mime_type in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }

    def extract(
        self,
        file_path: Path,
        filename: str,
        mime_type: str,
        *,
        ocr_strategy: str = "auto",
    ) -> ExtractionPayload:
        wb = load_workbook(file_path, data_only=True)
        document_id = str(uuid4())
        segments: list[TextSegment] = []
        raw_parts: list[str] = []

        for sheet_idx, sheet in enumerate(wb.worksheets, start=1):
            rows_as_text: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(v).strip() for v in row if v is not None and str(v).strip()]
                if values:
                    rows_as_text.append(" | ".join(values))
            sheet_text = f"Sheet: {sheet.title}\n" + "\n".join(rows_as_text)
            raw_parts.append(sheet_text)
            segments.append(
                TextSegment(type="sheet", index=sheet_idx, label=sheet.title, text=sheet_text)
            )

        raw_text = "\n\n".join(raw_parts)
        return ExtractionPayload(
            document_id=document_id,
            metadata=DocumentMetadata(
                filename=filename,
                mime_type=mime_type,
                source_type="xlsx",
                sheet_names=[ws.title for ws in wb.worksheets],
            ),
            extraction=ExtractionMethod(extractor=self.name),
            raw_text=raw_text,
            segments=segments,
            chunks=build_chunks(document_id, segments),
        )
