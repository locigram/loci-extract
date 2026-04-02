from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExtractionOptions(BaseModel):
    include_chunks: bool = True
    ocr_strategy: Literal["auto", "always", "never"] = "auto"


class ExtractionWarning(BaseModel):
    code: str
    message: str


class ExtractionMethod(BaseModel):
    extractor: str
    ocr_used: bool = False
    status: Literal["success", "partial", "failed"] = "success"
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class TextSegment(BaseModel):
    type: Literal["page", "sheet", "paragraph", "table", "section"]
    index: int | None = None
    label: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    source_refs: list[dict[str, Any]] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    filename: str
    mime_type: str
    source_type: str
    page_count: int | None = None
    sheet_names: list[str] = Field(default_factory=list)
    language: str | None = None


class ExtractionPayload(BaseModel):
    document_id: str
    metadata: DocumentMetadata
    extraction: ExtractionMethod
    raw_text: str
    segments: list[TextSegment] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
