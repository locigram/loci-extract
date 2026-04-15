from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


StructuredDocType = Literal["w2", "1099-nec", "receipt", "tax_return_package", "financial_statement", "unknown"]


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


class ClassificationResult(BaseModel):
    doc_type: StructuredDocType
    confidence: float = 0.0
    strategy: Literal["rules", "hint", "layout", "donut", "vlm"] = "rules"
    matched_signals: list[str] = Field(default_factory=list)


class ReviewMetadata(BaseModel):
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class StructuredDocument(BaseModel):
    document_type: StructuredDocType
    schema_version: str = "1.0"
    fields: dict[str, Any] = Field(default_factory=dict)
    review: ReviewMetadata = Field(default_factory=ReviewMetadata)


class StructuredExtractionResponse(BaseModel):
    document_id: str
    classification: ClassificationResult
    raw_extraction: ExtractionPayload
    structured: StructuredDocument
    extra: dict[str, Any] = Field(default_factory=dict)
