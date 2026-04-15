"""Pydantic models for extraction profiles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VlmEndpointConfig(BaseModel):
    """Configuration for a VLM endpoint override within a profile."""

    base_url: str | None = None
    model: str | None = None
    timeout: float = 30.0
    api_key: str | None = None


class ClassifierConfig(BaseModel):
    """Configuration for document classification within a profile."""

    strategy: Literal["rules", "layout", "donut-irs", "vlm", "auto"] = "rules"
    model_name: str | None = None
    confidence_threshold: float = 0.75


class ExtractionProfile(BaseModel):
    """Named extraction profile bundling OCR, classification, and enrichment settings."""

    name: str
    description: str = ""
    ocr_strategy: Literal["auto", "always", "never", "force_image", "vlm", "vlm_hybrid"] = "auto"
    ocr_backend: Literal["auto", "tesseract", "paddleocr"] = "auto"
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    enable_llm_enrichment: bool = False
    mask_pii: bool = True
    vlm: VlmEndpointConfig = Field(default_factory=VlmEndpointConfig)
