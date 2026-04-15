"""OCR backend abstraction layer.

Provides a pluggable interface for OCR engines (Tesseract, PaddleOCR, etc.)
so extractors can select backends at runtime without hardcoding imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from PIL import Image


OcrBackendName = Literal["auto", "tesseract", "paddleocr"]


@dataclass
class WordBox:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float


@dataclass
class OcrBackendResult:
    text: str
    confidence: float | None = None
    words: list[WordBox] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OcrBackend(Protocol):
    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def run(
        self,
        image: Image.Image,
        *,
        variant_name: str | None = None,
        rotation: int = 0,
    ) -> OcrBackendResult: ...


class OcrBackendNotAvailableError(RuntimeError):
    """Raised when a requested OCR backend is not installed or not usable."""


def get_backend(name: OcrBackendName = "auto") -> OcrBackend:
    """Resolve an OCR backend by name, with auto-fallback.

    "auto" prefers PaddleOCR when available, otherwise Tesseract.
    Raises OcrBackendNotAvailableError if the explicitly requested backend is missing.
    """
    from app.ocr_backends.paddleocr_backend import PaddleOCRBackend
    from app.ocr_backends.tesseract_backend import TesseractBackend

    if name == "paddleocr":
        backend = PaddleOCRBackend()
        if not backend.is_available():
            raise OcrBackendNotAvailableError(
                "PaddleOCR is not available. Install paddlepaddle-gpu and paddleocr, "
                "or use ocr_backend=auto to fall back to Tesseract."
            )
        return backend

    if name == "tesseract":
        backend = TesseractBackend()
        if not backend.is_available():
            raise OcrBackendNotAvailableError(
                "Tesseract is not available. Install the tesseract binary."
            )
        return backend

    # auto: prefer PaddleOCR, fall back to Tesseract
    paddle = PaddleOCRBackend()
    if paddle.is_available():
        return paddle
    return TesseractBackend()
