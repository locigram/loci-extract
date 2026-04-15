"""Tesseract OCR backend — wraps pytesseract for the OcrBackend protocol."""

from __future__ import annotations

from PIL import Image

from app.capabilities import tesseract_available
from app.ocr_backends import OcrBackendResult


class TesseractBackend:
    @property
    def name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        return tesseract_available()

    def run(
        self,
        image: Image.Image,
        *,
        variant_name: str | None = None,
        rotation: int = 0,
    ) -> OcrBackendResult:
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
        return OcrBackendResult(text=text)
