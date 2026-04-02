from __future__ import annotations

from pathlib import Path

from app.extractors.base import BaseExtractor
from app.extractors.docx import DocxExtractor
from app.extractors.image_ocr import ImageOcrExtractor
from app.extractors.pdf import PdfExtractor
from app.extractors.plaintext import PlainTextExtractor
from app.extractors.xlsx import XlsxExtractor


class UnsupportedDocumentError(ValueError):
    pass


EXTRACTORS: list[BaseExtractor] = [
    PdfExtractor(),
    DocxExtractor(),
    XlsxExtractor(),
    ImageOcrExtractor(),
    PlainTextExtractor(),
]


def choose_extractor(filename: str, mime_type: str) -> BaseExtractor:
    for extractor in EXTRACTORS:
        if extractor.supports(filename, mime_type):
            return extractor
    raise UnsupportedDocumentError(f"No extractor available for {filename} ({mime_type})")


def extract_file(file_path: Path, filename: str, mime_type: str, *, ocr_strategy: str = "auto"):
    extractor = choose_extractor(filename, mime_type)
    return extractor.extract(file_path, filename, mime_type, ocr_strategy=ocr_strategy)
