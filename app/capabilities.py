from __future__ import annotations

from shutil import which


def command_available(name: str) -> bool:
    return which(name) is not None


def tesseract_available() -> bool:
    return command_available("tesseract")


def ocrmypdf_available() -> bool:
    return command_available("ocrmypdf")


def ghostscript_available() -> bool:
    return command_available("gs")


def detect_capabilities() -> dict[str, object]:
    return {
        "ocr": {
            "tesseract": tesseract_available(),
            "ocrmypdf": ocrmypdf_available(),
            "ghostscript": ghostscript_available(),
        },
        "pdf": {
            "pdftoppm": command_available("pdftoppm"),
            "pdfinfo": command_available("pdfinfo"),
            "pymupdf": True,
        },
        "documents": {
            "docx": True,
            "xlsx": True,
            "images": True,
        },
    }
