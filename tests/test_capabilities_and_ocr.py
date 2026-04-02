from pathlib import Path

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from app.extractors.image_ocr import ImageOcrExtractor
from app.extractors.pdf import PdfExtractor
from app.main import app


client = TestClient(app)


def test_capabilities_endpoint() -> None:
    response = client.get('/capabilities')
    assert response.status_code == 200
    payload = response.json()
    assert 'ocr' in payload
    assert 'pdf' in payload
    assert 'documents' in payload


def test_image_ocr_gracefully_handles_missing_tesseract(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.image_ocr.tesseract_available', lambda: False)
    image_path = tmp_path / 'sample.png'
    Image.new('RGB', (20, 20), color='white').save(image_path)

    payload = ImageOcrExtractor().extract(image_path, 'sample.png', 'image/png')

    assert payload.extraction.status == 'partial'
    warning_codes = {warning.code for warning in payload.extraction.warnings}
    assert 'tesseract_not_available' in warning_codes
    assert payload.extra['ocr_available'] is False
    assert payload.raw_text == ''


def test_pdf_extractor_uses_ocr_fallback_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.extractors.pdf.pytesseract.image_to_string', lambda image: 'ocr text')

    pdf_path = tmp_path / 'blank.pdf'
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(pdf_path, 'blank.pdf', 'application/pdf', ocr_strategy='auto')

    assert payload.extraction.ocr_used is True
    assert payload.extraction.status == 'success'
    assert payload.raw_text == 'ocr text'
    assert payload.extra['ocr_available'] is True
    assert payload.extra['ocr_backend'] == 'tesseract'
