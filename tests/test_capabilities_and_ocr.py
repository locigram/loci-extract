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
    assert payload.extra['ocr_attempted'] is True
    assert payload.extra['result_source'] == 'ocr'
    assert all(segment.metadata['source'] == 'ocr' for segment in payload.segments)


def test_pdf_extractor_always_uses_ocr_when_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.extractors.pdf.pytesseract.image_to_string', lambda image: 'ocr override text')

    pdf_path = tmp_path / 'text-layer.pdf'
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'parser text')
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(pdf_path, 'text-layer.pdf', 'application/pdf', ocr_strategy='always')

    assert payload.extraction.ocr_used is True
    assert payload.extraction.status == 'success'
    assert payload.raw_text == 'ocr override text'
    assert payload.extra['ocr_attempted'] is True
    assert payload.extra['result_source'] == 'ocr'
    assert payload.extra['page_provenance'] == [
        {'page_number': 1, 'source': 'ocr', 'has_text': True, 'text_length': 17}
    ]
    assert all(segment.metadata['source'] == 'ocr' for segment in payload.segments)


def test_pdf_extractor_always_falls_back_to_parser_text_when_ocr_finds_nothing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.extractors.pdf.pytesseract.image_to_string', lambda image: '   ')

    pdf_path = tmp_path / 'text-layer.pdf'
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'parser text')
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(pdf_path, 'text-layer.pdf', 'application/pdf', ocr_strategy='always')

    assert payload.extraction.ocr_used is True
    assert payload.extraction.status == 'success'
    assert payload.raw_text == 'parser text'
    assert payload.extra['ocr_attempted'] is True
    assert payload.extra['result_source'] == 'parser_fallback'
    assert payload.extra['page_provenance'] == [
        {'page_number': 1, 'source': 'parser_fallback', 'has_text': True, 'text_length': 11}
    ]
    assert all(segment.metadata['source'] == 'parser_fallback' for segment in payload.segments)
    warning_codes = {warning.code for warning in payload.extraction.warnings}
    assert 'ocr_no_text_detected' in warning_codes


def test_pdf_extractor_tracks_mixed_page_provenance_for_always_ocr(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    ocr_results = iter(['ocr page one', '   '])
    monkeypatch.setattr('app.extractors.pdf.pytesseract.image_to_string', lambda image: next(ocr_results))

    pdf_path = tmp_path / 'mixed.pdf'
    document = fitz.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), 'parser one')
    second_page = document.new_page()
    second_page.insert_text((72, 72), 'parser two')
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(pdf_path, 'mixed.pdf', 'application/pdf', ocr_strategy='always')

    assert payload.extraction.ocr_used is True
    assert payload.extraction.status == 'success'
    assert payload.extra['result_source'] == 'ocr'
    assert payload.extra['page_provenance'] == [
        {'page_number': 1, 'source': 'ocr', 'has_text': True, 'text_length': 12},
        {'page_number': 2, 'source': 'parser_fallback', 'has_text': True, 'text_length': 10},
    ]
    assert [segment.metadata['source'] for segment in payload.segments] == ['ocr', 'parser_fallback']
