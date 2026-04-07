from pathlib import Path

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from app.extractors.image_ocr import ImageOcrExtractor
from app.extractors.pdf import PdfExtractor
from app.main import app
from app.ocr import extract_best_ocr_result


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


def test_image_ocr_uses_best_pass_and_records_quality(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.image_ocr.tesseract_available', lambda: True)

    def fake_ocr(image):
        pass_name = getattr(image, '_ocr_pass_name', '')
        if pass_name == 'soft_upscale':
            return 'tax form w2 employee name wages withheld'
        if pass_name == 'threshold_180_upscale':
            return 'w2'
        return ' '

    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', fake_ocr)
    image_path = tmp_path / 'sample.png'
    Image.new('RGB', (50, 50), color='white').save(image_path)

    payload = ImageOcrExtractor().extract(image_path, 'sample.png', 'image/png')

    assert payload.extraction.status == 'success'
    assert payload.extra['selected_ocr_pass'] == 'soft_upscale'
    assert payload.extra['ocr_score'] > 25
    assert len(payload.extra['ocr_passes']) >= 3
    assert payload.extra['page_provenance'][0]['selected_ocr_pass'] == 'soft_upscale'


def test_extract_best_ocr_result_prefers_rotated_variant(monkeypatch) -> None:
    def fake_image_to_string(image):
        if getattr(image, '_ocr_rotation', 0) == 90:
            return 'rotated tax statement wages withheld employee name'
        return ' '

    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', fake_image_to_string)
    monkeypatch.setattr('app.ocr.pytesseract.image_to_data', lambda *args, **kwargs: {'text': []})

    result = extract_best_ocr_result(Image.new('RGB', (120, 60), color='white'))

    assert result['selected_pass'] == 'soft_upscale_rot90'
    assert result['selected_rotation'] == 90
    assert any(pass_result['rotation_degrees'] == 90 for pass_result in result['ocr_passes'])



def test_pdf_extractor_uses_ocr_fallback_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', lambda image: 'ocr text')

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
    assert payload.extra['ocr_average_score'] > 0
    assert all(segment.metadata['source'] == 'ocr' for segment in payload.segments)


def test_pdf_extractor_always_uses_ocr_when_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', lambda image: 'ocr override text')

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
        {
            'page_number': 1,
            'source': 'ocr',
            'has_text': True,
            'text_length': 17,
            'ocr_score': payload.extra['page_provenance'][0]['ocr_score'],
            'selected_ocr_pass': payload.extra['page_provenance'][0]['selected_ocr_pass'],
            'selected_ocr_rotation': payload.extra['page_provenance'][0]['selected_ocr_rotation'],
        }
    ]
    assert payload.extra['page_provenance'][0]['ocr_score'] > 0
    assert all(segment.metadata['source'] == 'ocr' for segment in payload.segments)


def test_pdf_extractor_always_falls_back_to_parser_text_when_ocr_finds_nothing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', lambda image: '   ')

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
    assert payload.extra['page_provenance'][0]['source'] == 'parser_fallback'
    assert payload.extra['page_provenance'][0]['ocr_score'] == 0.0
    assert all(segment.metadata['source'] == 'parser_fallback' for segment in payload.segments)
    warning_codes = {warning.code for warning in payload.extraction.warnings}
    assert 'ocr_no_text_detected' in warning_codes


def test_pdf_extractor_tracks_mixed_page_provenance_for_always_ocr(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    ocr_results = iter(
        [
            {'text': 'ocr page one', 'score': 22.0, 'selected_pass': 'soft_upscale', 'processed_mode': 'L', 'preprocessing': ['grayscale'], 'ocr_passes': []},
            {'text': '', 'score': 0.0, 'selected_pass': 'soft_upscale', 'processed_mode': 'L', 'preprocessing': ['grayscale'], 'ocr_passes': []},
        ]
    )
    monkeypatch.setattr('app.extractors.pdf.extract_best_ocr_result', lambda image: next(ocr_results))

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
    assert payload.extra['page_provenance'][0]['source'] == 'ocr'
    assert payload.extra['page_provenance'][1]['source'] == 'parser_fallback'
    assert payload.extra['page_provenance'][0]['ocr_score'] == 22.0
    assert payload.extra['page_provenance'][1]['ocr_score'] == 0.0
    assert [segment.metadata['source'] for segment in payload.segments] == ['ocr', 'parser_fallback']


def test_pdf_extractor_detects_glyph_garbage_parser_output() -> None:
    extractor = PdfExtractor()
    assert extractor._looks_like_glyph_garbage('normal readable financial statement text') is False
    assert extractor._looks_like_glyph_garbage('\u0013*() *,$ +\u0005 \u001d0.:,1\u0001%011)8\u0001\u001343+42030:2\u0001\u0011884*0)9043') is True
    assert extractor._looks_like_glyph_garbage('(cid:6)(cid:30)(cid:30)(cid:40)(cid:45)(cid:39)(cid:44) (cid:17)(cid:45)(cid:38)(cid:29)') is True



def test_pdf_extractor_auto_uses_ocrmypdf_for_glyph_garbage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.extractors.pdf.ocrmypdf_available', lambda: True)

    pdf_path = tmp_path / 'glyph-garbage.pdf'
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'parser text that will be replaced')
    document.save(pdf_path)
    document.close()

    extractor = PdfExtractor()
    monkeypatch.setattr(extractor, '_extract_pdf_text', lambda document, max_pages=None: {1: '(cid:6)(cid:30)(cid:30)(cid:40)(cid:45)(cid:39)(cid:44)'})
    monkeypatch.setattr(extractor, '_extract_pdf_tables', lambda file_path, max_pages=None: {})

    rerouted_pdf = tmp_path / 'rerouted.pdf'
    rerouted_pdf.write_bytes(pdf_path.read_bytes())
    monkeypatch.setattr(extractor, '_run_ocrmypdf_fallback', lambda file_path: rerouted_pdf)

    parser_calls = {'count': 0}

    def fake_extract_text(document, max_pages=None):
        parser_calls['count'] += 1
        if parser_calls['count'] == 1:
            return {1: '(cid:6)(cid:30)(cid:30)(cid:40)(cid:45)(cid:39)(cid:44)'}
        return {1: 'Balance Sheet readable after OCRmyPDF'}

    monkeypatch.setattr(extractor, '_extract_pdf_text', fake_extract_text)

    payload = extractor.extract(pdf_path, 'glyph-garbage.pdf', 'application/pdf', ocr_strategy='auto')

    assert payload.extraction.status == 'success'
    assert payload.extraction.ocr_used is True
    assert payload.raw_text == 'Balance Sheet readable after OCRmyPDF'
    assert payload.extra['result_source'] == 'ocrmypdf'
    assert payload.extra['ocr_backend'] == 'ocrmypdf'
    assert payload.extra['ocr_attempted'] is True
    assert payload.extra['ocrmypdf_trigger_reason'] == 'parser_glyph_garbage'
    assert payload.extra['page_provenance'][0]['source'] == 'ocrmypdf'
    assert payload.segments[0].metadata['source'] == 'ocrmypdf'



def test_pdf_extractor_surfaces_pdfplumber_table_segments(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / 'basic-table.pdf'
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'Quarterly summary')
    document.save(pdf_path)
    document.close()

    class FakePdfPage:
        def extract_tables(self):
            return [[['Name', 'Role'], ['Drew', 'Admin']]]

    class FakePdf:
        pages = [FakePdfPage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr('app.extractors.pdf.pdfplumber.open', lambda *args, **kwargs: FakePdf())

    payload = PdfExtractor().extract(pdf_path, 'basic-table.pdf', 'application/pdf')

    table_segments = [segment for segment in payload.segments if segment.type == 'table']
    assert payload.extraction.status == 'success'
    assert len(table_segments) == 1
    assert table_segments[0].text == 'Name | Role\nDrew | Admin'
    assert table_segments[0].metadata['page_number'] == 1
    assert table_segments[0].metadata['detection_method'] == 'pdfplumber'



def test_pdf_extractor_marks_low_quality_ocr_as_partial(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', lambda image: 'w2')

    pdf_path = tmp_path / 'blank.pdf'
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    payload = PdfExtractor().extract(pdf_path, 'blank.pdf', 'application/pdf', ocr_strategy='auto')

    warning_codes = {warning.code for warning in payload.extraction.warnings}
    assert payload.extraction.status == 'partial'
    assert 'ocr_low_quality' in warning_codes
    assert payload.extra['ocr_average_score'] < 25



def test_pdf_extractor_enrichment_surfaces_ocr_quality_summary(monkeypatch) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr(
        'app.extractors.pdf.extract_best_ocr_result',
        lambda image: {
            'text': 'Form W-2 Wage and Tax Statement 2024 Employee name John Doe',
            'score': 5.0,
            'selected_pass': 'soft_upscale',
            'processed_mode': 'L',
            'preprocessing': ['grayscale'],
            'ocr_passes': [],
        },
    )

    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()

    response = client.post(
        '/extract',
        files={'file': ('ocr-w2.pdf', pdf_bytes, 'application/pdf')},
        data={'ocr_strategy': 'always'},
    )

    assert response.status_code == 200
    payload = response.json()
    summary = payload['extra']['ocr_quality_summary']
    assert summary['attempted'] is True
    assert summary['result_source'] == 'ocr'
    assert summary['average_score'] == 5.0
    assert summary['low_quality'] is True
    assert summary['weak_pages'] == [1]
    assert payload['extra']['ocr_evidence_snippets'][0]['page_number'] == 1
    assert payload['extra']['ocr_evidence_snippets'][0]['selected_ocr_pass'] == 'soft_upscale'
    assert 'Form W-2 Wage and Tax Statement' in payload['extra']['ocr_evidence_snippets'][0]['snippet']
