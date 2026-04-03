from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_extract_text_file_without_chunks() -> None:
    response = client.post(
        '/extract',
        files={'file': ('hello.txt', b'hello\n\nworld', 'text/plain')},
        data={'include_chunks': 'false'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['chunks'] == []
    assert payload['extra']['ocr_strategy'] == 'auto'


def test_invalid_ocr_strategy_rejected() -> None:
    response = client.post(
        '/extract',
        files={'file': ('hello.txt', b'hello', 'text/plain')},
        data={'ocr_strategy': 'sometimes'},
    )
    assert response.status_code == 400
    assert 'Invalid ocr_strategy' in response.json()['detail']


def test_blank_pdf_reports_ocr_unavailable() -> None:
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    response = client.post(
        '/extract',
        files={'file': ('blank.pdf', BytesIO(pdf_bytes), 'application/pdf')},
        data={'ocr_strategy': 'always'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['metadata']['source_type'] == 'pdf'
    assert payload['extraction']['status'] == 'partial'
    warning_codes = {warning['code'] for warning in payload['extraction']['warnings']}
    assert 'pdf_no_text_layer' in warning_codes
    assert 'ocr_not_available' in warning_codes
    assert payload['extra']['ocr_available'] is False
    assert payload['extra']['ocr_strategy'] == 'always'
    assert payload['extra']['ocr_attempted'] is False
    assert payload['extra']['result_source'] == 'none'
    assert payload['extra']['content_detected'] is False
    assert payload['extra']['empty_content'] is True
    assert payload['extra']['partial_reason'] == 'empty_content'
    assert payload['extra']['warning_codes'] == ['pdf_no_text_layer', 'ocr_not_available']
    assert payload['extra']['page_provenance'] == [
        {'page_number': 1, 'source': 'none', 'has_text': False, 'text_length': 0}
    ]


def test_pdf_text_layer_with_always_strategy_reports_parser_result_source_when_ocr_unavailable() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'parser text')
    pdf_bytes = document.tobytes()
    response = client.post(
        '/extract',
        files={'file': ('text-layer.pdf', BytesIO(pdf_bytes), 'application/pdf')},
        data={'ocr_strategy': 'always'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['extraction']['status'] == 'success'
    warning_codes = {warning['code'] for warning in payload['extraction']['warnings']}
    assert 'ocr_not_available' in warning_codes
    assert payload['raw_text'].strip() == 'parser text'
    assert payload['extra']['ocr_attempted'] is False
    assert payload['extra']['result_source'] == 'parser'
    assert payload['extra']['segment_count'] >= 1
    assert payload['extra']['chunk_count'] >= 1
    assert payload['extra']['non_empty_page_count'] == 1
    assert payload['extra']['content_detected'] is True
    assert payload['extra']['empty_content'] is False
    assert payload['extra']['page_provenance'] == [
        {'page_number': 1, 'source': 'parser', 'has_text': True, 'text_length': 11}
    ]


def test_blank_pdf_with_never_strategy_reports_disabled_without_attempting_ocr() -> None:
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    response = client.post(
        '/extract',
        files={'file': ('blank.pdf', BytesIO(pdf_bytes), 'application/pdf')},
        data={'ocr_strategy': 'never'},
    )
    assert response.status_code == 200
    payload = response.json()
    warning_codes = {warning['code'] for warning in payload['extraction']['warnings']}
    assert payload['extraction']['status'] == 'partial'
    assert 'pdf_no_text_layer' in warning_codes
    assert 'ocr_disabled' in warning_codes
    assert payload['extra']['ocr_attempted'] is False
    assert payload['extra']['result_source'] == 'none'
    assert payload['extra']['empty_content'] is True
    assert payload['extra']['partial_reason'] == 'empty_content'


def test_extract_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setenv('LOCI_EXTRACT_MAX_UPLOAD_BYTES', '8')
    response = client.post(
        '/extract',
        files={'file': ('big.txt', b'0123456789', 'text/plain')},
    )
    assert response.status_code == 413
    assert 'Maximum allowed size is 8 bytes' in response.json()['detail']


def test_pdf_page_limit_applies_partial_warning(monkeypatch) -> None:
    monkeypatch.setenv('LOCI_EXTRACT_MAX_PDF_PAGES', '2')
    document = fitz.open()
    for idx in range(3):
        page = document.new_page()
        page.insert_text((72, 72), f'page {idx + 1}')
    pdf_bytes = document.tobytes()
    response = client.post(
        '/extract',
        files={'file': ('three-pages.pdf', BytesIO(pdf_bytes), 'application/pdf')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['metadata']['page_count'] == 3
    assert payload['extraction']['status'] == 'partial'
    warning_codes = {warning['code'] for warning in payload['extraction']['warnings']}
    assert 'pdf_page_limit_applied' in warning_codes
    assert payload['extra']['page_limit_applied'] is True
    assert payload['extra']['processed_page_count'] == 2
    assert payload['extra']['max_pdf_pages'] == 2
    assert payload['extra']['non_empty_page_count'] == 2
