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
