from io import BytesIO

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def test_extract_structured_w2_response() -> None:
    response = client.post(
        '/extract/structured',
        files={
            'file': (
                'w2.txt',
                b"Form W-2 Wage and Tax Statement 2024\nEmployee name John Doe\nEmployee's social security number XXX-XX-1234\nEmployer name Example Payroll Inc\nEmployer identification number 12-3456789\n1 Wages, tips, other compensation 85000.00",
                'text/plain',
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'w2'
    assert payload['structured']['document_type'] == 'w2'
    assert payload['raw_extraction']['metadata']['source_type'] == 'text'
    assert payload['structured']['fields']['employee']['ssn_masked'] == 'XXX-XX-1234'
    assert payload['structured']['fields']['evidence']['employee_name'] is not None


def test_extract_structured_1099_nec_response() -> None:
    response = client.post(
        '/extract/structured',
        files={
            'file': (
                '1099.txt',
                b"Form 1099-NEC 2024\nRecipient's name Jane Contractor\nRecipient's TIN XXX-XX-4321\nPayer's name ACME Services LLC\nPayer's TIN 98-7654321\n1 Nonemployee compensation 25000.00",
                'text/plain',
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == '1099-nec'
    assert payload['structured']['fields']['boxes']['1_nonemployee_compensation'] == 25000.0
    assert payload['structured']['fields']['evidence']['box_1'] is not None


def test_extract_structured_unknown_document_requires_review() -> None:
    response = client.post(
        '/extract/structured',
        files={'file': ('unknown.txt', b'hello world', 'text/plain')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'unknown'
    assert payload['structured']['review']['requires_human_review'] is True


def test_extract_structured_doc_type_hint_override() -> None:
    response = client.post(
        '/extract/structured',
        files={'file': ('mystery.txt', b'hello world', 'text/plain')},
        data={'doc_type_hint': 'receipt'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'receipt'
    assert payload['classification']['strategy'] == 'hint'


def test_extract_structured_mask_pii_false() -> None:
    response = client.post(
        '/extract/structured',
        files={
            'file': (
                'w2.txt',
                b"Form W-2 Wage and Tax Statement 2024\nEmployee name John Doe\nEmployee's social security number 123-45-6789\nEmployer name Example Payroll Inc\nEmployer identification number 12-3456789\n1 Wages, tips, other compensation 85000.00",
                'text/plain',
            )
        },
        data={'mask_pii': 'false'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['structured']['fields']['employee']['ssn_masked'] == '123-45-6789'


def test_extract_structured_without_chunks_preserves_raw_payload() -> None:
    response = client.post(
        '/extract/structured',
        files={'file': ('receipt.txt', b'Receipt\n03/14/2024\nSubtotal 10.00\nTax 0.80\nTotal 10.80\nVisa', 'text/plain')},
        data={'include_chunks': 'false'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['raw_extraction']['chunks'] == []
    assert payload['structured']['document_type'] == 'receipt'


def test_extract_structured_pdf_with_ocr_provenance_requires_review_for_tax_doc(monkeypatch) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)

    def fake_ocr(image):
        pass_name = getattr(image, '_ocr_pass_name', '')
        if pass_name == 'soft_upscale':
            return "Form W-2 Wage and Tax Statement 2024\nEmployee name John Doe\nEmployer name ACME Payroll\n1 Wages, tips, other compensation 1000.00"
        return 'w2'

    monkeypatch.setattr('app.ocr.pytesseract.image_to_string', fake_ocr)
    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    response = client.post(
        '/extract/structured',
        files={'file': ('ocr-w2.pdf', BytesIO(pdf_bytes), 'application/pdf')},
        data={'ocr_strategy': 'always'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'w2'
    assert payload['structured']['review']['requires_human_review'] is True
    assert 'ocr_backed_tax_document' in payload['structured']['review']['review_reasons']
    assert payload['raw_extraction']['extra']['ocr_average_score'] > 25


def test_extract_structured_low_quality_ocr_adds_quality_review_reason(monkeypatch) -> None:
    monkeypatch.setattr('app.extractors.pdf.tesseract_available', lambda: True)
    monkeypatch.setattr(
        'app.extractors.pdf.extract_best_ocr_result',
        lambda image: {
            'text': "Form W-2 Wage and Tax Statement 2024\nEmployee name John Doe\nEmployer name ACME Payroll\n1 Wages, tips, other compensation 1000.00",
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
        '/extract/structured',
        files={'file': ('ocr-w2.pdf', BytesIO(pdf_bytes), 'application/pdf')},
        data={'ocr_strategy': 'always'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'w2'
    assert payload['structured']['review']['requires_human_review'] is True
    assert 'low_ocr_quality_tax_document' in payload['structured']['review']['review_reasons']
    assert 'weak_ocr_evidence' in payload['structured']['review']['review_reasons']
    assert payload['raw_extraction']['extra']['ocr_quality_summary']['low_quality'] is True
    assert payload['raw_extraction']['extra']['ocr_evidence_snippets'][0]['page_number'] == 1



def test_extract_structured_ocr_backed_receipt_requires_review(monkeypatch) -> None:
    monkeypatch.setattr('app.extractors.image_ocr.tesseract_available', lambda: True)
    monkeypatch.setattr(
        'app.extractors.image_ocr.extract_best_ocr_result',
        lambda image: {
            'text': 'Coffee Shop\n03/14/2024\nSubtotal 10.00\nTax 0.80\nTotal 10.80',
            'score': 6.0,
            'selected_pass': 'soft_upscale',
            'processed_mode': 'L',
            'preprocessing': ['grayscale'],
            'ocr_passes': [],
        },
    )

    image_bytes = BytesIO()
    Image.new('RGB', (8, 8), color='white').save(image_bytes, format='PNG')
    image_bytes.seek(0)

    response = client.post(
        '/extract/structured',
        files={'file': ('receipt.png', image_bytes, 'image/png')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'receipt'
    assert payload['structured']['review']['requires_human_review'] is True
    assert 'ocr_backed_receipt' in payload['structured']['review']['review_reasons']
    assert 'low_ocr_quality_receipt' in payload['structured']['review']['review_reasons']
    assert 'weak_ocr_evidence' in payload['structured']['review']['review_reasons']
