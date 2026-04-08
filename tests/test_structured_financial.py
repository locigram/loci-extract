from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app.classification.rules import classify_document
from app.main import app


client = TestClient(app)


def test_classify_financial_statement_from_balance_sheet_text() -> None:
    result = classify_document(
        filename='financials.pdf',
        mime_type='application/pdf',
        raw_text='Balance Sheet Account Number Account Name Accounting Basis: Cash Liabilities & Capital',
    )
    assert result.doc_type == 'financial_statement'
    assert 'balance sheet' in result.matched_signals



def test_extract_structured_financial_statement_from_balance_sheet_pdf() -> None:
    document = fitz.open()
    page = document.new_page()
    lines = [
        'Balance Sheet - PMG',
        'Properties: Niguel Villas Condominium Association',
        'As of: 01/31/2025',
        'Accounting Basis: Cash',
        'Account',
        'Number',
        '1018-0000',
        '1021-0000',
        'Account Name',
        'Cash',
        'SUNWEST BANK-OPERATING',
        'Balance',
        '158,678.65',
        '328,402.40',
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 14
    pdf_bytes = document.tobytes()

    response = client.post(
        '/extract/structured',
        files={'file': ('financial-statement.pdf', BytesIO(pdf_bytes), 'application/pdf')},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['classification']['doc_type'] == 'financial_statement'
    structured = payload['structured']
    assert structured['document_type'] == 'financial_statement'
    assert structured['fields']['report_type'] == 'balance_sheet'
    assert structured['fields']['organization_name'] == 'Niguel Villas Condominium Association'
    assert structured['fields']['statement_date'] == '2025-01-31'
    assert structured['fields']['accounting_basis'] == 'Cash'
    assert len(structured['fields']['line_items']) == 2
    assert structured['fields']['line_items'][0]['account_number'] == '1018-0000'
    assert structured['fields']['line_items'][0]['account_name'] == 'Cash'
    assert structured['fields']['line_items'][0]['balance'] == 158678.65
    assert structured['fields']['line_items'][1]['account_name'] == 'SUNWEST BANK-OPERATING'
