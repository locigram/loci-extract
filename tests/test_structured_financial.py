from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app.classification.rules import classify_document
from app.main import app
from app.schemas import DocumentMetadata, ExtractionMethod, ExtractionPayload, TextSegment
from app.structured.financial_statement import build_financial_statement_document


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
        'ASSETS',
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
    assert structured['fields']['line_items'][0]['section'] == 'Assets'
    assert structured['fields']['line_items'][1]['account_name'] == 'SUNWEST BANK-OPERATING'



def test_build_financial_statement_document_keeps_item_names_and_section_boundaries() -> None:
    raw_text = '\n'.join(
        [
            'Balance Sheet - PMG',
            'Properties: Niguel Villas Condominium Association',
            'As of: 01/31/2025',
            'Accounting Basis: Cash',
            'Account',
            'Number',
            '2024-0000',
            '2025-0000',
            '3003-0000',
            '3004-0000',
            '3005-0000',
            '3021-0000',
            'Account Name',
            'DUE TO/FROM',
            'DUE TO/FROM RESERVES',
            'Total DUE TO/FROM',
            'Liabilities',
            'PREPAID ASSESSMENTS',
            'Capital',
            'RESERVE ALLOCATION',
            'TERMITE CONTROL',
            'EQUITY',
            'Appfolio Opening Balance Equity',
            'Total EQUITY',
            'Balance',
            '100.00',
            '200.00',
            '300.00',
            '400.00',
            '500.00',
            '600.00',
        ]
    )
    payload = ExtractionPayload(
        document_id='doc-1',
        metadata=DocumentMetadata(
            filename='financials.pdf',
            mime_type='application/pdf',
            source_type='pdf',
            page_count=1,
            sheet_names=[],
            language=None,
        ),
        extraction=ExtractionMethod(extractor='pymupdf', ocr_used=False, status='success', warnings=[]),
        raw_text=raw_text,
        segments=[
            TextSegment(
                type='page',
                index=1,
                label='page-1',
                text=raw_text,
                metadata={'page_number': 1, 'source': 'parser'},
            )
        ],
        chunks=[],
        extra={'page_provenance': [{'page_number': 1, 'source': 'parser', 'has_text': True, 'text_length': len(raw_text)}]},
    )

    structured = build_financial_statement_document(payload)
    line_items = structured.fields['line_items']

    assert [item['account_name'] for item in line_items] == [
        'DUE TO/FROM RESERVES',
        'Total DUE TO/FROM',
        'PREPAID ASSESSMENTS',
        'TERMITE CONTROL',
        'Appfolio Opening Balance Equity',
        'Total EQUITY',
    ]
    assert [item['section'] for item in line_items] == [
        'Due To/From',
        'Due To/From',
        'Liabilities',
        'Reserve Allocation',
        'Equity',
        'Equity',
    ]
    assert [item['is_total'] for item in line_items] == [False, True, False, False, False, True]
    assert structured.fields['sections'] == [
        {'name': 'Due To/From', 'line_item_count': 2, 'total_line_item_count': 1},
        {'name': 'Liabilities', 'line_item_count': 1, 'total_line_item_count': 0},
        {'name': 'Reserve Allocation', 'line_item_count': 1, 'total_line_item_count': 0},
        {'name': 'Equity', 'line_item_count': 2, 'total_line_item_count': 1},
    ]
