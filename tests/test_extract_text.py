from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_extract_text_file() -> None:
    response = client.post(
        '/extract',
        files={'file': ('hello.txt', b'hello\n\nworld', 'text/plain')},
        data={'include_chunks': 'true'},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['metadata']['source_type'] == 'text'
    assert 'hello' in payload['raw_text']
    assert len(payload['chunks']) >= 1
