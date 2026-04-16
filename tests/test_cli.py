"""CLI smoke: argparse + invocation of core via monkeypatch."""

from __future__ import annotations

import json
from pathlib import Path

from loci_extract import cli
from loci_extract.schema import Document, Extraction


def _stub_extraction():
    return Extraction(documents=[Document(
        document_type="W2",
        tax_year=2025,
        data={
            "employer": {"name": "Acme", "ein": "12-3456789", "address": "123 Main"},
            "employee": {"name": "Jane", "ssn_last4": "XXX-XX-1234", "address": "456 Elm"},
            "federal": {"box1_wages": 10000.0, "box2_federal_withheld": 1500.0,
                         "box3_ss_wages": 10000.0, "box4_ss_withheld": 620.0,
                         "box5_medicare_wages": 10000.0, "box6_medicare_withheld": 145.0},
            "box12": [], "box13": {}, "box14_other": [], "state": [], "local": [],
        },
    )])


def test_cli_json_to_stdout(monkeypatch, tmp_path, capsys):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF\n")
    monkeypatch.setattr(cli, "extract_document", lambda p, opts, progress_callback=None: _stub_extraction())
    rc = cli.main([str(pdf)])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["documents"][0]["document_type"] == "W2"


def test_cli_csv_to_file(monkeypatch, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF\n")
    out_path = tmp_path / "out.csv"
    monkeypatch.setattr(cli, "extract_document", lambda p, opts, progress_callback=None: _stub_extraction())
    rc = cli.main([str(pdf), "-o", str(out_path), "--format", "csv"])
    assert rc == 0
    body = out_path.read_text()
    assert body.startswith("document_type,")
    assert "Jane" in body


def test_cli_batch_directory(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.pdf").write_bytes(b"%PDF\n")
    (tmp_path / "b.pdf").write_bytes(b"%PDF\n")
    def fake_batch(paths, opts, progress_callback=None):
        return [(Path(p), _stub_extraction()) for p in paths]
    monkeypatch.setattr(cli, "extract_batch", fake_batch)
    rc = cli.main([str(tmp_path), "--batch"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert len(data["documents"]) == 2  # one per pdf


def test_cli_missing_file(monkeypatch, tmp_path, capsys):
    rc = cli.main([str(tmp_path / "nope.pdf")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err.lower()


def test_cli_detect_only(monkeypatch, tmp_path, capsys):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF\n")
    fake_result = {
        "document_type": "W2",
        "document_family": "tax",
        "confidence": 0.95,
        "strategy": "text",
        "encoding_broken": False,
        "strategy_reason": "text layer OK",
    }
    monkeypatch.setattr(cli, "detect_document", lambda p, opts, progress_callback=None: fake_result)
    rc = cli.main([str(pdf), "--detect-only"])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["document_type"] == "W2"
    assert parsed["document_family"] == "tax"
