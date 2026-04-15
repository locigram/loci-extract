"""Tests for Tesseract-OSD-based page orientation correction.

No tesseract binary required — ``pytesseract.image_to_osd`` is stubbed.
These exercises confirm the wiring between ``extract_pages``,
``correct_orientation``, and the PNG re-save step so that
``fix_orientation=True`` actually rotates pages before OCR.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


def _make_image(size=(200, 100), color=(255, 255, 255)):
    return Image.new("RGB", size, color)


def test_correct_orientation_rotates_on_high_confidence():
    from loci_extract.ocr import correct_orientation

    img = _make_image(size=(300, 100))  # wider than tall
    stub_osd = {"rotate": 180, "orientation_conf": 3.5}
    with patch("pytesseract.image_to_osd", return_value=stub_osd):
        out, was_rotated = correct_orientation(img)
    assert was_rotated is True
    # 180° rotate preserves dimensions
    assert out.size == img.size


def test_correct_orientation_skips_low_confidence():
    from loci_extract.ocr import correct_orientation

    img = _make_image()
    stub_osd = {"rotate": 90, "orientation_conf": 0.5}  # below 1.5 threshold
    with patch("pytesseract.image_to_osd", return_value=stub_osd):
        out, was_rotated = correct_orientation(img)
    assert was_rotated is False
    assert out is img  # returns the original unchanged


def test_correct_orientation_skips_angle_zero():
    from loci_extract.ocr import correct_orientation

    img = _make_image()
    stub_osd = {"rotate": 0, "orientation_conf": 99.0}
    with patch("pytesseract.image_to_osd", return_value=stub_osd):
        _, was_rotated = correct_orientation(img)
    assert was_rotated is False


def test_correct_orientation_swallows_exceptions():
    from loci_extract.ocr import correct_orientation

    img = _make_image()
    with patch("pytesseract.image_to_osd", side_effect=RuntimeError("tesseract missing")):
        out, was_rotated = correct_orientation(img)
    assert was_rotated is False
    assert out is img


def test_extract_pages_fix_orientation_true_invokes_correction(tmp_path: Path):
    """When ``fix_orientation=True``, each rendered page PNG is passed through
    ``correct_orientation``, and rotated PNGs are saved back to disk before
    the OCR engine reads them. Stub OSD to claim every page is rotated."""
    from loci_extract import ocr as ocr_module

    fake_png_paths = {1: tmp_path / "p1.png", 2: tmp_path / "p2.png"}
    for p in fake_png_paths.values():
        _make_image(size=(100, 80)).save(p)

    with (
        patch.object(ocr_module, "select_engine", return_value=("tesseract", False)),
        patch.object(ocr_module, "_render_pdf_pages", return_value=fake_png_paths),
        patch.object(ocr_module, "_ocr_tesseract", return_value={1: "P1", 2: "P2"}),
        patch.object(
            ocr_module,
            "correct_orientation",
            side_effect=lambda img: (img.rotate(180, expand=True), True),
        ) as mock_correct,
    ):
        out = ocr_module.extract_pages(
            "/fake/doc.pdf", [1, 2], engine="tesseract", fix_orientation=True,
        )

    assert out == {1: "P1", 2: "P2"}
    assert mock_correct.call_count == 2  # invoked once per page


def test_extract_pages_fix_orientation_false_bypasses_correction(tmp_path: Path):
    """``fix_orientation=False`` must skip the correction step entirely."""
    from loci_extract import ocr as ocr_module

    fake_png_paths = {1: tmp_path / "p1.png"}
    _make_image().save(fake_png_paths[1])

    with (
        patch.object(ocr_module, "select_engine", return_value=("tesseract", False)),
        patch.object(ocr_module, "_render_pdf_pages", return_value=fake_png_paths),
        patch.object(ocr_module, "_ocr_tesseract", return_value={1: "ok"}),
        patch.object(ocr_module, "correct_orientation") as mock_correct,
    ):
        ocr_module.extract_pages(
            "/fake/doc.pdf", [1], engine="tesseract", fix_orientation=False,
        )

    mock_correct.assert_not_called()


def test_extract_pages_fix_orientation_default_is_true(tmp_path: Path):
    """Default behavior rotates — callers must opt out, not opt in."""
    from loci_extract import ocr as ocr_module

    fake_png_paths = {1: tmp_path / "p1.png"}
    _make_image().save(fake_png_paths[1])

    with (
        patch.object(ocr_module, "select_engine", return_value=("tesseract", False)),
        patch.object(ocr_module, "_render_pdf_pages", return_value=fake_png_paths),
        patch.object(ocr_module, "_ocr_tesseract", return_value={1: "ok"}),
        patch.object(
            ocr_module, "correct_orientation", return_value=(_make_image(), False),
        ) as mock_correct,
    ):
        ocr_module.extract_pages("/fake/doc.pdf", [1], engine="tesseract")

    mock_correct.assert_called_once()


def test_core_threads_fix_orientation_to_extract_pages(monkeypatch, tmp_path: Path):
    """``core._gather_page_text`` must pass ``opts.fix_orientation`` through
    to ``ocr.extract_pages``. Guards against regression after the plumbing
    fix (core.py was previously calling extract_pages with no flag)."""
    from loci_extract import core as core_module
    from loci_extract.core import ExtractionOptions

    pdf = tmp_path / "fake.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    captured: dict = {}

    def fake_ocr(pdf_path, pages, *, engine, gpu, dpi, fix_orientation):
        captured["fix_orientation"] = fix_orientation
        captured["engine"] = engine
        return dict.fromkeys(pages, "text")

    monkeypatch.setattr(core_module, "ocr_extract_pages", fake_ocr)
    monkeypatch.setattr(
        core_module,
        "detect_page_types",
        lambda _p: {1: "image"},
    )
    monkeypatch.setattr(
        "loci_extract.detector.get_extraction_strategy",
        lambda _p: {"encoding_broken": False, "reason": "text"},
    )

    opts = ExtractionOptions(
        model_url="http://x/v1", fix_orientation=False, ocr_engine="tesseract",
    )
    core_module._gather_page_text(pdf, opts, client=None, progress=None)

    assert captured["fix_orientation"] is False

    opts2 = ExtractionOptions(model_url="http://x/v1", ocr_engine="tesseract")  # default True
    core_module._gather_page_text(pdf, opts2, client=None, progress=None)
    assert captured["fix_orientation"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
