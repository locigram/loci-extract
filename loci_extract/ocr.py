"""OCR engine abstraction — tesseract / easyocr / paddleocr.

Auto-selection picks the best-available backend: CUDA → easyocr, MPS → easyocr,
otherwise tesseract. ``easyocr`` and ``paddleocr`` imports are guarded so a
base install (without ``[ocr]`` extras) still works with tesseract only.

Public API:
    select_engine(engine, gpu) -> (engine_name, use_gpu)
    extract_pages(pdf_path, pages, *, engine, gpu, dpi) -> {page: text}
    available_engines() -> {"tesseract": bool, "easyocr": bool, "paddleocr": bool}
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger("loci_extract.ocr")

EngineName = Literal["auto", "tesseract", "easyocr", "paddleocr"]
GpuSetting = Literal["auto", "true", "false"]


# ---------------------------------------------------------------------------
# Availability probing
# ---------------------------------------------------------------------------


def _tesseract_available() -> bool:
    try:
        from shutil import which

        import pytesseract  # noqa: F401

        return which("tesseract") is not None
    except ImportError:
        return False


def _easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False


def _paddleocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def available_engines() -> dict[str, bool]:
    return {
        "tesseract": _tesseract_available(),
        "easyocr": _easyocr_available(),
        "paddleocr": _paddleocr_available(),
    }


def _resolve_gpu(gpu: GpuSetting) -> bool:
    if gpu == "true":
        return True
    if gpu == "false":
        return False
    return _cuda_available() or _mps_available()


def select_engine(engine: EngineName, gpu: GpuSetting) -> tuple[str, bool]:
    """Return (engine_name, use_gpu) after auto-detection.

    ``auto`` picks easyocr when any accelerator is available, otherwise
    falls back to tesseract. If the chosen engine is unavailable, falls
    through to the best-available.
    """
    use_gpu = _resolve_gpu(gpu)
    if engine == "auto":
        if (use_gpu or _cuda_available() or _mps_available()) and _easyocr_available():
            return "easyocr", use_gpu
        if _tesseract_available():
            return "tesseract", False
        if _paddleocr_available():
            return "paddleocr", use_gpu
        raise RuntimeError(
            "No OCR engine is available. Install system tesseract or `pip install .[ocr]` for easyocr/paddleocr."
        )
    # Explicit engine requested — verify or fall through.
    if engine == "tesseract" and _tesseract_available():
        return "tesseract", False
    if engine == "easyocr" and _easyocr_available():
        return "easyocr", use_gpu
    if engine == "paddleocr" and _paddleocr_available():
        return "paddleocr", use_gpu
    logger.warning("Requested OCR engine %r is not available; falling back to auto-select.", engine)
    return select_engine("auto", gpu)


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------


def _render_pdf_pages(pdf_path: Path, pages: list[int], dpi: int, output_dir: Path) -> dict[int, Path]:
    """Render requested 1-indexed pages as PNGs. Returns {page: png_path}."""
    from pdf2image import convert_from_path

    rendered: dict[int, Path] = {}
    for page_number in pages:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            output_folder=str(output_dir),
        )
        if not images:
            continue
        out_path = output_dir / f"page_{page_number}.png"
        images[0].save(out_path, format="PNG")
        rendered[page_number] = out_path
    return rendered


# ---------------------------------------------------------------------------
# Engine implementations
# ---------------------------------------------------------------------------


def _ocr_tesseract(png_paths: dict[int, Path]) -> dict[int, str]:
    import pytesseract
    from PIL import Image

    # --psm 6: Assume a single uniform block of text. --oem 3: default engine.
    config = "--psm 6 --oem 3"
    out: dict[int, str] = {}
    for page, png in png_paths.items():
        try:
            with Image.open(png) as img:
                out[page] = pytesseract.image_to_string(img, config=config) or ""
        except Exception as exc:
            logger.warning("Tesseract failed on page %d: %s", page, exc)
            out[page] = ""
    return out


# Cached readers to avoid reloading models for every call.
_easyocr_reader = None
_paddleocr_reader = None


def _ocr_easyocr(png_paths: dict[int, Path], use_gpu: bool) -> dict[int, str]:
    global _easyocr_reader
    import easyocr

    if _easyocr_reader is None:
        logger.info("Loading EasyOCR reader (gpu=%s) — may download models on first run.", use_gpu)
        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    reader = _easyocr_reader

    out: dict[int, str] = {}
    for page, png in png_paths.items():
        try:
            results = reader.readtext(str(png), detail=1)
        except Exception as exc:
            logger.warning("EasyOCR failed on page %d: %s", page, exc)
            out[page] = ""
            continue
        if results:
            confidences = [r[2] for r in results if len(r) >= 3]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            if avg_conf < 0.6:
                logger.warning("Page %d: low OCR confidence (%.2f)", page, avg_conf)
            out[page] = "\n".join(r[1] for r in results)
        else:
            out[page] = ""
    return out


def _ocr_paddleocr(png_paths: dict[int, Path], use_gpu: bool) -> dict[int, str]:
    global _paddleocr_reader
    from paddleocr import PaddleOCR

    if _paddleocr_reader is None:
        logger.info("Loading PaddleOCR reader (gpu=%s)", use_gpu)
        _paddleocr_reader = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )
    reader = _paddleocr_reader

    out: dict[int, str] = {}
    for page, png in png_paths.items():
        try:
            result = reader.ocr(str(png), cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR failed on page %d: %s", page, exc)
            out[page] = ""
            continue
        if result and result[0]:
            out[page] = "\n".join(line[1][0] for line in result[0] if line and len(line) >= 2)
        else:
            out[page] = ""
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_pages(
    pdf_path: str | Path,
    pages: list[int],
    *,
    engine: EngineName = "auto",
    gpu: GpuSetting = "auto",
    dpi: int = 300,
) -> dict[int, str]:
    """OCR the given 1-indexed ``pages``. Returns ``{page: text}``.

    Resolves the engine via :func:`select_engine`, renders the requested pages
    to PNG at the requested DPI (under a temp dir that's wiped after), and
    dispatches to the chosen engine. Empty-text pages still appear in the
    returned map.
    """
    if not pages:
        return {}
    resolved_engine, use_gpu = select_engine(engine, gpu)
    logger.info("OCR engine selected: %s (gpu=%s)", resolved_engine, use_gpu)

    with tempfile.TemporaryDirectory(prefix="loci-extract-ocr-") as tmp:
        tmp_dir = Path(tmp)
        png_paths = _render_pdf_pages(Path(pdf_path), pages, dpi=dpi, output_dir=tmp_dir)
        if not png_paths:
            return dict.fromkeys(pages, "")
        if resolved_engine == "tesseract":
            out = _ocr_tesseract(png_paths)
        elif resolved_engine == "easyocr":
            out = _ocr_easyocr(png_paths, use_gpu)
        elif resolved_engine == "paddleocr":
            out = _ocr_paddleocr(png_paths, use_gpu)
        else:
            raise RuntimeError(f"Unsupported OCR engine {resolved_engine!r}")
    # Normalize to include every requested page (empty string if OCR skipped it).
    return {page: out.get(page, "") for page in pages}


# ---------------------------------------------------------------------------
# Orientation correction (Tesseract OSD)
# ---------------------------------------------------------------------------


def correct_orientation(img):
    """Detect and correct rotation via Tesseract OSD.

    Returns ``(corrected_image, was_rotated: bool)``. On any failure or when
    Tesseract reports angle=0 / low confidence, returns the original image
    with ``was_rotated=False``. PIL rotate is counter-clockwise; Tesseract
    OSD reports the clockwise rotation needed, so we negate the angle."""
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0))
        confidence = float(osd.get("orientation_conf", 0))
        if confidence > 1.5 and angle != 0:
            corrected = img.rotate(-angle, expand=True)
            return corrected, True
    except Exception as exc:
        logger.debug("Orientation detection failed: %s", exc)
    return img, False


def extract_pages_detailed(
    pdf_path: str | Path,
    pages: list[int],
    *,
    engine: EngineName = "auto",
    gpu: GpuSetting = "auto",
    dpi: int = 300,
    fix_orientation: bool = True,
) -> list[dict]:
    """OCR the given pages and return per-page results with provenance.

    Returns ``[{page: int, text: str, rotated: bool, confidence: float | None}, ...]``.
    Like :func:`extract_pages` but with rotation fix + per-page confidence
    tracking. Use this when downstream code needs to populate
    ``FinancialMetadata.pages_rotated`` / ``ocr_confidence`` /
    ``ocr_low_confidence_pages``.
    """
    if not pages:
        return []
    resolved_engine, use_gpu = select_engine(engine, gpu)
    logger.info(
        "OCR (detailed) engine selected: %s (gpu=%s, fix_orientation=%s)",
        resolved_engine, use_gpu, fix_orientation,
    )
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="loci-extract-ocr-") as tmp:
        tmp_dir = Path(tmp)
        png_paths = _render_pdf_pages(Path(pdf_path), pages, dpi=dpi, output_dir=tmp_dir)
        if not png_paths:
            return [{"page": p, "text": "", "rotated": False, "confidence": None} for p in pages]

        rotated_map: dict[int, bool] = {}
        if fix_orientation:
            # Re-save rotated PNGs so the engine ingests the corrected image.
            for page, png in png_paths.items():
                with Image.open(png) as img:
                    corrected, was_rotated = correct_orientation(img)
                    if was_rotated:
                        corrected.save(png)
                    rotated_map[page] = was_rotated

        if resolved_engine == "tesseract":
            text_map = _ocr_tesseract(png_paths)
            confidences = {p: None for p in png_paths}  # tesseract path doesn't track conf
        elif resolved_engine == "easyocr":
            text_map, confidences = _ocr_easyocr_with_confidence(png_paths, use_gpu)
        elif resolved_engine == "paddleocr":
            text_map, confidences = _ocr_paddleocr_with_confidence(png_paths, use_gpu)
        else:
            raise RuntimeError(f"Unsupported OCR engine {resolved_engine!r}")

    out: list[dict] = []
    for p in pages:
        out.append({
            "page": p,
            "text": text_map.get(p, ""),
            "rotated": rotated_map.get(p, False),
            "confidence": confidences.get(p),
        })
    return out


def _ocr_easyocr_with_confidence(
    png_paths: dict[int, Path], use_gpu: bool
) -> tuple[dict[int, str], dict[int, float | None]]:
    global _easyocr_reader
    import easyocr

    if _easyocr_reader is None:
        logger.info("Loading EasyOCR reader (gpu=%s)", use_gpu)
        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    reader = _easyocr_reader

    texts: dict[int, str] = {}
    confs: dict[int, float | None] = {}
    for page, png in png_paths.items():
        try:
            results = reader.readtext(str(png), detail=1)
        except Exception as exc:
            logger.warning("EasyOCR failed on page %d: %s", page, exc)
            texts[page] = ""
            confs[page] = None
            continue
        if results:
            confidences = [r[2] for r in results if len(r) >= 3]
            avg = sum(confidences) / len(confidences) if confidences else 0.0
            texts[page] = "\n".join(r[1] for r in results)
            confs[page] = avg
        else:
            texts[page] = ""
            confs[page] = None
    return texts, confs


def _ocr_paddleocr_with_confidence(
    png_paths: dict[int, Path], use_gpu: bool
) -> tuple[dict[int, str], dict[int, float | None]]:
    global _paddleocr_reader
    from paddleocr import PaddleOCR

    if _paddleocr_reader is None:
        logger.info("Loading PaddleOCR reader (gpu=%s)", use_gpu)
        _paddleocr_reader = PaddleOCR(
            use_angle_cls=True, lang="en", use_gpu=use_gpu, show_log=False
        )
    reader = _paddleocr_reader

    texts: dict[int, str] = {}
    confs: dict[int, float | None] = {}
    for page, png in png_paths.items():
        try:
            result = reader.ocr(str(png), cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR failed on page %d: %s", page, exc)
            texts[page] = ""
            confs[page] = None
            continue
        if result and result[0]:
            lines = result[0]
            confidences = [line[1][1] for line in lines if line and len(line) >= 2]
            texts[page] = "\n".join(
                line[1][0] for line in lines if line and len(line) >= 2
            )
            confs[page] = (
                sum(confidences) / len(confidences) if confidences else None
            )
        else:
            texts[page] = ""
            confs[page] = None
    return texts, confs


__all__ = [
    "EngineName",
    "GpuSetting",
    "available_engines",
    "select_engine",
    "extract_pages",
    "extract_pages_detailed",
    "correct_orientation",
]
