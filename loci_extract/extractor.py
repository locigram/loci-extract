"""pdfminer.six text extraction tuned for tax form layouts.

``extract_text_pages(pdf_path, page_numbers)`` returns ``{page: text}``
for the requested pages. LAParams are tuned for the tight-grid layouts
typical of IRS forms:

- ``line_margin=0.3`` — groups lines within a box together
- ``char_margin=2.0`` — keeps adjacent fields on one line rather than
  splitting them into separate text runs

These defaults match EXTRACT_SPEC.md. For 1099s and K-1s the defaults
are still a reasonable starting point; re-tune if field accuracy drops.
"""

from __future__ import annotations

from pathlib import Path

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

_FORM_LAYOUT_PARAMS = LAParams(line_margin=0.3, char_margin=2.0)


def extract_text_pages(pdf_path: str | Path, page_numbers: list[int]) -> dict[int, str]:
    """Extract text from the given 1-indexed ``page_numbers``.

    Returns a dict keyed by page number. Pages that yield no text still
    appear in the map with an empty string so callers can distinguish
    "asked-for page was empty" from "didn't ask".
    """
    if not page_numbers:
        return {}
    out: dict[int, str] = {}
    for page_number in page_numbers:
        try:
            text = extract_text(
                str(pdf_path),
                page_numbers=[page_number - 1],
                laparams=_FORM_LAYOUT_PARAMS,
            )
        except Exception:
            text = ""
        out[page_number] = (text or "").strip()
    return out


# ---------------------------------------------------------------------------
# pdfplumber path — coordinate-aware extraction for QBO / Sage / coordinate-placed text
# ---------------------------------------------------------------------------


def extract_with_strategy(pdf_path: str | Path, strategy: str) -> list[dict]:
    """Dispatch to the right text-extraction strategy.

    Returns ``list[dict]`` with shape ``[{page: int, text: str, tables: list | None}, ...]``
    where ``page`` is **0-indexed** in reading order. ``text``/``pdfplumber`` only;
    ``ocr`` and ``vision`` strategies are routed elsewhere (ocr.py / vision.py).
    """
    if strategy == "text":
        return _extract_pdfminer(pdf_path)
    if strategy == "pdfplumber":
        return _extract_pdfplumber(pdf_path)
    raise ValueError(
        f"extract_with_strategy got strategy={strategy!r}; OCR/vision must go through ocr.py / vision.py"
    )


def _extract_pdfminer(pdf_path: str | Path) -> list[dict]:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    results: list[dict] = []
    for i, page_layout in enumerate(extract_pages(str(pdf_path), laparams=_FORM_LAYOUT_PARAMS)):
        text = "".join(
            el.get_text() for el in page_layout if isinstance(el, LTTextContainer)
        )
        results.append({"page": i, "text": text, "tables": None})
    return results


def _extract_pdfplumber(pdf_path: str | Path) -> list[dict]:
    """Coordinate-aware extraction. Tries `extract_tables()` first, falls back
    to grouping `extract_words()` by y-position and sorting by x-position."""
    import pdfplumber

    results: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                })
            except Exception:
                tables = []
            if tables:
                text = _tables_to_text(tables)
                results.append({"page": i, "text": text, "tables": tables})
            else:
                try:
                    words = page.extract_words(
                        x_tolerance=5, y_tolerance=5, keep_blank_chars=False
                    )
                except Exception:
                    words = []
                text = _reconstruct_text_from_words(words)
                results.append({"page": i, "text": text, "tables": None})
    return results


def _reconstruct_text_from_words(words: list[dict]) -> str:
    """Group words into rows by y-position (3pt tolerance), then sort by x.

    Recovers reading order for coordinate-placed text that pdfminer's default
    extractor returns out of order (common on QBO / Sage browser exports)."""
    if not words:
        return ""
    rows: dict[int, list] = {}
    for w in words:
        y_key = round(w["top"] / 3) * 3
        rows.setdefault(y_key, []).append(w)
    lines = []
    for y_key in sorted(rows.keys()):
        row_words = sorted(rows[y_key], key=lambda w: w["x0"])
        lines.append("  ".join(w["text"] for w in row_words))
    return "\n".join(lines)


def _tables_to_text(tables: list) -> str:
    """Render extracted tables as space-separated text, one row per line."""
    lines = []
    for table in tables:
        for row in table:
            if row:
                lines.append("  ".join(str(c or "") for c in row))
    return "\n".join(lines)


__all__ = [
    "extract_text_pages",
    "extract_with_strategy",
]
