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


__all__ = ["extract_text_pages"]
