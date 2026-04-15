"""PDF type detection (text-layer vs image) + document-type keyword hints.

Two responsibilities:

- ``detect_page_types(pdf_path)`` → ``{page_number: "text" | "image"}``
  Decides per page whether to route through the text-extractor (pdfminer) or
  through OCR / vision. Strips W-2-family boilerplate before the 100-char
  threshold so a form that prints only "Copy B" and "OMB No..." doesn't
  masquerade as a text page.

- ``identify_doc_types(text)`` → ``list[str]``
  Keyword-based multi-match. A single PDF often contains more than one
  document type (W-2 + 1099-NEC stapled together), so we return a list,
  not a single winner. The LLM makes the final call; this is just a hint
  for per-doc prompt augmentation.
"""

from __future__ import annotations

import re
from pathlib import Path

# Shared boilerplate we strip before the threshold check — these appear on
# nearly every tax form and shouldn't count as "meaningful text".
_BOILERPLATE_PATTERNS = [
    r"Copy [A-Z0-9]",
    r"OMB No\.?\s*\d+-\d+",
    r"Dept\.? of the Treasury",
    r"Internal Revenue Service",
    r"Department of the Treasury",
    r"Cat\.? No\.?\s*\w+",
    r"\bFor Official Use Only\b",
    r"\bVOID\b",
    r"\bCORRECTED\b",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), flags=re.IGNORECASE)

# Threshold below which a page is treated as image-only (no usable text layer).
_TEXT_PAGE_MIN_CHARS = 100


# ---------------------------------------------------------------------------
# Document-type keyword hints
# ---------------------------------------------------------------------------

# Ordered from most specific to least to avoid spurious matches. Each entry
# pairs a document type name with a set of anchor phrases (case-insensitive)
# that must appear contiguously or near-adjacent on the page.
_DOC_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("W2", ["wage and tax statement", "form w-2"]),
    ("1099-NEC", ["1099-nec", "nonemployee compensation"]),
    ("1099-MISC", ["1099-misc", "miscellaneous information", "miscellaneous income"]),
    ("1099-INT", ["1099-int", "interest income"]),
    ("1099-DIV", ["1099-div", "dividends and distributions"]),
    ("1099-B", ["1099-b", "proceeds from broker"]),
    ("1099-R", ["1099-r", "distributions from pensions", "annuities, retirement"]),
    ("1099-G", ["1099-g", "certain government payments"]),
    ("1099-SA", ["1099-sa", "distributions from an hsa", "distributions from hsa"]),
    ("1099-K", ["1099-k", "payment card and third party network"]),
    ("1099-S", ["1099-s", "proceeds from real estate"]),
    ("1099-C", ["1099-c", "cancellation of debt"]),
    ("1099-A", ["1099-a", "acquisition or abandonment"]),
    ("1098-T", ["1098-t", "tuition statement"]),
    ("1098-E", ["1098-e", "student loan interest"]),
    ("1098", ["1098", "mortgage interest statement"]),
    ("SSA-1099", ["ssa-1099", "social security benefit statement"]),
    ("RRB-1099", ["rrb-1099", "railroad retirement"]),
    ("K-1 1065", ["schedule k-1", "form 1065", "partner's share"]),
    ("K-1 1120-S", ["schedule k-1", "form 1120-s", "shareholder's share"]),
    ("K-1 1041", ["schedule k-1", "form 1041", "beneficiary's share"]),
]


# ---------------------------------------------------------------------------
# Page type detection
# ---------------------------------------------------------------------------


def _strip_boilerplate(text: str) -> str:
    return _BOILERPLATE_RE.sub("", text or "")


def _meaningful_char_count(text: str) -> int:
    stripped = _strip_boilerplate(text)
    # Collapse whitespace before counting so a page of boilerplate + blanks
    # doesn't squeak over the threshold.
    return len(re.sub(r"\s+", " ", stripped).strip())


def detect_page_types(pdf_path: str | Path) -> dict[int, str]:
    """Return ``{1: "text" | "image", 2: ..., ...}`` for each page.

    Opens the PDF with pdfminer, extracts text per page, strips boilerplate,
    and classifies as "image" when fewer than ``_TEXT_PAGE_MIN_CHARS`` chars
    remain. The result drives per-page routing in ``core.extract_document``.
    """
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

    # Extract once per page — pdfminer.extract_text with page_numbers param.
    # Use the same LAParams tuning the extractor module will use so the
    # threshold behavior is consistent.
    laparams = LAParams(line_margin=0.3, char_margin=2.0)

    # We don't know the page count up front without pdfminer — iterate until
    # extract_text returns empty (pdfminer raises on out-of-range pages in
    # some versions; we guard with a sanity limit).
    page_types: dict[int, str] = {}
    page_number = 0
    while True:
        page_number += 1
        try:
            text = extract_text(str(pdf_path), page_numbers=[page_number - 1], laparams=laparams)
        except Exception:
            break
        if text is None:
            break
        if page_number > 1 and text == "" and page_types:
            # pdfminer returns "" past the last page rather than raising.
            break
        is_text = _meaningful_char_count(text) >= _TEXT_PAGE_MIN_CHARS
        page_types[page_number] = "text" if is_text else "image"
        # Safety bail: runaway loop protection. Real PDFs top out at a few
        # hundred pages; we break at 1000 unconditionally.
        if page_number >= 1000:
            break

    return page_types


# ---------------------------------------------------------------------------
# Document-type hints
# ---------------------------------------------------------------------------


def identify_doc_types(text: str) -> list[str]:
    """Keyword-based doc-type hint. Returns every document type whose anchor
    phrases are ALL present in ``text`` (case-insensitive). Multi-match is
    deliberate — a stapled W-2 + 1099-NEC PDF should return both."""
    if not text:
        return []
    haystack = text.lower()
    matched: list[str] = []
    for doc_type, anchors in _DOC_TYPE_KEYWORDS:
        if doc_type in matched:
            continue
        if all(anchor.lower() in haystack for anchor in anchors):
            matched.append(doc_type)
    return matched


__all__ = ["detect_page_types", "identify_doc_types"]
