"""Multi-section PDF boundary detection.

A single PDF often contains more than one logical document — e.g. a Balance
Sheet on pages 1-2 followed by an Income Statement on pages 3-9, or a
W-2 packet stapled to a 1099-NEC. ``detect_boundaries(pages)`` walks the
extracted-text-per-page list and returns one ``DocumentSection`` per detected
report, with start/end page ranges and a confidence score.

Each page's first ~20 lines are scanned for boundary signals. A "strong"
signal (>= 0.85 confidence) opens a new section. Weaker signals (>= 0.60)
contribute to the section title but don't open a new section on their own.

Per FINANCIAL_STATEMENTS_SPEC_V2.md §"Document Boundary Detection".
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class DocumentSection:
    start_page: int      # 0-indexed, inclusive
    end_page: int        # 0-indexed, inclusive
    document_type: str   # "BALANCE_SHEET" | "W2" | ... | "UNKNOWN"
    title: str           # First non-empty line of the section's start page
    confidence: float    # 0.0–1.0


# (regex, doc_type, confidence). Patterns run on the lowercase first ~20 lines
# of each page. Strong signals (>= 0.85) open a new section; weak signals
# (0.60) help select the doc_type when a strong signal is also present.
BOUNDARY_PATTERNS: list[tuple[str, str, float]] = [
    # Strong — explicit report headers (financial)
    (r"^balance sheet",                              "BALANCE_SHEET", 0.95),
    (r"^income statement",                           "INCOME_STATEMENT", 0.95),
    (r"^profit\s+(?:and\s+)?loss",                   "INCOME_STATEMENT", 0.95),
    (r"^trial balance",                              "TRIAL_BALANCE", 0.95),
    (r"^general ledger",                             "GENERAL_LEDGER", 0.95),
    (r"^accounts? receivable aging",                 "ACCOUNTS_RECEIVABLE_AGING", 0.95),
    (r"^accounts? payable aging",                    "ACCOUNTS_PAYABLE_AGING", 0.95),
    (r"^cash flow",                                  "CASH_FLOW_STATEMENT", 0.95),
    (r"^budget\s+vs\.?\s+actual",                    "BUDGET_VS_ACTUAL", 0.90),
    (r"^owner statement",                            "APPFOLIO_OWNER_STATEMENT", 0.90),
    (r"^reserve allocation",                         "RESERVE_ALLOCATION", 0.90),
    # Strong — tax forms
    (r"form\s+w-?2\b",                               "W2", 0.90),
    (r"wage and tax statement",                      "W2", 0.90),
    (r"form\s+1099.nec",                             "1099-NEC", 0.92),
    (r"form\s+1099.misc",                            "1099-MISC", 0.92),
    (r"form\s+1099.int",                             "1099-INT", 0.92),
    (r"form\s+1099.div",                             "1099-DIV", 0.92),
    (r"form\s+1099.b\b",                             "1099-B", 0.92),
    (r"form\s+1099.r\b",                             "1099-R", 0.92),
    (r"schedule\s+k.1.*1065",                        "K-1 1065", 0.92),
    (r"schedule\s+k.1.*1120.?s",                     "K-1 1120-S", 0.92),
    (r"schedule\s+k.1.*1041",                        "K-1 1041", 0.92),
    # Weaker
    (r"as of\s+\d{2}/\d{2}/\d{4}",                   "BALANCE_SHEET", 0.60),
    (r"for the (?:month|period|year)\s+(?:ended|ending)", "INCOME_STATEMENT", 0.60),
    (r"from\s+\d{2}/\d{2}/\d{4}\s+to\s+\d{2}/\d{2}/\d{4}", "INCOME_STATEMENT", 0.60),
]

# Strong-signal threshold — only signals at or above this open a new section.
STRONG_THRESHOLD = 0.85


def _scan_page_header(text: str) -> tuple[str | None, str, float]:
    """Scan the first ~20 lines of a page for boundary signals.

    Returns (document_type, title, confidence). If no strong signal hits,
    returns (None, "", 0.0)."""
    if not text:
        return None, "", 0.0
    first_lines = "\n".join(text.strip().split("\n")[:20]).lower()

    matched_type: str | None = None
    matched_conf = 0.0
    for pattern, doc_type, conf in BOUNDARY_PATTERNS:
        if conf > matched_conf and re.search(pattern, first_lines, re.MULTILINE):
            matched_type = doc_type
            matched_conf = conf

    title = ""
    for line in text.strip().split("\n"):
        s = line.strip()
        if s:
            title = s
            break
    return matched_type, title, matched_conf


def detect_boundaries(pages: list[dict]) -> list[DocumentSection]:
    """Walk per-page text and emit DocumentSections.

    ``pages`` is the extractor output: ``[{page: int, text: str, ...}, ...]``
    with 0-indexed page numbers in reading order.

    A new section opens when a page produces a strong signal (>= 0.85)
    different from the current section's type. The previous section is
    closed at the prior page. If no boundaries are found, returns one
    ``UNKNOWN`` section spanning all pages.
    """
    if not pages:
        return [DocumentSection(0, 0, "UNKNOWN", "", 0.0)]

    sections: list[DocumentSection] = []
    current_type: str | None = None
    current_title = ""
    current_conf = 0.0
    current_start = pages[0]["page"]

    for page_info in pages:
        page_num = page_info["page"]
        matched_type, matched_title, matched_conf = _scan_page_header(
            str(page_info.get("text") or "")
        )

        if (
            matched_type
            and matched_conf >= STRONG_THRESHOLD
            and matched_type != current_type
        ):
            if current_type is not None:
                sections.append(DocumentSection(
                    start_page=current_start,
                    end_page=page_num - 1,
                    document_type=current_type,
                    title=current_title,
                    confidence=current_conf,
                ))
            current_type = matched_type
            current_title = matched_title
            current_start = page_num
            current_conf = matched_conf

    # Close final section
    last_page = pages[-1]["page"]
    if current_type is not None:
        sections.append(DocumentSection(
            start_page=current_start,
            end_page=last_page,
            document_type=current_type,
            title=current_title,
            confidence=current_conf,
        ))

    if not sections:
        sections.append(DocumentSection(
            start_page=pages[0]["page"],
            end_page=last_page,
            document_type="UNKNOWN",
            title="",
            confidence=0.0,
        ))

    return sections


__all__ = ["DocumentSection", "BOUNDARY_PATTERNS", "STRONG_THRESHOLD", "detect_boundaries"]
