"""Document detection — extraction strategy + tax type + financial type + master detect().

Public surface (in dependency order):
  - detect_page_types(pdf_path) -> {page: "text" | "image"}     [legacy, pdfminer threshold]
  - identify_doc_types(text) -> list[str]                       [legacy multi-match keyword]
  - get_extraction_strategy(pdf_path) -> dict                   [v2 §Encoding Detection]
  - detect_tax_document_type(text) -> TaxDetectionResult        [TAX_DETECTION_SPEC]
  - detect_financial_document_type(text) -> str                 [v1 §FINANCIAL_SIGNATURES]
  - detect(pdf_path, text_sample) -> DocumentDetectionResult    [master entry point]

The legacy functions are preserved unchanged so the current ``core.py`` and
``test_detector.py`` continue to work. ``core.py`` will migrate to the master
``detect()`` in a separate refactor.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
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


# =============================================================================
# v2 + TAX_DETECTION_SPEC additions
# =============================================================================

# ── Encoding detection ──────────────────────────────────────────────────────


def _check_pdffonts(pdf_path: str | Path) -> list[str]:
    """Return list of font names with ``uni=no`` in pdffonts output. Empty
    when pdffonts is unavailable or all fonts are clean. A font with no
    ToUnicode map (Identity-H + no uni) means the text layer is glyph IDs,
    not Unicode codepoints — pdfminer extracts garbage. Trigger: route to OCR."""
    try:
        result = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    # The specific failure mode is Identity-H / Identity-V CID encoding with
    # no ToUnicode map. CID encodings use 2-byte glyph IDs that aren't standard
    # Unicode codepoints, so without a ToUnicode table pdfminer extracts
    # meaningless glyph indices. Single-byte encodings (MacRoman, WinAnsi,
    # etc.) with uni=no still decode correctly because the encoding name
    # itself maps to known characters.
    bad: list[str] = []
    for line in result.stdout.strip().split("\n")[2:]:
        if not re.search(r"\bIdentity-[HV]\b", line):
            continue
        tokens = line.split()
        if len(tokens) < 4:
            continue
        # pdffonts trailing columns are: ... emb sub uni object-ID(=2 tokens).
        # So uni sits at tokens[-4] when object-ID is "N M" (2 tokens), and
        # at tokens[-3] for single-token object IDs. Check both.
        uni_candidates = tokens[-4:-2] + tokens[-3:-2]
        if "no" in uni_candidates:
            bad.append(tokens[0])
    return bad


def _word_density_per_page(pdf_path: str | Path, maxpages: int = 3) -> float | None:
    """Ratio of word characters to total characters in the first N pages.
    Low density = lots of whitespace/padding = coordinate-placed text =
    pdfplumber will reconstruct columns better than pdfminer's reading order."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(str(pdf_path), maxpages=maxpages)
    except Exception:
        return None
    if not text:
        return None
    word_chars = sum(1 for c in text if c.isalnum())
    return word_chars / len(text)


_DOMAIN_ANCHORS = frozenset({
    "total", "balance", "account", "income", "expense", "asset",
    "wages", "tax", "amount", "date", "name", "net", "gross", "paid",
    "cash", "revenue", "debit", "credit", "equity", "liability",
})


def get_extraction_strategy(pdf_path: str | Path) -> dict:
    """Return the optimal extraction strategy for a PDF.

    Returns dict: ``{strategy, reason, encoding_broken}`` where
    ``strategy ∈ {"text", "pdfplumber", "ocr", "vision"}``.

    Order:
      1. pdffonts ToUnicode check (most reliable encoding indicator)
      2. Empty/short text layer → OCR (image-only PDF)
      3. High non-printable ratio → OCR (encoding corruption)
      4. No domain anchor words → OCR (text layer is garbage)
      5. Low word density → pdfplumber (coordinate-placed text)
      6. Otherwise → text (pdfminer)
    """
    pdf_path = str(pdf_path)

    # Step 1: pdffonts
    uni_missing = _check_pdffonts(pdf_path)
    if uni_missing:
        return {
            "strategy": "ocr",
            "reason": (
                f"Font(s) {uni_missing} have no ToUnicode map "
                f"(Identity-H, PScript5/Distiller workflow). "
                f"Text layer is glyph IDs. Must rasterize and OCR."
            ),
            "encoding_broken": True,
        }

    # Step 2-5: text-content checks
    try:
        from pdfminer.high_level import extract_text
        sample = extract_text(pdf_path, maxpages=2)
    except Exception as e:
        return {
            "strategy": "ocr",
            "reason": f"pdfminer failed: {e}",
            "encoding_broken": True,
        }

    if not sample or len(sample.strip()) < 100:
        return {
            "strategy": "ocr",
            "reason": "No text layer — image/scanned PDF.",
            "encoding_broken": False,
        }

    non_printable = sum(
        1 for c in sample
        if unicodedata.category(c) in ("Cc", "Cs", "Co", "Cn") and c not in "\n\t\r"
    )
    ratio = non_printable / max(len(sample), 1)
    if ratio > 0.15:
        return {
            "strategy": "ocr",
            "reason": (
                f"Non-printable character ratio {ratio:.1%} > 15%. "
                "Encoding corruption (ToUnicode missing but pdffonts unavailable)."
            ),
            "encoding_broken": True,
        }

    lower = sample.lower()
    found = sum(1 for w in _DOMAIN_ANCHORS if w in lower)
    if len(sample) > 300 and found < 2:
        return {
            "strategy": "ocr",
            "reason": (
                "Text layer present but no recognizable domain words. "
                "Likely encoding corruption."
            ),
            "encoding_broken": True,
        }

    word_density = _word_density_per_page(pdf_path)
    if word_density is not None and word_density < 0.4:
        return {
            "strategy": "pdfplumber",
            "reason": (
                f"Text layer OK but low word density ({word_density:.2f} word-chars/total). "
                "Likely coordinate-placed layout (QBO, Sage). Use pdfplumber."
            ),
            "encoding_broken": False,
        }

    return {"strategy": "text", "reason": "Text layer clean.", "encoding_broken": False}


# ── Tax document detection (TAX_DETECTION_SPEC.md) ──────────────────────────


@dataclass
class TaxDetectionResult:
    document_type: str          # "W2" | "1099-NEC" | ... | "K-1 1065" | "UNKNOWN"
    tax_year: int | None
    confidence: float           # 0.0–1.0
    issuer_software: str | None   # "ADP" | "Paychex" | "Intuit" | None
    is_summary_sheet: bool
    estimated_record_count: int      # employees/recipients in this PDF
    notes: list[str] = field(default_factory=list)


# Each entry: (regex, document_type, weight). Patterns run on lowercased text.
# Per-type weights accumulate; threshold is 0.5. Capped at 1.0.
TAX_SIGNATURES: list[tuple[str, str, float]] = [
    # ── W-2 ─────────────────────────────────────────────────────────────────
    (r"wage and tax statement",                     "W2", 0.9),
    (r"form\s+w-?2\b",                              "W2", 0.9),
    (r"box\s+1\b.*wages.*tips.*other\s+comp",       "W2", 0.8),
    (r"social security wages",                      "W2", 0.7),
    (r"medicare wages and tips",                    "W2", 0.7),
    (r"employer.s\s+(?:id|ein)\s+number",           "W2", 0.4),
    (r"employee.s\s+soc(?:ial)?\s*sec",             "W2", 0.4),
    (r"statutory\s+employee",                       "W2", 0.5),
    (r"third.party\s+sick\s+pay",                   "W2", 0.5),
    (r"w-?2\s+and\s+earnings\s+summary",            "W2", 0.95),
    (r"balancing\s+form\s+w-?2",                    "W2", 0.85),

    # ── 1099-NEC ─────────────────────────────────────────────────────────────
    (r"1099.nec\b",                                 "1099-NEC", 0.95),
    (r"nonemployee\s+compensation",                 "1099-NEC", 0.85),
    (r"form\s+1099.nec",                            "1099-NEC", 0.95),
    (r"box\s+1\b.*nonemployee",                     "1099-NEC", 0.90),

    # ── 1099-MISC ────────────────────────────────────────────────────────────
    (r"1099.misc\b",                                "1099-MISC", 0.95),
    (r"form\s+1099.misc",                           "1099-MISC", 0.95),
    (r"box\s+1\b.*rents\b",                         "1099-MISC", 0.85),
    (r"fishing\s+boat\s+proceeds",                  "1099-MISC", 0.90),
    (r"crop\s+insurance\s+proceeds",                "1099-MISC", 0.85),

    # ── 1099-INT ─────────────────────────────────────────────────────────────
    (r"1099.int\b",                                 "1099-INT", 0.95),
    (r"form\s+1099.int\b",                          "1099-INT", 0.95),
    (r"interest\s+income.*1099",                    "1099-INT", 0.80),
    (r"early\s+withdrawal\s+penalty",               "1099-INT", 0.70),
    (r"tax.exempt\s+interest",                      "1099-INT", 0.65),
    (r"bond\s+premium",                             "1099-INT", 0.60),

    # ── 1099-DIV ─────────────────────────────────────────────────────────────
    (r"1099.div\b",                                 "1099-DIV", 0.95),
    (r"form\s+1099.div\b",                          "1099-DIV", 0.95),
    (r"dividends\s+and\s+distributions",            "1099-DIV", 0.85),
    (r"qualified\s+dividends",                      "1099-DIV", 0.75),
    (r"total\s+ordinary\s+dividends",               "1099-DIV", 0.80),
    (r"section\s*199a\s+dividends",                 "1099-DIV", 0.80),

    # ── 1099-B ───────────────────────────────────────────────────────────────
    (r"1099.b\b",                                   "1099-B", 0.95),
    (r"form\s+1099.b\b",                            "1099-B", 0.95),
    (r"proceeds\s+from\s+broker",                   "1099-B", 0.85),
    (r"date\s+(?:acquired|sold)",                   "1099-B", 0.70),
    (r"cost\s+or\s+other\s+basis",                  "1099-B", 0.75),
    (r"wash\s+sale\s+loss\s+disallowed",            "1099-B", 0.85),
    (r"covered\s+security",                         "1099-B", 0.65),

    # ── 1099-R ───────────────────────────────────────────────────────────────
    (r"1099.r\b",                                   "1099-R", 0.95),
    (r"form\s+1099.r\b",                            "1099-R", 0.95),
    (r"distributions\s+from\s+pensions",            "1099-R", 0.85),
    (r"gross\s+distribution",                       "1099-R", 0.70),
    (r"taxable\s+amount\s+not\s+determined",        "1099-R", 0.80),
    (r"distribution\s+code",                        "1099-R", 0.70),
    (r"ira\s*/\s*sep\s*/\s*simple",                 "1099-R", 0.75),

    # ── 1099-G ───────────────────────────────────────────────────────────────
    (r"1099.g\b",                                   "1099-G", 0.95),
    (r"form\s+1099.g\b",                            "1099-G", 0.95),
    (r"unemployment\s+compensation",                "1099-G", 0.85),
    (r"state\s+or\s+local\s+income\s+tax\s+refund", "1099-G", 0.80),
    (r"taxable\s+grants",                           "1099-G", 0.75),
    (r"rtaa\s+payments",                            "1099-G", 0.85),

    # ── 1099-SA ──────────────────────────────────────────────────────────────
    (r"1099.sa\b",                                  "1099-SA", 0.95),
    (r"form\s+1099.sa\b",                           "1099-SA", 0.95),
    (r"distributions\s+from\s+an?\s+hsa",           "1099-SA", 0.85),
    (r"archer\s+msa",                               "1099-SA", 0.80),
    (r"medicare\s+advantage\s+msa",                 "1099-SA", 0.80),

    # ── 1099-K ───────────────────────────────────────────────────────────────
    (r"1099.k\b",                                   "1099-K", 0.95),
    (r"form\s+1099.k\b",                            "1099-K", 0.95),
    (r"payment\s+card\s+and\s+(?:third.party\s+)?network", "1099-K", 0.85),
    (r"merchant\s+category\s+code",                 "1099-K", 0.80),
    (r"gross\s+payment\s+card\s+transactions",      "1099-K", 0.85),

    # ── 1099-S ───────────────────────────────────────────────────────────────
    (r"1099.s\b",                                   "1099-S", 0.95),
    (r"form\s+1099.s\b",                            "1099-S", 0.95),
    (r"proceeds\s+from\s+real\s+estate",            "1099-S", 0.85),
    (r"date\s+of\s+closing",                        "1099-S", 0.75),
    (r"gross\s+proceeds.*real\s+estate",            "1099-S", 0.80),

    # ── 1099-C ───────────────────────────────────────────────────────────────
    (r"1099.c\b",                                   "1099-C", 0.95),
    (r"form\s+1099.c\b",                            "1099-C", 0.95),
    (r"cancellation\s+of\s+debt",                   "1099-C", 0.90),
    (r"amount\s+of\s+debt\s+discharged",            "1099-C", 0.85),
    (r"identifiable\s+event\s+code",                "1099-C", 0.80),

    # ── 1099-A ───────────────────────────────────────────────────────────────
    (r"1099.a\b",                                   "1099-A", 0.95),
    (r"form\s+1099.a\b",                            "1099-A", 0.95),
    (r"acquisition\s+or\s+abandonment",             "1099-A", 0.90),
    (r"balance\s+of\s+principal\s+outstanding",     "1099-A", 0.80),

    # ── 1098 (Mortgage Interest) ─────────────────────────────────────────────
    (r"\bform\s+1098\b(?!.(?:t|e|c|f|ma|q))",       "1098", 0.90),
    (r"mortgage\s+interest\s+statement",            "1098", 0.90),
    (r"mortgage\s+interest\s+received",             "1098", 0.85),
    (r"outstanding\s+mortgage\s+principal",         "1098", 0.75),
    (r"points\s+paid\s+on\s+purchase",              "1098", 0.70),
    (r"mortgage\s+insurance\s+premiums",            "1098", 0.65),

    # ── 1098-T ───────────────────────────────────────────────────────────────
    (r"1098.t\b",                                   "1098-T", 0.95),
    (r"form\s+1098.t\b",                            "1098-T", 0.95),
    (r"tuition\s+statement",                        "1098-T", 0.90),
    (r"qualified\s+tuition",                        "1098-T", 0.75),
    (r"at\s+least\s+half.time\s+student",           "1098-T", 0.80),
    (r"scholarships\s+or\s+grants",                 "1098-T", 0.65),

    # ── 1098-E ───────────────────────────────────────────────────────────────
    (r"1098.e\b",                                   "1098-E", 0.95),
    (r"form\s+1098.e\b",                            "1098-E", 0.95),
    (r"student\s+loan\s+interest\s+statement",      "1098-E", 0.90),
    (r"student\s+loan\s+interest\s+received",       "1098-E", 0.85),

    # ── SSA-1099 ─────────────────────────────────────────────────────────────
    (r"ssa.1099\b",                                 "SSA-1099", 0.95),
    (r"social\s+security\s+benefit\s+statement",    "SSA-1099", 0.90),
    (r"net\s+benefits\s+for\s+\d{4}",               "SSA-1099", 0.80),
    (r"benefit\s+statement.*social\s+security",     "SSA-1099", 0.85),

    # ── RRB-1099 ─────────────────────────────────────────────────────────────
    (r"rrb.1099\b",                                 "RRB-1099", 0.95),
    (r"railroad\s+retirement\s+(?:board|benefits)", "RRB-1099", 0.85),
    (r"tier\s+[12]\s+(?:tax|benefit)",              "RRB-1099", 0.75),

    # ── K-1 variants (use schema's "K-1 NNNN" string format) ────────────────
    (r"schedule\s+k.1.*1065",                       "K-1 1065", 0.95),
    (r"partner.s\s+share\s+of\s+income",            "K-1 1065", 0.85),
    (r"partnership.*schedule\s+k.?1",               "K-1 1065", 0.85),
    (r"partner.s\s+capital\s+account",              "K-1 1065", 0.75),
    (r"guaranteed\s+payments",                      "K-1 1065", 0.70),

    (r"schedule\s+k.1.*1120.?s",                    "K-1 1120-S", 0.95),
    (r"shareholder.s\s+share\s+of\s+income",        "K-1 1120-S", 0.85),
    (r"s\s+corporation.*schedule\s+k.?1",           "K-1 1120-S", 0.85),
    (r"shareholder.s\s+(?:pro.rata|percentage)",    "K-1 1120-S", 0.75),

    (r"schedule\s+k.1.*1041",                       "K-1 1041", 0.95),
    (r"beneficiary.s\s+share\s+of\s+income",        "K-1 1041", 0.85),
    (r"estate\s+or\s+trust.*schedule\s+k.?1",       "K-1 1041", 0.85),
    (r"fiduciary",                                  "K-1 1041", 0.55),
]


ISSUER_SIGNATURES: list[tuple[str, str]] = [
    (r"\badp\b",                                    "ADP"),
    (r"automatic\s+data\s+processing",              "ADP"),
    (r"w-?2\s+and\s+earnings\s+summary",            "ADP"),
    (r"balancing\s+form\s+w-?2",                    "ADP"),
    (r"\bpaychex\b",                                "Paychex"),
    (r"\bintuit\b",                                 "Intuit"),
    (r"\bquickbooks\b",                             "Intuit"),
    (r"\bgusto\b",                                  "Gusto"),
    (r"\bbamboohr\b",                               "BambooHR"),
    (r"\brippling\b",                               "Rippling"),
    (r"\bjustworks\b",                              "Justworks"),
    (r"\bturbotax\b",                               "TurboTax"),
    (r"\bhr\s*block\b",                             "H&R Block"),
    (r"\bfidelity\b",                               "Fidelity"),
    (r"\bvanguard\b",                               "Vanguard"),
    (r"\bcharles\s+schwab\b",                       "Schwab"),
    (r"\btd\s+ameritrade\b",                        "TD Ameritrade"),
    (r"\betrade\b",                                 "E*Trade"),
    (r"\bnavient\b",                                "Navient"),
    (r"\bgreat\s+lakes\b",                          "Great Lakes"),
    (r"\bnelnet\b",                                 "Nelnet"),
]


SUMMARY_SHEET_SIGNALS: list[str] = [
    r"total\s+employees",
    r"total\s+forms\s+(?:count|processed)",
    r"batch\s+no",
    r"for\s*:\s*batch",
    r"w-?2/w-?3\s+totals",
    r"balancing\s+form\s+w-?2",
    r"total\s+eforms",
    r"\btotals\b.*\bfor\b.*\bcompany\b",
]


def _detect_tax_year(text: str) -> int | None:
    """Extract tax year from header patterns; falls back to first 20XX integer."""
    for pattern in [
        r"(?:tax\s+year|year)\s+(\d{4})",
        r"wage\s+and\s+tax\s+statement\s+(\d{4})",
        r"form\s+w-?2\s+.*?(\d{4})",
        r"(\d{4})\s+form\s+w-?2",
        r"(?:for\s+)?(?:calendar\s+)?year\s+(\d{4})",
        r"(?:january|december)\s+\d{1,2},\s+(\d{4})",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            if 2010 <= year <= 2035:
                return year
    for m in re.finditer(r"\b(20\d{2})\b", text):
        year = int(m.group(1))
        if 2010 <= year <= 2035:
            return year
    return None


def _estimate_record_count(text: str, doc_type: str) -> int:
    """Estimate distinct records (employees/recipients) by SSN pattern count.
    W-2 SSNs typically appear 4x per record (Copy B/C/2/2) — divide raw count by 4."""
    if doc_type == "UNKNOWN":
        return 1
    ssn_matches = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)
    if not ssn_matches:
        return 1
    if doc_type == "W2":
        unique_ssns = len(set(ssn_matches))
        if unique_ssns == len(ssn_matches):
            return unique_ssns
        return max(1, round(len(ssn_matches) / 4))
    if doc_type.startswith("1099"):
        return max(1, len(set(ssn_matches)))
    return max(1, len(set(ssn_matches)))


def _runner_up(scores: dict, winner: str) -> str | None:
    remaining = {k: v for k, v in scores.items() if k != winner}
    if not remaining:
        return None
    return max(remaining, key=remaining.get)


def detect_tax_document_type(text: str) -> TaxDetectionResult:
    """Identify tax document type from extracted text via weighted regex scoring.

    Cumulative scoring: every matching pattern adds its weight to the doc_type's
    score, capped at 1.0. Threshold 0.5. Ambiguity resolvers handle 1098/T/E,
    NEC vs MISC (box 1 label decisive), and K-1 variants (highest score wins)."""
    lower = text.lower()

    scores: dict[str, float] = {}
    for pattern, doc_type, weight in TAX_SIGNATURES:
        if re.search(pattern, lower):
            scores[doc_type] = scores.get(doc_type, 0.0) + weight
    scores = {k: min(v, 1.0) for k, v in scores.items()}

    # Ambiguity resolvers
    for specific, general in [("1098-T", "1098"), ("1098-E", "1098")]:
        if scores.get(specific, 0) >= 0.7 and scores.get(general, 0) > 0:
            scores[general] = 0.0
    if re.search(r"nonemployee\s+compensation", lower):
        scores.pop("1099-MISC", None)
    if re.search(r"box\s*1\b.*rents\b", lower):
        scores.pop("1099-NEC", None)

    k1_variants = {k: v for k, v in scores.items() if k in ("K-1 1065", "K-1 1120-S", "K-1 1041")}
    if len(k1_variants) > 1:
        best_k1 = max(k1_variants, key=k1_variants.get)
        for k in k1_variants:
            if k != best_k1:
                scores[k] = 0.0

    if not scores or max(scores.values()) < 0.5:
        best_type, confidence = "UNKNOWN", 0.0
    else:
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

    tax_year = _detect_tax_year(text)
    issuer = next((name for pattern, name in ISSUER_SIGNATURES if re.search(pattern, lower)), None)
    is_summary = any(re.search(p, lower) for p in SUMMARY_SHEET_SIGNALS)
    record_count = _estimate_record_count(text, best_type)

    notes: list[str] = []
    if confidence < 0.7 and best_type != "UNKNOWN":
        notes.append(f"Low-confidence detection ({confidence:.0%}). Verify document type is {best_type}.")
    if is_summary:
        notes.append(
            "Employer/issuer summary sheet detected. "
            "Individual employee copies may be in a separate file."
        )
    if record_count > 1:
        notes.append(f"Estimated {record_count} records in this PDF.")
    runner = _runner_up(scores, best_type)
    if runner and scores.get(runner, 0) > 0.4:
        notes.append(f"Secondary match: {runner} ({scores[runner]:.0%}). Verify if extraction looks wrong.")

    return TaxDetectionResult(
        document_type=best_type,
        tax_year=tax_year,
        confidence=confidence,
        issuer_software=issuer,
        is_summary_sheet=is_summary,
        estimated_record_count=record_count,
        notes=notes,
    )


# ── Financial document detection (v1 §FINANCIAL_SIGNATURES) ─────────────────


FINANCIAL_SIGNATURES: dict[str, list[str]] = {
    "INCOME_STATEMENT_COMPARISON": [
        r"(?:ytd|year.to.date)\s+(?:actual|budget)",
        r"(?:current|this)\s+(?:month|period)\s+(?:actual|budget)",
        r"(?:prior year|last year)\s+(?:actual|budget)",
        r"\$\s+var(?:iance)?",
        r"%\s+var(?:iance)?",
        # QB Desktop 2-period comparison: "Jan - Dec 25" and "Jan - Dec 24"
        # with "$ Change" column heading.
        r"jan\s*-\s*dec\s*\d{2,4}",
        r"\$\s*change",
        # Two date columns next to each other in period-end format
        r"dec\s+\d{1,2},\s+\d{2,4}.*dec\s+\d{1,2},\s+\d{2,4}",
    ],
    "BUDGET_VS_ACTUAL": [
        r"annual budget",
        r"ytd budget",
        r"ytd actual",
        r"\$\s+var(?:iance)?",
    ],
    "ACCOUNTS_RECEIVABLE_AGING": [
        r"a(?:ccounts)?\s*r(?:eceivable)?\s*aging",
        r"current\s+(?:1-30|0-30)\s+(?:31-60|30-60)",
    ],
    "ACCOUNTS_PAYABLE_AGING": [
        r"a(?:ccounts)?\s*p(?:ayable)?\s*aging",
    ],
    "GENERAL_LEDGER": [
        r"general ledger",
        r"transaction detail",
        r"(?:date|memo|ref)\s+(?:debit|credit)\s+balance",
        # QB Desktop GL column header row:
        # "Type Date Num Adj Name Memo Split Debit Credit Balance"
        r"\btype\b.*\bdate\b.*\bnum\b.*\bsplit\b.*\bdebit\b.*\bcredit\b",
        r"\bmemo\b.*\bsplit\b.*\bdebit\b",
    ],
    "TRIAL_BALANCE": [
        r"trial balance",
        r"debit\s+credit",
        r"total debits",
    ],
    "RESERVE_ALLOCATION": [
        r"reserve allocation",
        r"reserve fund",
        r"(?:component|item)\s+(?:cost|balance|funded)",
        r"(?:tile|shake)\s+roof",
        r"asphalt replacement",
        r"contingency",
    ],
    "BALANCE_SHEET": [
        r"balance sheet",
        r"total assets",
        r"liabilities\s+[&and]+\s+(?:capital|equity|stockholders)",
        r"current assets",
        r"accounts receivable",
        # QB Desktop BS formatting
        r"total\s+liabilities\s*&\s*equity",
        r"total\s+current\s+assets",
        r"total\s+fixed\s+assets",
        r"total\s+other\s+assets",
    ],
    "INCOME_STATEMENT": [
        r"(?:income|profit)\s+(?:and\s+loss|statement|p&l)",
        r"total (?:income|revenue)",
        r"total expenses",
        r"net (?:income|loss|operating)",
        r"operating income",
        # QB Desktop P&L structure
        r"ordinary\s+income/expense",
        r"cost\s+of\s+goods\s+sold",
        r"gross\s+profit",
        r"net\s+ordinary\s+income",
        r"total\s+cogs",
    ],
}


# Priority: more specific types first so they shadow generic matches.
_FINANCIAL_PRIORITY = [
    "INCOME_STATEMENT_COMPARISON",
    "BUDGET_VS_ACTUAL",
    "ACCOUNTS_RECEIVABLE_AGING",
    "ACCOUNTS_PAYABLE_AGING",
    "GENERAL_LEDGER",
    "TRIAL_BALANCE",
    "RESERVE_ALLOCATION",
    "BALANCE_SHEET",
    "INCOME_STATEMENT",
]


def detect_financial_document_type(text: str) -> str:
    """Return the best-matching financial document type, or "FINANCIAL_UNKNOWN".

    Specialization rule: when a more specific variant scores >= 2 (multiple
    specialization signals), prefer it over its base type even if the base
    scored higher overall on generic structure words. This prevents QB 2-period
    P&Ls (which trigger both base INCOME_STATEMENT structure AND the
    COMPARISON period-column markers) from collapsing to the single-period
    schema."""
    text_lower = text.lower()
    scores = {
        doc_type: sum(1 for p in FINANCIAL_SIGNATURES.get(doc_type, []) if re.search(p, text_lower))
        for doc_type in _FINANCIAL_PRIORITY
    }

    # Specialization overrides: (specific, base) — specific wins when
    # specialization signals >= 2 AND base also matched.
    specializations = [
        ("INCOME_STATEMENT_COMPARISON", "INCOME_STATEMENT"),
        ("BUDGET_VS_ACTUAL", "INCOME_STATEMENT"),
    ]
    for specific, base in specializations:
        if (
            scores.get(specific, 0) >= 2
            and scores.get(base, 0) > 0
            and scores[specific] < scores[base]
        ):
            # Boost the specialization over the base.
            scores[specific] = scores[base] + 1

    best = max(scores, key=scores.get)
    if scores[best] >= 1:
        return best
    return "FINANCIAL_UNKNOWN"


# ── Master detect() ──────────────────────────────────────────────────────────


@dataclass
class DocumentDetectionResult:
    # Extraction strategy
    strategy: str                    # "text" | "pdfplumber" | "ocr" | "vision"
    encoding_broken: bool
    strategy_reason: str

    # Document identity
    document_type: str
    document_family: str             # "tax" | "financial_simple" | "financial_multi" | "financial_txn" | "financial_reserve" | "unknown"
    confidence: float

    # Tax-specific (None for financial)
    tax_year: int | None = None
    issuer_software: str | None = None
    is_summary_sheet: bool = False
    estimated_record_count: int = 1

    notes: list[str] = field(default_factory=list)


# Hardcoded fallback family map; the canonical map lives in prompts.py and is
# preferred when available. The fallback is only used at import time before
# prompts.py is loaded (or in standalone usage).
_FALLBACK_FAMILY_MAP: dict[str, str] = {
    "W2": "tax",
    "1099-NEC": "tax", "1099-MISC": "tax", "1099-INT": "tax", "1099-DIV": "tax",
    "1099-B": "tax", "1099-R": "tax", "1099-G": "tax", "1099-SA": "tax",
    "1099-K": "tax", "1099-S": "tax", "1099-C": "tax", "1099-A": "tax",
    "1098": "tax", "1098-T": "tax", "1098-E": "tax",
    "SSA-1099": "tax", "RRB-1099": "tax",
    "K-1 1065": "tax", "K-1 1120-S": "tax", "K-1 1041": "tax",
    "BALANCE_SHEET": "financial_simple",
    "INCOME_STATEMENT": "financial_simple",
    "TRIAL_BALANCE": "financial_simple",
    "INCOME_STATEMENT_COMPARISON": "financial_multi",
    "BUDGET_VS_ACTUAL": "financial_multi",
    "ACCOUNTS_RECEIVABLE_AGING": "financial_txn",
    "ACCOUNTS_PAYABLE_AGING": "financial_txn",
    "GENERAL_LEDGER": "financial_txn",
    "RESERVE_ALLOCATION": "financial_reserve",
}


def _resolve_family(doc_type: str) -> str:
    try:
        from loci_extract.prompts import DOCUMENT_FAMILY_MAP
        family = DOCUMENT_FAMILY_MAP.get(doc_type)
        if family is not None:
            return family.value if hasattr(family, "value") else family
    except ImportError:
        pass
    return _FALLBACK_FAMILY_MAP.get(doc_type, "unknown")


def detect(pdf_path: str | Path, text_sample: str) -> DocumentDetectionResult:
    """Master detection: extraction strategy + tax type → financial type → unknown.

    Called by core.py after a text sample is extracted (clean or OCR'd). Returns
    everything core.py needs to route the document to the right prompt, token
    budget, and schema."""
    strategy_info = get_extraction_strategy(pdf_path)

    # Tax detection first — IRS form numbers are stable
    tax_result = detect_tax_document_type(text_sample)
    if tax_result.document_type != "UNKNOWN" and tax_result.confidence >= 0.5:
        return DocumentDetectionResult(
            strategy=strategy_info["strategy"],
            encoding_broken=strategy_info["encoding_broken"],
            strategy_reason=strategy_info["reason"],
            document_type=tax_result.document_type,
            document_family=_resolve_family(tax_result.document_type),
            confidence=tax_result.confidence,
            tax_year=tax_result.tax_year,
            issuer_software=tax_result.issuer_software,
            is_summary_sheet=tax_result.is_summary_sheet,
            estimated_record_count=tax_result.estimated_record_count,
            notes=tax_result.notes,
        )

    # Financial detection second
    fin_type = detect_financial_document_type(text_sample)
    if fin_type != "FINANCIAL_UNKNOWN":
        return DocumentDetectionResult(
            strategy=strategy_info["strategy"],
            encoding_broken=strategy_info["encoding_broken"],
            strategy_reason=strategy_info["reason"],
            document_type=fin_type,
            document_family=_resolve_family(fin_type),
            confidence=0.8,
            notes=[],
        )

    # Unknown
    return DocumentDetectionResult(
        strategy=strategy_info["strategy"],
        encoding_broken=strategy_info["encoding_broken"],
        strategy_reason=strategy_info["reason"],
        document_type="UNKNOWN",
        document_family="unknown",
        confidence=0.0,
        notes=[
            "Could not identify document type from text content. "
            "Check OCR quality or try --vision flag."
        ],
    )


__all__ = [
    "detect_page_types",
    "identify_doc_types",
    "get_extraction_strategy",
    "detect_tax_document_type",
    "detect_financial_document_type",
    "detect",
    "TaxDetectionResult",
    "DocumentDetectionResult",
    "TAX_SIGNATURES",
    "ISSUER_SIGNATURES",
    "SUMMARY_SHEET_SIGNALS",
    "FINANCIAL_SIGNATURES",
]
