from __future__ import annotations

import re
from datetime import datetime

_AMOUNT_RE = re.compile(r"\(?\$?\s*-?[\d,]+(?:\.\d{2})?\)?")
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)


def normalize_whitespace(text: str | None) -> str:
    return " ".join((text or "").split())


def parse_amount(text: str | None) -> float | None:
    if not text:
        return None
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    token = match.group(0).strip()
    negative = token.startswith("(") and token.endswith(")")
    cleaned = token.replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def extract_last4(text: str | None) -> str | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits[-4:] if len(digits) >= 4 else None


def mask_identifier(text: str | None) -> str | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) < 4:
        return text
    last4 = digits[-4:]
    digit_index = 0
    chars: list[str] = []
    reveal_from = len(digits) - 4
    for char in text:
        if char.isdigit():
            chars.append(last4[digit_index - reveal_from] if digit_index >= reveal_from else "*")
            digit_index += 1
        else:
            chars.append(char)
    return "".join(chars)


def find_first_date(text: str) -> str | None:
    patterns = (
        (r"\b(\d{1,2}/\d{1,2}/\d{4})\b", "%m/%d/%Y"),
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (rf"\b(({_MONTH_NAMES})\s+\d{{1,2}},\s*\d{{4}})\b", None),
    )
    for pattern, fmt in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        if fmt:
            try:
                return datetime.strptime(token, fmt).date().isoformat()
            except ValueError:
                continue
        for month_fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(token, month_fmt).date().isoformat()
            except ValueError:
                continue
    return None
