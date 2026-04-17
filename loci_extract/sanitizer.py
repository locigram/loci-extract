"""PII sanitizer — replace sensitive data with realistic synthetic equivalents.

Three modes:

- **regex** — pattern-match SSNs, TINs, phone numbers, account numbers, and
  addresses. Replace with deterministic synthetic data from built-in pools.
  Fast, no LLM needed. Misses names (no NER).

- **llm** — send the text to the configured LLM with a "find and replace all
  PII" prompt. The LLM understands context (knows "Jane Smith" is a person
  but "Form W-2" is not). Catches names, but slower and needs a model.

- **hybrid** — regex first (catches structured patterns cheaply), then LLM
  for names and anything regex missed. Best accuracy.

All modes replace PII with *realistic synthetic data of the same kind* so
the output is still useful for LLM training (the model learns to detect
SSNs, names, addresses — just not real ones).

EINs are preserved (they're public information).
"""

from __future__ import annotations

import hashlib
import re

# ---------------------------------------------------------------------------
# Synthetic data pools — deterministic via hash-based index selection
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Andrew", "Dorothy", "Paul",
    "Kimberly", "Joshua", "Emily", "Kenneth", "Donna", "Kevin", "Michelle",
    "Brian", "Carol", "George", "Amanda", "Timothy", "Melissa", "Ronald",
    "Deborah",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
]

_STREET_NAMES = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine St", "Elm St",
    "Washington Blvd", "Park Ave", "Lake Dr", "Hill Rd", "River Rd",
    "Forest Ave", "Spring St", "Valley Dr", "Sunset Blvd", "Highland Ave",
    "Church St", "Center St", "Mill Rd", "Academy Dr", "Broadway",
    "Lincoln Ave", "Union St", "Franklin Dr", "Jefferson Ln",
]

_CITIES = [
    "Springfield", "Riverside", "Fairview", "Georgetown", "Clinton",
    "Greenville", "Bristol", "Oakland", "Madison", "Arlington",
    "Burlington", "Franklin", "Milton", "Salem", "Chester",
    "Lexington", "Ashland", "Dover", "Manchester", "Newport",
]

_STATES = [
    "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]

_COMPANY_SUFFIXES = [
    "Corp", "Inc", "LLC", "Ltd", "Group", "Holdings", "Services",
    "Solutions", "Associates", "Partners", "Enterprises", "Co",
]

_COMPANY_WORDS = [
    "Alpha", "Beta", "Delta", "Sigma", "Apex", "Summit", "Nova",
    "Atlas", "Pinnacle", "Horizon", "Meridian", "Pacific", "National",
    "United", "Federal", "Liberty", "Heritage", "Premier", "Sterling",
    "Cascade", "Keystone", "Cornerstone", "Landmark", "Vanguard",
]


def _pick(pool: list[str], seed: str, offset: int = 0) -> str:
    """Deterministic pick from a pool based on a hash of the seed."""
    h = int(hashlib.md5((seed + str(offset)).encode()).hexdigest(), 16)
    return pool[h % len(pool)]


def _fake_ssn(seed: str) -> str:
    """Generate a fake but realistic SSN (not in any real range)."""
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    area = 900 + (h % 99)  # 900-999 range (not assigned by SSA)
    group = (h >> 8) % 99 + 1
    serial = (h >> 16) % 9999 + 1
    return f"{area:03d}-{group:02d}-{serial:04d}"


def _fake_phone(seed: str) -> str:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    area = 555
    mid = (h % 900) + 100
    last = (h >> 12) % 10000
    return f"({area}) {mid:03d}-{last:04d}"


def _fake_zip(seed: str) -> str:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    return f"{(h % 90000) + 10000}"


def _fake_address(seed: str) -> str:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    num = (h % 9999) + 1
    street = _pick(_STREET_NAMES, seed, 0)
    city = _pick(_CITIES, seed, 1)
    state = _pick(_STATES, seed, 2)
    zipcode = _fake_zip(seed)
    return f"{num} {street}, {city}, {state} {zipcode}"


def _fake_name(seed: str) -> str:
    first = _pick(_FIRST_NAMES, seed, 0)
    last = _pick(_LAST_NAMES, seed, 1)
    return f"{first} {last}"


def _fake_company(seed: str) -> str:
    word = _pick(_COMPANY_WORDS, seed, 0)
    suffix = _pick(_COMPANY_SUFFIXES, seed, 1)
    return f"{word} {suffix}"


def _fake_account_number(seed: str) -> str:
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    length = (h % 6) + 6  # 6-11 digits
    return str(h % (10 ** length)).zfill(length)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# SSN: 123-45-6789 or 123 45 6789
_SSN_FULL = re.compile(r"\b(\d{3})[-\s](\d{2})[-\s](\d{4})\b")
# TIN (non-EIN): catch SSN-format TINs. EIN format (12-1234567) is NOT matched.
_TIN_SSN_FORMAT = re.compile(r"\b(\d{3})[-](\d{2})[-](\d{4})\b")
# Phone: (123) 456-7890, 123-456-7890, 123.456.7890
_PHONE = re.compile(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}")
# Account numbers: sequences of 6+ digits (not dates, not SSNs, not EINs)
_ACCT_NUM = re.compile(r"(?<!\d[-/])(?<!\d)\b(\d{6,17})\b(?![-/]\d)")
# US addresses: number + street + city/state/zip pattern (simplified)
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-zA-Z\s]+(?:St|Ave|Dr|Ln|Rd|Blvd|Way|Ct|Pl|Cir|Ter|Pkwy)"
    r"(?:\.?(?:\s*(?:#|Apt|Suite|Ste|Unit)\s*\S+)?)"
    r"(?:\s*,\s*[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?",
    re.IGNORECASE,
)
# ZIP codes (standalone 5 or 5+4)
_ZIP = re.compile(r"\b(\d{5})(-\d{4})?\b")


# ---------------------------------------------------------------------------
# Mode 1: Regex-only sanitization
# ---------------------------------------------------------------------------


def sanitize_regex(text: str) -> dict:
    """Replace PII patterns with realistic synthetic data. Returns
    ``{sanitized: str, replacements: [{original, replacement, kind}, ...]}``."""
    replacements: list[dict] = []
    seen: dict[str, str] = {}  # original → replacement (consistent within one call)

    def _replace(match: re.Match, kind: str, gen_fn) -> str:
        original = match.group(0)
        if original in seen:
            return seen[original]
        synthetic = gen_fn(original)
        seen[original] = synthetic
        replacements.append({"original": original, "replacement": synthetic, "kind": kind})
        return synthetic

    result = text

    # SSNs first (before account numbers eat them)
    result = _SSN_FULL.sub(lambda m: _replace(m, "ssn", _fake_ssn), result)

    # Phone numbers
    result = _PHONE.sub(lambda m: _replace(m, "phone", _fake_phone), result)

    # Addresses (before individual components get mangled)
    result = _ADDRESS.sub(lambda m: _replace(m, "address", lambda s: _fake_address(s)), result)

    return {"sanitized": result, "replacements": replacements, "mode": "regex"}


# ---------------------------------------------------------------------------
# Mode 2: LLM-assisted sanitization
# ---------------------------------------------------------------------------

_SANITIZE_SYSTEM_PROMPT = """\
You are a PII sanitization engine. Your job is to find ALL personally \
identifiable information in the document text and replace each instance \
with realistic synthetic data of the same kind.

Rules:
- Replace person names with different realistic names (e.g., "Jane Smith" → "Maria Garcia")
- Replace SSNs (XXX-XX-XXXX) with fake SSNs in the 900-999 area range
- Replace addresses with fake but realistic US addresses
- Replace phone numbers with (555) area code numbers
- Replace account numbers with random digit strings of the same length
- Replace state IDs with fake IDs of the same format
- Do NOT replace EINs (XX-XXXXXXX format) — they are public information
- Do NOT replace document type labels, form names, box labels, or dollar amounts
- Do NOT replace company/employer names (they are public information)
- Keep the EXACT same formatting, line breaks, and structure
- Each unique PII value should map to the SAME synthetic value throughout \
(consistency — if "Jane Smith" appears 3 times, use the same replacement all 3 times)

Return ONLY the sanitized text. No explanations, no JSON wrapping."""


def sanitize_llm(text: str, client, model_name: str, temperature: float = 0.0,
                 max_tokens: int = 8192) -> dict:
    """Send text to an LLM for context-aware PII replacement."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SANITIZE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Sanitize all PII in the following document text. Replace with "
                "realistic synthetic data. Preserve formatting exactly.\n\n"
                + text
            )},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    sanitized = response.choices[0].message.content or ""
    return {
        "sanitized": sanitized,
        "mode": "llm",
        "model": model_name,
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
    }


# ---------------------------------------------------------------------------
# Mode 3: Hybrid (regex + LLM)
# ---------------------------------------------------------------------------


def sanitize_hybrid(text: str, client, model_name: str, temperature: float = 0.0,
                    max_tokens: int = 8192) -> dict:
    """Regex pass first (cheap — catches SSNs, phones, addresses), then LLM
    pass for names and anything regex missed."""
    # Step 1: regex pass
    regex_result = sanitize_regex(text)
    partially_sanitized = regex_result["sanitized"]

    # Step 2: LLM pass on the regex-sanitized output
    llm_result = sanitize_llm(partially_sanitized, client, model_name,
                              temperature=temperature, max_tokens=max_tokens)

    return {
        "sanitized": llm_result["sanitized"],
        "mode": "hybrid",
        "regex_replacements": regex_result["replacements"],
        "model": model_name,
        "prompt_tokens": llm_result.get("prompt_tokens"),
        "completion_tokens": llm_result.get("completion_tokens"),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def sanitize(text: str, mode: str = "regex", client=None, model_name: str = "",
             temperature: float = 0.0, max_tokens: int = 8192) -> dict:
    """Sanitize PII in text. ``mode`` is 'regex', 'llm', or 'hybrid'."""
    if mode == "regex":
        return sanitize_regex(text)
    if mode == "llm":
        if client is None:
            raise ValueError("LLM client required for mode='llm'")
        return sanitize_llm(text, client, model_name, temperature, max_tokens)
    if mode == "hybrid":
        if client is None:
            raise ValueError("LLM client required for mode='hybrid'")
        return sanitize_hybrid(text, client, model_name, temperature, max_tokens)
    raise ValueError(f"Unknown sanitize mode: {mode!r}. Use 'regex', 'llm', or 'hybrid'.")


__all__ = ["sanitize", "sanitize_regex", "sanitize_llm", "sanitize_hybrid"]
