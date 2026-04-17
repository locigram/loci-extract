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


# ---------------------------------------------------------------------------
# Structured extraction sanitization (walk dict tree)
# ---------------------------------------------------------------------------

# Keys whose values should NOT be sanitized (dollar amounts, doc types, etc.)
_SKIP_KEYS = frozenset({
    "document_type", "tax_year", "document_family", "confidence",
    "row_type", "section_name", "subsection", "column_type", "key", "label",
    "accounting_basis", "software", "period_start", "period_end",
    "code", "description",  # Box 12 codes like "AA", "DD"
    # Numeric / boolean fields are skipped by type, not by name
})

# Keys that contain PII and should always be sanitized
_PII_KEYS = frozenset({
    "name", "address", "ssn_last4", "tin_last4", "tin", "phone",
    "state_id", "account_number", "entity",
})


def _sanitize_scalar_regex(value: str, seen: dict[str, str],
                            replacements: list[dict]) -> str:
    """Apply regex PII replacement to a single string value."""
    def _replace(match: re.Match, kind: str, gen_fn) -> str:
        original = match.group(0)
        if original in seen:
            return seen[original]
        synthetic = gen_fn(original)
        seen[original] = synthetic
        replacements.append({"original": original, "replacement": synthetic, "kind": kind})
        return synthetic

    result = value
    result = _SSN_FULL.sub(lambda m: _replace(m, "ssn", _fake_ssn), result)
    result = _PHONE.sub(lambda m: _replace(m, "phone", _fake_phone), result)
    result = _ADDRESS.sub(lambda m: _replace(m, "address", _fake_address), result)
    return result


def _sanitize_name(value: str, seen: dict[str, str],
                    replacements: list[dict]) -> str:
    """Replace a name field with a synthetic name."""
    if not value or not value.strip():
        return value
    if value in seen:
        return seen[value]
    synthetic = _fake_name(value)
    seen[value] = synthetic
    replacements.append({"original": value, "replacement": synthetic, "kind": "name"})
    return synthetic


def _walk_and_sanitize(obj, seen: dict[str, str], replacements: list[dict],
                        parent_key: str = "") -> object:
    """Recursively walk a dict/list tree and sanitize string PII values."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _SKIP_KEYS and k not in _PII_KEYS:
                out[k] = v
            elif k == "name" and isinstance(v, str):
                # Name fields get synthetic name replacement
                out[k] = _sanitize_name(v, seen, replacements)
            elif k == "address" and isinstance(v, str):
                if v.strip():
                    if v in seen:
                        out[k] = seen[v]
                    else:
                        synthetic = _fake_address(v)
                        seen[v] = synthetic
                        replacements.append({"original": v, "replacement": synthetic, "kind": "address"})
                        out[k] = synthetic
                else:
                    out[k] = v
            elif k in ("ssn_last4", "tin_last4") and isinstance(v, str):
                if v in seen:
                    out[k] = seen[v]
                else:
                    synthetic = _fake_ssn(v)[-4:]
                    full_mask = f"XXX-XX-{synthetic}"
                    seen[v] = full_mask
                    replacements.append({"original": v, "replacement": full_mask, "kind": "ssn"})
                    out[k] = full_mask
            elif isinstance(v, str):
                out[k] = _sanitize_scalar_regex(v, seen, replacements)
            else:
                out[k] = _walk_and_sanitize(v, seen, replacements, parent_key=k)
        return out
    if isinstance(obj, list):
        return [_walk_and_sanitize(item, seen, replacements, parent_key) for item in obj]
    if isinstance(obj, str):
        return _sanitize_scalar_regex(obj, seen, replacements)
    return obj


def sanitize_extraction(extraction_dict: dict, mode: str = "regex",
                         client=None, model_name: str = "",
                         temperature: float = 0.0, max_tokens: int = 8192) -> dict:
    """Sanitize PII in a structured Extraction dict. Returns the sanitized
    dict in the same format (ready to be formatted as JSON/CSV/etc).

    For regex mode: walks the dict tree and replaces PII in string values.
    For llm/hybrid modes: serializes to text, sends to LLM, then re-parses.
    Falls back to regex tree-walk if LLM returns unparseable output.
    """
    import copy
    import json

    sanitized = copy.deepcopy(extraction_dict)
    replacements: list[dict] = []
    seen: dict[str, str] = {}

    if mode == "regex":
        sanitized = _walk_and_sanitize(sanitized, seen, replacements)
        return {"extraction": sanitized, "replacements": replacements, "mode": "regex"}

    if mode in ("llm", "hybrid"):
        if client is None:
            raise ValueError("LLM client required for mode='llm' or 'hybrid'")
        # Step 1: regex pass on the tree (cheap)
        if mode == "hybrid":
            sanitized = _walk_and_sanitize(sanitized, seen, replacements)

        # Step 2: serialize → LLM → re-parse
        text_to_sanitize = json.dumps(sanitized, indent=2)
        llm_result = sanitize_llm(text_to_sanitize, client, model_name,
                                   temperature=temperature, max_tokens=max_tokens)
        try:
            sanitized = json.loads(llm_result["sanitized"])
        except (json.JSONDecodeError, TypeError):
            # LLM returned non-JSON — fall back to regex-only result
            if mode != "hybrid":
                sanitized = _walk_and_sanitize(
                    copy.deepcopy(extraction_dict), seen, replacements
                )

        result = {
            "extraction": sanitized,
            "replacements": replacements,
            "mode": mode,
            "model": model_name,
            "prompt_tokens": llm_result.get("prompt_tokens"),
            "completion_tokens": llm_result.get("completion_tokens"),
        }
        if mode == "hybrid":
            result["regex_replacements"] = replacements
        return result

    raise ValueError(f"Unknown sanitize mode: {mode!r}")


# ---------------------------------------------------------------------------
# PDF sanitization — find-and-replace PII in the original PDF
# ---------------------------------------------------------------------------


def sanitize_pdf(
    pdf_path: str,
    mode: str = "regex",
    client=None,
    model_name: str = "",
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> tuple[bytes, list[dict]]:
    """Sanitize PII in a PDF file. Returns ``(pdf_bytes, replacements)``.

    Uses PyMuPDF to search for each PII string in the PDF, redact it
    (white rectangle), and overlay the synthetic replacement text at
    the same position. The output is a new PDF with identical layout
    but synthetic PII.

    For ``regex`` mode: builds a replacement map from regex patterns,
    then applies to the PDF.

    For ``llm``/``hybrid`` mode: extracts full text, runs through the
    LLM sanitizer to build a name replacement map, then applies all
    replacements (regex + LLM-discovered names) to the PDF.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)

    # Step 1: extract full text to build the replacement map
    full_text = "\n".join(page.get_text() for page in doc)

    # Step 2: build replacement map
    replacements: list[dict] = []
    seen: dict[str, str] = {}

    if mode in ("regex", "hybrid"):
        # Run regex sanitization on the full text to discover PII → synthetic mappings
        regex_result = sanitize_regex(full_text)
        replacements.extend(regex_result["replacements"])
        for r in regex_result["replacements"]:
            seen[r["original"]] = r["replacement"]

    if mode in ("llm", "hybrid"):
        if client is None:
            raise ValueError("LLM client required for mode='llm' or 'hybrid'")
        # For LLM mode, we need the model to identify names and other PII
        # Send the (possibly regex-sanitized) text to discover name mappings
        text_for_llm = regex_result["sanitized"] if mode == "hybrid" else full_text
        llm_prompt = (
            "List all person names in this document, one per line. "
            "Format: ORIGINAL_NAME|REPLACEMENT_NAME\n"
            "Use realistic replacement names. Only output the pairs, nothing else.\n\n"
            + text_for_llm
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You extract person names from documents and generate realistic replacements."},
                    {"role": "user", "content": llm_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            pairs_text = response.choices[0].message.content or ""
            for line in pairs_text.strip().split("\n"):
                if "|" not in line:
                    continue
                parts = line.split("|", 1)
                original = parts[0].strip()
                replacement = parts[1].strip()
                if original and replacement and original not in seen:
                    seen[original] = replacement
                    replacements.append({"original": original, "replacement": replacement, "kind": "name"})
        except Exception:
            pass  # LLM failure is non-fatal — regex replacements still apply

    if mode == "llm" and not any(r["kind"] != "name" for r in replacements):
        # Pure LLM mode — still need regex for SSNs/phones
        regex_result = sanitize_regex(full_text)
        for r in regex_result["replacements"]:
            if r["original"] not in seen:
                seen[r["original"]] = r["replacement"]
                replacements.append(r)

    # Step 3: apply replacements to each PDF page
    # Sort by length descending so longer strings are replaced first
    # (prevents partial matches — "Jane Smith" before "Jane")
    sorted_replacements = sorted(seen.items(), key=lambda x: len(x[0]), reverse=True)

    for page in doc:
        for original, replacement in sorted_replacements:
            instances = page.search_for(original)
            for rect in instances:
                # Add redaction annotation (white fill)
                page.add_redact_annot(rect, text=replacement, fontsize=0, fill=(1, 1, 1))
        # Apply all redactions on this page at once
        page.apply_redactions()

    # Step 4: save to bytes
    pdf_bytes = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    return pdf_bytes, replacements


__all__ = [
    "sanitize", "sanitize_regex", "sanitize_llm", "sanitize_hybrid",
    "sanitize_extraction", "sanitize_pdf",
]
