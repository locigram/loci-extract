"""OpenAI-compatible LLM client + JSON parse with retry + SSN redaction.

Primary entry point:
    parse_extraction(client, text, *, system_prompt, model_name, ...) -> Extraction

The function handles:
- markdown fence stripping before ``json.loads``
- retry-on-invalid with the ValidationError fed back into the next prompt
- SSN redaction on the OUTPUT only (the model needs full text to locate
  recipients; redaction is a final step before we return or write to disk)

Design choice: we don't return the raw JSON; we return a validated ``Extraction``
pydantic model. Callers that need JSON should call ``.model_dump_json()`` on it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from loci_extract.schema import Extraction

logger = logging.getLogger("loci_extract.llm")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_client(model_url: str, api_key: str = "local"):
    """Build an openai.OpenAI client pointed at ``model_url``.

    ``api_key`` defaults to ``"local"`` because Ollama, vLLM, LM Studio, and
    llama.cpp all ignore auth on localhost — the client just has to send
    *something* non-empty.
    """
    import openai

    return openai.OpenAI(base_url=model_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Response scrubbing
# ---------------------------------------------------------------------------


_FENCE_START_RE = re.compile(r"^```(?:json)?\s*", flags=re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```\s*$")


def strip_code_fence(raw: str) -> str:
    """Remove ```json ... ``` markdown fences some models wrap JSON in."""
    if not raw:
        return ""
    cleaned = raw.strip()
    cleaned = _FENCE_START_RE.sub("", cleaned)
    cleaned = _FENCE_END_RE.sub("", cleaned)
    return cleaned.strip()


def extract_json_object(raw: str) -> str:
    """Best-effort: extract the outermost JSON object from mixed prose.

    Some models (especially the 7B/8B tier) emit a preamble before the JSON.
    Once the fence has been stripped, we look for the first ``{`` and match
    the outermost balanced braces. Quoted-string awareness keeps ``{`` and
    ``}`` inside strings from throwing off the depth counter.
    """
    if not raw:
        return ""
    cleaned = strip_code_fence(raw)
    start = cleaned.find("{")
    if start == -1:
        return cleaned  # let json.loads raise
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : idx + 1]
    return cleaned[start:]


# ---------------------------------------------------------------------------
# SSN/TIN redaction
# ---------------------------------------------------------------------------


_SSN_RE = re.compile(r"\b(\d{3})-(\d{2})-(\d{4})\b")
_EIN_RE = re.compile(r"\b(\d{2})-(\d{7})\b")


def _redact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        # Full 9-digit SSN → XXX-XX-1234
        value = _SSN_RE.sub(lambda m: f"XXX-XX-{m.group(3)}", value)
        # NOTE: EIN is NOT redacted. Employer EINs are public-ish; keeping
        # them makes downstream tax-prep export (Lacerte, TXF) work.
    return value


def redact_ssn_in_output(payload: Any) -> Any:
    """Walk a dict/list/primitive tree and mask any full SSN to last-4 form.

    Applied after schema validation, right before we return the Extraction
    or serialize to stdout/file. Keeps full SSNs out of outputs without
    blinding the LLM — the model sees the document's full text in order to
    locate the taxpayer in the first place.
    """
    if isinstance(payload, dict):
        return {k: redact_ssn_in_output(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [redact_ssn_in_output(v) for v in payload]
    return _redact_scalar(payload)


# ---------------------------------------------------------------------------
# Core parse-with-retry
# ---------------------------------------------------------------------------


def _call_chat(
    client,
    *,
    model_name: str,
    system_prompt: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_extraction(
    client,
    text: str,
    *,
    system_prompt: str,
    model_name: str = "local",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retry: int = 2,
    redact: bool = True,
) -> Extraction:
    """Call the LLM with ``text`` and return a validated ``Extraction``.

    On ``JSONDecodeError`` or ``ValidationError``, retries up to ``retry``
    times, feeding the error back into the next prompt. Raises the last
    error if we exhaust retries.
    """
    prompt = text
    last_error: Exception | None = None
    last_raw: str = ""

    for attempt in range(retry + 1):
        raw = _call_chat(
            client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_text=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        last_raw = raw
        try:
            json_str = extract_json_object(raw)
            parsed = json.loads(json_str)
            extraction = Extraction.model_validate(parsed)
            # Validate each per-type data payload too; this catches the
            # model producing a ``data`` block that doesn't match the
            # declared document_type.
            extraction.validate_all()
            if redact:
                redacted = redact_ssn_in_output(extraction.model_dump())
                extraction = Extraction.model_validate(redacted)
            return extraction
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt < retry:
                logger.warning(
                    "LLM output rejected (attempt %d/%d): %s", attempt + 1, retry + 1, _brief(exc)
                )
                prompt = (
                    f"{text}\n\n"
                    f"---\n"
                    f"Your previous response was invalid:\n{_brief(exc)}\n\n"
                    "Return ONLY a valid JSON object matching the schema. No prose, no markdown, "
                    "no backticks. Every field referenced in the schema must be present with the "
                    "correct type; use null/0.0 for missing values as instructed."
                )
                continue
    logger.error("LLM JSON parse failed after %d attempts. Last output:\n%s", retry + 1, last_raw[:2000])
    assert last_error is not None
    raise last_error


def _brief(exc: Exception) -> str:
    msg = str(exc)
    if len(msg) > 500:
        msg = msg[:500] + " … (truncated)"
    return msg


__all__ = [
    "make_client",
    "parse_extraction",
    "strip_code_fence",
    "extract_json_object",
    "redact_ssn_in_output",
]
