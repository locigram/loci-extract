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


# ---------------------------------------------------------------------------
# Token budgets + finish_reason="length" auto-bump retry
# ---------------------------------------------------------------------------


# Per-doc-type response token budget. Picked to comfortably fit the typical
# JSON for that doc type + headroom. Multi-column / GL / K-1 are intentionally
# generous; the ``call_llm_raw`` retry below auto-bumps another 1.5x on
# finish_reason=="length" if the model still truncates.
TOKEN_BUDGETS: dict[str, int] = {
    # Tax forms
    "W2": 2048,
    "1099-NEC": 1024,
    "1099-MISC": 1024,
    "1099-INT": 1024,
    "1099-DIV": 1024,
    "1099-B": 4096,   # transaction list
    "1099-R": 1024,
    "1099-G": 1024,
    "1099-SA": 1024,
    "1099-K": 1024,
    "1099-S": 1024,
    "1099-C": 1024,
    "1099-A": 1024,
    "1098": 1024,
    "1098-T": 1024,
    "1098-E": 1024,
    "SSA-1099": 1024,
    "RRB-1099": 1024,
    "K-1 1065": 3000,
    "K-1 1120-S": 3000,
    "K-1 1041": 3000,
    # Financial — single period
    "BALANCE_SHEET": 4096,
    "INCOME_STATEMENT": 4096,
    "CASH_FLOW_STATEMENT": 3000,
    "TRIAL_BALANCE": 6000,
    # Financial — multi-column. These reports (12-month P&L, Budget vs Actual
    # with $/% variance columns) can have 200+ accounts × 10+ columns. Need
    # generous output budgets; finish_reason=length retry bumps 1.5x further.
    "INCOME_STATEMENT_COMPARISON": 24000,
    "BUDGET_VS_ACTUAL": 24000,
    "QB_PROFIT_LOSS": 16000,
    # Financial — transaction-level (use chunking)
    "GENERAL_LEDGER": 8000,
    "ACCOUNTS_RECEIVABLE_AGING": 4096,
    "ACCOUNTS_PAYABLE_AGING": 4096,
    "QB_GENERAL_LEDGER": 8000,
    # Reserve
    "RESERVE_ALLOCATION": 6000,
    # Unknown
    "FINANCIAL_UNKNOWN": 8000,
}

DEFAULT_TOKEN_BUDGET = 4096


def get_token_budget(document_type: str, override: int | None = None) -> int:
    """Return the response-token budget for a document type.

    Override wins. Otherwise looks up TOKEN_BUDGETS, falls back to default.
    Warns to log when the budget is large enough that the model context window
    needs explicit consideration."""
    if override:
        return override
    budget = TOKEN_BUDGETS.get(document_type, DEFAULT_TOKEN_BUDGET)
    if budget > 6000:
        logger.warning(
            "Document type %s has token budget %d. Ensure model context window "
            "is >= %d (response + prompt headroom).",
            document_type, budget, budget + 8000,
        )
    return budget


def call_llm_raw(
    client,
    *,
    model_name: str,
    system_prompt: str,
    user_text: str,
    document_type: str,
    max_tokens_override: int | None = None,
    temperature: float = 0.0,
    retries: int = 2,
) -> dict:
    """Call the LLM and return ``{raw, prompt_tokens, completion_tokens,
    finish_reason, llm_calls, llm_retries}``.

    On ``finish_reason == "length"``, bumps ``max_tokens`` by 1.5x and retries
    up to ``retries`` times. Strips markdown fences from ``raw``. Raises
    ``ValueError`` if still truncated after retries.
    """
    max_tokens = get_token_budget(document_type, max_tokens_override)
    llm_calls = 0
    llm_retries = 0
    last_finish: str | None = None
    last_raw = ""
    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None

    while True:
        llm_calls += 1
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        last_raw = strip_code_fence(getattr(choice.message, "content", "") or "")
        last_finish = getattr(choice, "finish_reason", None)

        usage = getattr(response, "usage", None)
        if usage is not None:
            last_prompt_tokens = getattr(usage, "prompt_tokens", None)
            last_completion_tokens = getattr(usage, "completion_tokens", None)

        if last_finish == "length" and llm_retries < retries:
            llm_retries += 1
            new_max = int(max_tokens * 1.5)
            logger.warning(
                "LLM response truncated (finish_reason=length) at max_tokens=%d. "
                "Retrying with max_tokens=%d.",
                max_tokens, new_max,
            )
            max_tokens = new_max
            continue

        if last_finish == "length":
            raise ValueError(
                f"Response still truncated after {retries} retries (finish_reason=length). "
                f"Use chunking for this document or raise --max-tokens."
            )

        break

    return {
        "raw": last_raw,
        "prompt_tokens": last_prompt_tokens,
        "completion_tokens": last_completion_tokens,
        "finish_reason": last_finish,
        "llm_calls": llm_calls,
        "llm_retries": llm_retries,
    }


__all__ = [
    "make_client",
    "parse_extraction",
    "strip_code_fence",
    "extract_json_object",
    "redact_ssn_in_output",
    "get_token_budget",
    "call_llm_raw",
    "TOKEN_BUDGETS",
    "DEFAULT_TOKEN_BUDGET",
]
