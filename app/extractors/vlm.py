"""VLM-based document extraction — single-pass vision language model pipeline.

Sends page images directly to a VLM (e.g., Qwen3-VL, Granite Vision) and gets
structured JSON back. No OCR binary, no text layer parsing — image in, data out.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image

from app.llm.client import LlmClient
from app.schemas import ClassificationResult, StructuredDocType

logger = logging.getLogger("loci.extractors.vlm")


def _two_pass_enabled() -> bool:
    return os.getenv("LOCI_EXTRACT_VLM_TWO_PASS", "1").strip().lower() not in ("0", "false", "no", "off")


# Tax forms where structured-field extraction benefits most from a schema-aware
# second pass. Other document types stay single-pass to avoid doubling latency.
_TWO_PASS_DOC_TYPES: set[str] = {"w2", "1099-nec", "tax_return_package", "financial_statement"}

# Map VLM doc_type responses to our StructuredDocType
_DOC_TYPE_ALIASES: dict[str, StructuredDocType] = {
    "w2": "w2",
    "w-2": "w2",
    "form w-2": "w2",
    "1099": "1099-nec",
    "1099-nec": "1099-nec",
    "1040": "tax_return_package",
    "tax_return": "tax_return_package",
    "tax return": "tax_return_package",
    "tax_return_package": "tax_return_package",
    "receipt": "receipt",
    "invoice": "receipt",
    "financial_statement": "financial_statement",
    "financial statement": "financial_statement",
    "balance_sheet": "financial_statement",
    "balance sheet": "financial_statement",
    "income_statement": "financial_statement",
}

_CLASSIFY_SYSTEM = """You are a document classification and extraction expert. Analyze the document image and return a JSON object with:
- "doc_type": one of "w2", "1099-nec", "receipt", "tax_return_package", "financial_statement", "unknown"
- "confidence": a float between 0 and 1 indicating your confidence in the classification
- "raw_text": the full text content you can read in the document
- "fields": a dict of key-value pairs extracted from the document (field names should be descriptive)

For tax forms (W-2, 1099, 1040): extract employer/payer info, recipient/employee info, amounts, tax year, box values.
For receipts: extract merchant, date, items, subtotal, tax, total.
For financial statements: extract organization, date, line items with account names and balances.
For other documents: extract whatever structured fields are visible.

Return ONLY valid JSON, no markdown fencing."""

_EXTRACT_SYSTEM = """You are a document data extraction expert. Given a document image and the document type, extract all relevant fields as a JSON object.

Return ONLY valid JSON, no markdown fencing. Include a "raw_text" field with the full readable text."""

# Layout descriptions — the VLM uses these as spatial priors so it knows WHERE
# to look for each field rather than hunting through the whole page.
_FORM_LAYOUT_HINTS: dict[str, str] = {
    "w2": (
        "A W-2 Wage and Tax Statement has this layout:\n"
        "- Top-left boxes: a (Employee SSN), b (Employer EIN — 9 digits formatted XX-XXXXXXX), "
        "c (Employer name and address, 3-4 lines), d (Control number), "
        "e (Employee first name and initial), f (Employee address and ZIP).\n"
        "- Main numbered grid, 2 columns:\n"
        "  Box 1 Wages, tips — Box 2 Federal income tax withheld\n"
        "  Box 3 Social security wages — Box 4 Social security tax withheld\n"
        "  Box 5 Medicare wages — Box 6 Medicare tax withheld\n"
        "  Box 7 Social security tips — Box 8 Allocated tips\n"
        "  Box 9 (usually blank) — Box 10 Dependent care benefits\n"
        "  Box 11 Nonqualified plans — Box 12 Codes (letters a-d each with a code letter + amount, e.g. 'DD 2500.00')\n"
        "  Box 13 three checkboxes: Statutory employee / Retirement plan / Third-party sick pay — Box 14 Other\n"
        "- State/Local section at bottom: Box 15 (State + Employer state ID), Box 16 (State wages), "
        "Box 17 (State income tax), Box 18 (Local wages), Box 19 (Local tax), Box 20 (Locality name).\n"
        "Read only numeric values for amount boxes. Dollars should include cents when present."
    ),
    "1099-nec": (
        "A 1099-NEC form has this layout:\n"
        "- Top-left: PAYER's name, street address, city, state, ZIP (multi-line block).\n"
        "- Next: PAYER's TIN (federal ID) and RECIPIENT's TIN (SSN or EIN) in two adjacent boxes.\n"
        "- Middle: RECIPIENT's name block (name line then street), then city/state/ZIP on a separate line.\n"
        "- Right column numbered boxes:\n"
        "  Box 1 Nonemployee compensation (primary dollar amount)\n"
        "  Box 2 Payer made direct sales (checkbox)\n"
        "  Box 3 (reserved)\n"
        "  Box 4 Federal income tax withheld\n"
        "  Box 5 State tax withheld / Box 6 State/Payer's state no. / Box 7 State income\n"
        "- Top-right usually shows tax year in bold (e.g. 2024)."
    ),
    "tax_return_package": (
        "A Form 1040 has this layout:\n"
        "- Top band: tax year (4-digit e.g. '2024' printed large at top right), 'U.S. Individual Income Tax Return'.\n"
        "- Name/address section: taxpayer's first name, last name, SSN; spouse's first name, last name, SSN; "
        "home address; city/state/ZIP.\n"
        "- Filing Status checkboxes: Single / Married filing jointly / Married filing separately / "
        "Head of household / Qualifying surviving spouse.\n"
        "- Digital assets question (yes/no checkbox).\n"
        "- Standard Deduction / Dependents section.\n"
        "- Income lines 1a-1z, 2a-2b, 3a-3b, 4a-4b, 5a-5b, 6a-6b, 7, 8, 9 (total income).\n"
        "- Line 10 Adjustments, Line 11 Adjusted Gross Income (AGI).\n"
        "- Line 12 Standard deduction, 13 QBI deduction, 14 subtotal, 15 Taxable income.\n"
        "- Tax/Credits (lines 16-24), Payments (25-33), Refund or Amount owed (34-37).\n"
        "Page 2 continues with signatures. Read line numbers exactly as printed."
    ),
    "financial_statement": (
        "A financial statement / balance sheet has this layout:\n"
        "- Header: organization name, statement type (Balance Sheet / Income Statement / P&L), "
        "'As Of:' or 'For Period Ending' date, accounting basis.\n"
        "- Body: a table of accounts with columns typically [Account Number | Account Name | Balance]. "
        "Section headings (Assets, Liabilities, Capital, etc.) appear as bold lines without amounts. "
        "Total lines appear below each section.\n"
        "Preserve account numbers exactly (format NNNN-NNNN). Balances may be negative, "
        "shown as -N or (N). Read all rows including zeros."
    ),
    "receipt": (
        "A receipt or invoice has this layout:\n"
        "- Top: merchant name (often largest text), address, phone.\n"
        "- Date and receipt/invoice number.\n"
        "- Line items: description, quantity, unit price, total per line.\n"
        "- Totals block at bottom: Subtotal, Tax, Tip (if any), Total.\n"
        "- Payment: payment method (Visa/Mastercard/Cash/etc) and last 4 digits if card."
    ),
}

# Per-doc-type extraction prompts — the model must return fields with these exact names.
_DOC_TYPE_PROMPTS: dict[str, str] = {
    "w2": (
        "Extract these W-2 fields and return ONLY JSON:\n"
        "{\n"
        '  "raw_text": "<full readable text>",\n'
        '  "fields": {\n'
        '    "tax_year": <int>, "employer_name": <string>, "employer_ein": <string>,\n'
        '    "employer_address": <string>, "employee_name": <string>, "employee_ssn": <string>,\n'
        '    "employee_address": <string>,\n'
        '    "box1_wages": <number>, "box2_federal_tax": <number>,\n'
        '    "box3_ss_wages": <number>, "box4_ss_tax": <number>,\n'
        '    "box5_medicare_wages": <number>, "box6_medicare_tax": <number>,\n'
        '    "box7_ss_tips": <number>, "box8_allocated_tips": <number>,\n'
        '    "box10_dependent_care": <number>, "box11_nonqualified_plans": <number>,\n'
        '    "box12_codes": [{"code": "DD", "amount": <number>}, ...],\n'
        '    "box13_statutory": <bool>, "box13_retirement": <bool>, "box13_third_party_sick": <bool>,\n'
        '    "box14_other": <string or null>,\n'
        '    "state": <string>, "state_id": <string>,\n'
        '    "box16_state_wages": <number>, "box17_state_tax": <number>,\n'
        '    "box18_local_wages": <number>, "box19_local_tax": <number>,\n'
        '    "box20_locality_name": <string or null>\n'
        "  }\n"
        "}\n"
        "Leave fields null if not visible. Do not guess."
    ),
    "1099-nec": (
        "Extract these 1099-NEC fields and return ONLY JSON:\n"
        "{\n"
        '  "raw_text": "<full readable text>",\n'
        '  "fields": {\n'
        '    "tax_year": <int>, "payer_name": <string>, "payer_tin": <string>,\n'
        '    "payer_address": <string>, "recipient_name": <string>, "recipient_tin": <string>,\n'
        '    "recipient_address": <string>,\n'
        '    "box1_nonemployee_compensation": <number>,\n'
        '    "box2_direct_sales_over_5000": <bool>,\n'
        '    "box4_federal_tax_withheld": <number>,\n'
        '    "box5_state_tax_withheld": <number>, "box6_state_payer_id": <string>,\n'
        '    "box7_state_income": <number>\n'
        "  }\n"
        "}\n"
        "Leave fields null if not visible."
    ),
    "tax_return_package": (
        "Extract these Form 1040 fields and return ONLY JSON:\n"
        "{\n"
        '  "raw_text": "<full readable text>",\n'
        '  "fields": {\n'
        '    "tax_year": <int>, "filing_status": <string>,\n'
        '    "taxpayer_name": <string>, "taxpayer_ssn": <string>,\n'
        '    "spouse_name": <string or null>, "spouse_ssn": <string or null>,\n'
        '    "address": <string>,\n'
        '    "line_1z_total_wages": <number>, "line_9_total_income": <number>,\n'
        '    "line_10_adjustments": <number>, "line_11_agi": <number>,\n'
        '    "line_12_standard_deduction": <number>, "line_15_taxable_income": <number>,\n'
        '    "line_16_tax": <number>, "line_24_total_tax": <number>,\n'
        '    "line_33_total_payments": <number>,\n'
        '    "line_34_overpayment": <number>, "line_37_amount_owed": <number>\n'
        "  }\n"
        "}\n"
        "Leave null for missing lines."
    ),
    "financial_statement": (
        "Extract these balance sheet / financial statement fields and return ONLY JSON:\n"
        "{\n"
        '  "raw_text": "<full readable text>",\n'
        '  "fields": {\n'
        '    "organization_name": <string>, "statement_date": <string>,\n'
        '    "report_type": <string>, "accounting_basis": <string>,\n'
        '    "line_items": [\n'
        '      {"account_number": "1000-0000", "account_name": "Cash", "balance": 1234.56, "section": "Assets"}\n'
        "    ],\n"
        '    "sections": [{"name": "Assets", "total": 10000.00}, ...]\n'
        "  }\n"
        "}\n"
        "Preserve account numbers verbatim."
    ),
    "receipt": (
        "Extract receipt fields and return ONLY JSON:\n"
        "{\n"
        '  "raw_text": "<full readable text>",\n'
        '  "fields": {\n'
        '    "merchant_name": <string>, "date": <string>, "time": <string or null>,\n'
        '    "receipt_number": <string or null>,\n'
        '    "items": [{"name": <string>, "quantity": <number>, "price": <number>}],\n'
        '    "subtotal": <number>, "tax": <number>, "tip": <number or null>,\n'
        '    "total": <number>, "payment_method": <string or null>\n'
        "  }\n"
        "}"
    ),
}


def vlm_classify(
    client: LlmClient,
    image: Image.Image,
) -> ClassificationResult | None:
    """Classify a document image using a VLM. Returns None on failure."""
    try:
        result = client.vision_extract_json(
            _CLASSIFY_SYSTEM,
            "Classify this document and extract its contents.",
            image,
        )
        if result is None:
            return None

        raw_type = str(result.get("doc_type", "unknown")).lower().strip()
        doc_type = _DOC_TYPE_ALIASES.get(raw_type, "unknown")
        confidence = float(result.get("confidence", 0.0))

        return ClassificationResult(
            doc_type=doc_type,
            confidence=min(confidence, 1.0),
            strategy="vlm",
            matched_signals=[f"vlm_doc_type:{raw_type}"],
        )
    except Exception as exc:
        logger.warning("VLM classification failed: %s", exc)
        return None


_PASS1_SYSTEM = (
    "You are a precise document OCR and classification expert. Read ALL visible "
    "text verbatim without paraphrasing, including handwriting, stamps, and "
    "small print. Identify the document type. Return ONLY valid JSON."
)

_PASS1_USER = (
    "Read every word of text visible on this page, preserving line breaks with \\n. "
    "Also identify the document type.\n\n"
    "Return JSON:\n"
    "{\n"
    '  "raw_text": "<all visible text, line breaks preserved>",\n'
    '  "doc_type": <one of "w2", "1099-nec", "tax_return_package", "financial_statement", "receipt", "unknown">,\n'
    '  "confidence": <float 0-1>\n'
    "}\n"
    "If unsure of doc_type, use 'unknown'. Do not output anything except the JSON."
)


def vlm_extract_two_pass(
    client: LlmClient,
    image: Image.Image,
    *,
    doc_type_hint: str | None = None,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Two-pass VLM extraction for tax/structured forms.

    Pass 1: verbatim text read + document type classification.
    Pass 2: schema-guided structured field extraction, using both the image
            and the raw text from pass 1 to reduce ambiguity.

    For documents not in _TWO_PASS_DOC_TYPES, returns after pass 1 with
    an empty fields dict (caller still gets full raw_text).
    """
    if trace is not None:
        trace["image_size"] = list(image.size)
        trace["two_pass"] = True

    # Pass 1: OCR + classify
    try:
        pass1 = client.vision_extract_json(_PASS1_SYSTEM, _PASS1_USER, image)
    except Exception as exc:
        logger.warning("VLM two-pass: pass 1 failed: %s", exc)
        if trace is not None:
            trace["pass1"] = {"ok": False, "error": str(exc)}
        return None

    if not pass1 or not pass1.get("raw_text"):
        if trace is not None:
            trace["pass1"] = {"ok": False, "error": "empty_response"}
        return None

    raw_text = str(pass1.get("raw_text", "")).strip()
    pass1_type_raw = str(pass1.get("doc_type", "unknown")).lower().strip()
    resolved_type = doc_type_hint or _DOC_TYPE_ALIASES.get(pass1_type_raw, "unknown")
    if trace is not None:
        trace["pass1"] = {
            "ok": True,
            "chars": len(raw_text),
            "doc_type_raw": pass1_type_raw,
            "doc_type_resolved": resolved_type,
            "confidence": float(pass1.get("confidence", 0.0)),
        }

    # Pass 2: only for known tax/structured forms
    if resolved_type not in _TWO_PASS_DOC_TYPES:
        if trace is not None:
            trace["pass2"] = {"skipped": True, "reason": f"doc_type={resolved_type}"}
        return {"raw_text": raw_text, "fields": {}, "doc_type_guess": resolved_type}

    layout_hint = _FORM_LAYOUT_HINTS.get(resolved_type, "")
    schema_prompt = _DOC_TYPE_PROMPTS.get(resolved_type, "")
    # Include pass-1 text as an aid — the model can reconcile image + text
    text_excerpt = raw_text[:3000]
    user_prompt = (
        f"{layout_hint}\n\n"
        f"The OCR pass already extracted this text from the page (use it to disambiguate numbers and names, "
        f"but the IMAGE is the source of truth):\n---\n{text_excerpt}\n---\n\n"
        f"{schema_prompt}"
    )
    try:
        pass2 = client.vision_extract_json(
            "You are a tax document data extraction expert. Return ONLY valid JSON matching the requested schema.",
            user_prompt,
            image,
        )
    except Exception as exc:
        logger.warning("VLM two-pass: pass 2 failed: %s", exc)
        if trace is not None:
            trace["pass2"] = {"ok": False, "error": str(exc)}
        return {"raw_text": raw_text, "fields": {}, "doc_type_guess": resolved_type}

    fields: dict[str, Any] = {}
    pass2_raw_text = ""
    if pass2:
        returned_fields = pass2.get("fields")
        if isinstance(returned_fields, dict):
            fields = returned_fields
        pass2_raw_text = str(pass2.get("raw_text") or "").strip()

    # Prefer pass-2 raw_text if it's substantially richer (sometimes pass 2 re-reads
    # more carefully), otherwise keep pass 1's text.
    final_text = pass2_raw_text if len(pass2_raw_text) > len(raw_text) + 50 else raw_text

    if trace is not None:
        trace["pass2"] = {
            "ok": bool(pass2),
            "fields_count": len(fields),
            "raw_text_chars": len(pass2_raw_text),
        }

    return {"raw_text": final_text, "fields": fields, "doc_type_guess": resolved_type}


def vlm_extract_page(
    client: LlmClient,
    image: Image.Image,
    *,
    doc_type: str = "unknown",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Extract structured data from a single page image using a VLM.

    Returns a dict with 'raw_text' and 'fields', or None on failure.
    Falls back to raw text extraction if JSON extraction fails.

    When LOCI_EXTRACT_VLM_TWO_PASS is enabled (default), uses a two-pass
    approach: pass 1 reads text + classifies, pass 2 extracts schema-specific
    fields. Falls through to single-pass if two-pass fails.

    When ``trace`` is provided, records which path was taken and the
    response character count.
    """
    # Try two-pass first for known tax/structured forms (and for unknown types —
    # the pass-1 classification may promote them into two-pass territory).
    if _two_pass_enabled():
        hint = doc_type if doc_type not in ("unknown", "", None) else None
        result = vlm_extract_two_pass(client, image, doc_type_hint=hint, trace=trace)
        if result is not None and result.get("raw_text"):
            if trace is not None:
                trace["attempt"] = "two_pass"
                trace["parsed_ok"] = True
                trace["response_chars"] = len(result["raw_text"])
                trace["had_fields"] = bool(result.get("fields"))
            return result
        # Fall through to single-pass on two-pass failure
        if trace is not None:
            trace["two_pass_fallback"] = True

    if trace is not None:
        trace["image_size"] = list(image.size)

    # First try: structured JSON extraction
    try:
        prompt = _DOC_TYPE_PROMPTS.get(doc_type, (
            "Extract all visible text and structured fields from this document. "
            "Return as JSON with a 'fields' dict and 'raw_text' string."
        ))

        result = client.vision_extract_json(
            _EXTRACT_SYSTEM,
            prompt,
            image,
        )
        if result and result.get("raw_text"):
            if trace is not None:
                trace["attempt"] = "structured"
                trace["parsed_ok"] = True
                trace["response_chars"] = len(str(result.get("raw_text", "")))
                trace["had_fields"] = bool(result.get("fields"))
            return result
    except Exception as exc:
        logger.warning("VLM structured extraction failed: %s", exc)

    # Fallback: ask for plain text only (simpler prompt, more reliable)
    try:
        logger.info("Falling back to plain text VLM extraction")
        result = client.vision_extract_json(
            "You are an OCR system. Extract all visible text from this document image exactly as written.",
            "Read all text in this image. Return JSON: {\"raw_text\": \"the full text here\"}",
            image,
        )
        if result and result.get("raw_text"):
            if trace is not None:
                trace["attempt"] = "plain_text_json"
                trace["parsed_ok"] = True
                trace["response_chars"] = len(str(result.get("raw_text", "")))
                trace["had_fields"] = bool(result.get("fields"))
            return result
    except Exception as exc:
        logger.warning("VLM plain text extraction also failed: %s", exc)

    # Last resort: try to get ANY response from the VLM as raw text
    try:
        import base64
        from io import BytesIO
        import httpx

        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        body = {
            "model": client.model,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Read all the text in this image. Just output the text, nothing else."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"

        response = httpx.post(
            f"{client.base_url}/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=client.timeout,
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Strip any thinking tags that some models produce
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                logger.info("Raw VLM response recovered %d chars", len(content))
                if trace is not None:
                    trace["attempt"] = "raw_fallback"
                    trace["parsed_ok"] = False
                    trace["response_chars"] = len(content)
                    trace["had_fields"] = False
                return {"raw_text": content, "fields": {}}
    except Exception as exc:
        logger.warning("VLM raw fallback also failed: %s", exc)

    if trace is not None:
        trace["attempt"] = "none"
        trace["parsed_ok"] = False
        trace["response_chars"] = 0
        trace["had_fields"] = False
    return None


_VERIFY_SYSTEM = """You are a document text quality verifier. You will be given text extracted from a PDF page.
Determine if the text is readable, coherent, and represents real document content.

Return ONLY a JSON object with:
- "usable": true if the text is real readable content, false if it's garbage/corrupted/encoded junk
- "reason": brief explanation (e.g. "readable English text", "encoded glyph references", "control characters", "random symbols")
- "language": detected language if readable, null if garbage
- "confidence": float 0-1 of how sure you are

Signs of garbage text: (cid:XX) patterns, excessive control characters, random symbol sequences,
base64-like strings, hex dumps, font encoding artifacts, text that makes no semantic sense.

Signs of usable text: recognizable words, coherent sentences, form field labels, numbers that
look like amounts/dates, proper nouns, addresses, etc.

Return ONLY valid JSON, no markdown."""


def verify_text_quality(
    client: LlmClient,
    text: str,
    *,
    min_confidence: float = 0.7,
) -> dict[str, Any]:
    """Use a fast LLM to verify if extracted text is real readable content.

    Returns {"usable": bool, "reason": str, "confidence": float}.
    """
    if not text or len(text.strip()) < 10:
        return {"usable": False, "reason": "text_too_short", "confidence": 1.0}

    # Quick heuristic pre-check before burning an LLM call
    stripped = text.strip()
    if stripped.count("(cid:") >= 3:
        return {"usable": False, "reason": "cid_glyph_garbage", "confidence": 1.0}

    control_chars = sum(1 for c in stripped if ord(c) < 32 and c not in "\n\r\t")
    if len(stripped) > 0 and control_chars / len(stripped) > 0.12:
        return {"usable": False, "reason": "high_control_char_density", "confidence": 1.0}

    # Use LLM for semantic verification
    try:
        # Only send first 500 chars — enough to judge quality
        sample = stripped[:500]
        result = client.complete_json(
            _VERIFY_SYSTEM,
            f"Verify this extracted text:\n\n{sample}",
        )
        if result is None:
            # LLM unavailable — fall back to assuming text is usable if it passed heuristics
            return {"usable": True, "reason": "llm_unavailable_heuristic_pass", "confidence": 0.5}

        usable = bool(result.get("usable", False))
        confidence = float(result.get("confidence", 0.0))
        reason = str(result.get("reason", "unknown"))

        # Only trust the LLM verdict if it's confident enough
        if confidence < min_confidence:
            return {"usable": True, "reason": f"low_confidence_default_accept:{reason}", "confidence": confidence}

        return {"usable": usable, "reason": reason, "confidence": confidence}

    except Exception as exc:
        logger.warning("Text verification failed: %s", exc)
        return {"usable": True, "reason": f"verification_error:{exc}", "confidence": 0.3}


def vlm_classify_and_extract(
    client: LlmClient,
    image: Image.Image,
) -> dict[str, Any] | None:
    """Single-pass: classify AND extract in one VLM call. Returns combined result or None."""
    try:
        result = client.vision_extract_json(
            _CLASSIFY_SYSTEM,
            "Classify this document and extract all fields.",
            image,
        )
        return result
    except Exception as exc:
        logger.warning("VLM classify+extract failed: %s", exc)
        return None
