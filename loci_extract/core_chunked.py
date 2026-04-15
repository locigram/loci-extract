"""Financial-document extraction: chunk → multi-LLM call → merge → verify.

Used by ``core.extract_document`` when the detected family is one of the
``financial_*`` families. Tax documents bypass this entirely and use
``parse_extraction()`` directly to preserve byte-exact tax behavior.

Pipeline:
  1. detect_financial_document_type → choose family + per-type prompt
  2. chunk_for_llm → 1+ TextChunks
  3. for each chunk: call_llm_raw → strip+parse JSON → store partial dict
  4. _merge_chunks (per-doc-type strategy) → single dict
  5. verify_section_totals → metadata.totals_*
  6. compute_derived_fields → *_calculated fields
  7. wrap as Extraction.documents[0]
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from loci_extract.chunker import chunk_for_llm
from loci_extract.detector import detect_financial_document_type
from loci_extract.llm import call_llm_raw, extract_json_object, redact_ssn_in_output
from loci_extract.prompts import get_prompt
from loci_extract.schema import Document, Extraction
from loci_extract.verifier import compute_derived_fields, verify_section_totals

if TYPE_CHECKING:
    from loci_extract.core import ExtractionOptions, ProgressCallback

logger = logging.getLogger("loci_extract.core_chunked")


def extract_financial_document(
    *,
    client,
    raw_text: str,
    opts: ExtractionOptions,
    family: str,
    progress: ProgressCallback | None = None,
) -> Extraction:
    """End-to-end financial extraction. Returns an Extraction with one document."""

    document_type = detect_financial_document_type(raw_text)
    if document_type == "FINANCIAL_UNKNOWN":
        document_type = "BALANCE_SHEET"  # safe-ish default for the simple family

    if progress:
        progress(f"financial doc type: {document_type}")

    chunks = chunk_for_llm(raw_text, document_type, max_input_tokens=opts.chunk_size_tokens)
    if progress and len(chunks) > 1:
        progress(f"chunked into {len(chunks)} segments")

    system_prompt = get_prompt(document_type)

    partials: list[dict] = []
    total_calls = 0
    total_retries = 0
    finish_reasons: list[str] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    # Run chunks in parallel — they're independent LLM calls with no
    # cross-chunk dependency (merge runs after all return). ThreadPool suits
    # HTTP-bound OpenAI-compat clients; llama-server's cont-batching handles
    # 2-4 concurrent slots natively. Capped at min(opts.max_parallel, len(chunks))
    # to avoid thrashing when chunks=1 (most BS/IS docs).
    max_parallel = max(1, opts.max_parallel)
    workers = min(max_parallel, len(chunks))
    if progress and len(chunks) > 1 and workers > 1:
        progress(f"running {len(chunks)} chunks with {workers}-way parallelism")

    def _run_one(chunk):
        user_text = _build_chunk_user_text(chunk, document_type)
        try:
            return chunk, call_llm_raw(
                client=client,
                model_name=opts.model_name,
                system_prompt=system_prompt,
                user_text=user_text,
                document_type=document_type,
                max_tokens_override=opts.max_tokens if opts.max_tokens != 4096 else None,
                temperature=opts.temperature,
                retries=opts.retry,
            ), None
        except Exception as exc:
            return chunk, None, exc

    chunk_results: list[tuple] = []
    if workers == 1:
        chunk_results = [_run_one(c) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, c) for c in chunks]
            chunk_results = [f.result() for f in as_completed(futures)]

    # Re-sort results by chunk_index so merge processes them in source order
    chunk_results.sort(key=lambda t: t[0].chunk_index)

    for chunk, llm_resp, exc in chunk_results:
        if exc is not None:
            logger.warning(
                "Chunk %d/%d LLM call failed: %s",
                chunk.chunk_index + 1, chunk.total_chunks, exc,
            )
            continue

        total_calls += llm_resp.get("llm_calls", 1)
        total_retries += llm_resp.get("llm_retries", 0)
        if llm_resp.get("finish_reason"):
            finish_reasons.append(llm_resp["finish_reason"])
        if llm_resp.get("prompt_tokens"):
            prompt_tokens_total += llm_resp["prompt_tokens"]
        if llm_resp.get("completion_tokens"):
            completion_tokens_total += llm_resp["completion_tokens"]

        try:
            json_str = extract_json_object(llm_resp["raw"])
            parsed = json.loads(json_str)
            partials.append(parsed)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Chunk %d/%d produced invalid JSON: %s",
                chunk.chunk_index + 1, chunk.total_chunks, exc,
            )
            continue

    if not partials:
        # All chunks failed. Emit a visible error and return a document
        # carrying the failure diagnostics in metadata so downstream CSV /
        # JSON consumers can detect it rather than seeing a silently-empty
        # Extraction. Batch path continues past this so one bad file doesn't
        # abort a whole directory run.
        msg = (
            f"financial extraction failed: {document_type} produced 0 valid "
            f"chunks after {total_calls} LLM call(s), {total_retries} retry(ies); "
            f"last finish_reason={finish_reasons[-1] if finish_reasons else 'n/a'}. "
            f"Common cause: response token budget too small for a long "
            f"multi-column report. Try a larger model ctx window or "
            f"--chunk-size lower to split input into more chunks."
        )
        logger.error(msg)
        if progress:
            progress(f"ERROR: {msg}")
        failed_doc = Document(
            document_type=document_type,
            tax_year=2025,
            data={"entity": {"name": "EXTRACTION FAILED"}},
            metadata={
                "notes": [msg],
                "totals_verified": False,
                "llm_calls": total_calls,
                "llm_retries": total_retries,
                "finish_reason": finish_reasons[-1] if finish_reasons else None,
            },
        )
        return Extraction(documents=[failed_doc])

    merged = _merge_chunks(partials, document_type)

    # ── Post-processing: verify totals + compute derived ─────────────────────
    if opts.verify_totals:
        verification = verify_section_totals(merged)
        meta = merged.setdefault("metadata", {})
        meta["totals_verified"] = verification.verified
        meta["totals_mismatches"] = verification.mismatches
        if verification.balance_sheet_balanced is not None:
            meta["balance_sheet_balanced"] = verification.balance_sheet_balanced
        existing_notes = meta.get("notes", []) or []
        if verification.notes:
            meta["notes"] = list(existing_notes) + verification.notes
        if progress:
            progress(
                f"totals verified: {verification.verified} "
                f"({len(verification.mismatches)} mismatch(es))"
            )

    derived = compute_derived_fields(merged, document_type)
    merged.update(derived)

    # Pipeline metadata enrichment
    meta = merged.setdefault("metadata", {})
    meta.setdefault("llm_calls", total_calls)
    meta.setdefault("llm_retries", total_retries)
    if finish_reasons:
        meta.setdefault("finish_reason", finish_reasons[-1])
    if prompt_tokens_total:
        meta.setdefault("prompt_tokens", prompt_tokens_total)
    if completion_tokens_total:
        meta.setdefault("completion_tokens", completion_tokens_total)

    # Apply SSN redaction on the merged dict (cheap walk)
    if opts.redact:
        merged = redact_ssn_in_output(merged)

    # Pop top-level "metadata" — Document carries it as its own field
    doc_metadata = merged.pop("metadata", {}) or {}

    # Tax year fallback (financial docs may not have one)
    tax_year = _extract_tax_year(merged)

    document = Document(
        document_type=document_type,
        tax_year=tax_year if tax_year is not None else 2025,
        data=merged,
        metadata=doc_metadata,
    )
    return Extraction(documents=[document])


# ---------------------------------------------------------------------------
# Chunk merging — per-doc-type strategies
# ---------------------------------------------------------------------------


def _build_chunk_user_text(chunk, document_type: str) -> str:
    if chunk.total_chunks <= 1:
        return chunk.text
    header = (
        f"[CHUNK {chunk.chunk_index + 1} OF {chunk.total_chunks}]\n"
        f"This is one segment of a larger {document_type} document.\n"
    )
    if chunk.account_context:
        header += f"This chunk begins at: {chunk.account_context}\n"
    if chunk.chunk_index > 0:
        header += (
            "Entity/header information was in chunk 1. For this chunk, extract "
            "only the accounts/transactions present. Set entity fields to null "
            "if you can't see the header — they will be taken from chunk 1.\n"
        )
    return header + "\n" + chunk.text


def _merge_chunks(partials: list[dict], document_type: str) -> dict:
    """Merge N partial extraction dicts. Strategy depends on document_type."""
    if not partials:
        return {}
    if len(partials) == 1:
        return partials[0]

    merged: dict = {}

    # Entity: first chunk with a populated entity.name
    for p in partials:
        entity = p.get("entity") or {}
        if entity.get("name"):
            merged["entity"] = entity
            break
    if "entity" not in merged:
        merged["entity"] = partials[0].get("entity", {})

    # Metadata: union notes, OR-combine flags
    merged_meta: dict = {}
    all_notes: list[str] = []
    all_rotated: list[int] = []
    for p in partials:
        meta = p.get("metadata", {}) or {}
        all_notes.extend(meta.get("notes", []) or [])
        all_rotated.extend(meta.get("pages_rotated", []) or [])
        for flag in ("encoding_broken", "is_corrected", "is_void", "is_summary_sheet"):
            if meta.get(flag):
                merged_meta[flag] = True
    merged_meta["notes"] = list(dict.fromkeys(all_notes))  # preserve order, dedup
    if all_rotated:
        merged_meta["pages_rotated"] = sorted(set(all_rotated))
    merged["metadata"] = merged_meta

    if document_type in ("GENERAL_LEDGER", "QB_GENERAL_LEDGER"):
        merged["accounts"] = _merge_gl_accounts(partials)

    elif document_type in ("QB_TRANSACTION_LIST",):
        merged["transactions"] = []
        for p in partials:
            merged["transactions"].extend(p.get("transactions", []) or [])

    elif document_type in (
        "INCOME_STATEMENT_COMPARISON", "BUDGET_VS_ACTUAL", "QB_PROFIT_LOSS"
    ):
        merged["columns"] = partials[0].get("columns", []) or []
        seen: set = set()
        merged["line_items"] = []
        for p in partials:
            for item in p.get("line_items", []) or []:
                key = (item.get("account_number"), item.get("account_name"))
                if key not in seen:
                    seen.add(key)
                    merged["line_items"].append(item)

    elif document_type == "BALANCE_SHEET":
        merged["assets"] = _merge_financial_side(partials, "assets")
        merged["liabilities"] = _merge_financial_side(partials, "liabilities")
        merged["equity"] = _merge_financial_side(partials, "equity")
        for key in ("total_liabilities_and_equity_reported",):
            for p in partials:
                if p.get(key) is not None:
                    merged[key] = p[key]
                    break

    elif document_type == "INCOME_STATEMENT":
        merged["income"] = _merge_financial_side(partials, "income")
        merged["expenses"] = _merge_financial_side(partials, "expenses")
        merged["other_income"] = _merge_financial_side(partials, "other_income")
        for key in ("operating_income_reported", "net_income_reported"):
            for p in partials:
                if p.get(key) is not None:
                    merged[key] = p[key]
                    break

    else:
        # Generic merge: concatenate any list fields, take scalars from first non-None
        for p in partials[1:]:
            for k, v in p.items():
                if k in ("entity", "metadata"):
                    continue
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                elif k not in merged:
                    merged[k] = v
        # Also seed scalar/dict fields from chunk 0 where missing
        for k, v in partials[0].items():
            if k in ("entity", "metadata"):
                continue
            if k not in merged:
                merged[k] = v

    return merged


def _merge_gl_accounts(partials: list[dict]) -> list[dict]:
    """Merge GL accounts across chunks. An account split at a chunk boundary
    appears in two partials with the same account_number; merge transactions."""
    accounts_by_key: dict[str, dict] = {}
    order: list[str] = []
    for p in partials:
        for acct in p.get("accounts", []) or []:
            key = acct.get("account_number") or acct.get("account_name", "")
            if key in accounts_by_key:
                existing = accounts_by_key[key]
                existing["transactions"] = (
                    existing.get("transactions", []) + (acct.get("transactions", []) or [])
                )
                if acct.get("ending_balance") is not None:
                    existing["ending_balance"] = acct["ending_balance"]
            else:
                accounts_by_key[key] = dict(acct)
                order.append(key)
    return [accounts_by_key[k] for k in order]


def _merge_financial_side(partials: list[dict], key: str) -> dict:
    """Merge a balance sheet or P&L side (assets/liabilities/equity/income/expenses)
    across chunks. Sections are merged by section_name."""
    merged_side: dict = {}
    sections_by_name: dict[str, dict] = {}
    section_order: list[str] = []
    for p in partials:
        side = p.get(key) or {}
        if not side:
            continue
        for k, v in side.items():
            if k != "sections" and k not in merged_side:
                merged_side[k] = v
        for section in side.get("sections", []) or []:
            sname = section.get("section_name", "")
            if sname in sections_by_name:
                sections_by_name[sname]["accounts"] = (
                    sections_by_name[sname].get("accounts", [])
                    + (section.get("accounts", []) or [])
                )
                if section.get("section_total") is not None:
                    sections_by_name[sname]["section_total"] = section["section_total"]
            else:
                sections_by_name[sname] = dict(section)
                section_order.append(sname)
    merged_side["sections"] = [sections_by_name[n] for n in section_order]
    return merged_side


def _extract_tax_year(data: dict) -> int | None:
    """Best-effort tax year detection from the merged dict."""
    import re
    if "tax_year" in data:
        try:
            return int(data["tax_year"])
        except (TypeError, ValueError):
            pass
    entity = data.get("entity", {}) or {}
    for k in ("period_end", "period_start"):
        val = entity.get(k, "")
        if val:
            m = re.search(r"(\d{4})", str(val))
            if m:
                year = int(m.group(1))
                if 2010 <= year <= 2035:
                    return year
    return None


__all__ = [
    "extract_financial_document",
    "_merge_chunks",
    "_merge_gl_accounts",
    "_merge_financial_side",
]
