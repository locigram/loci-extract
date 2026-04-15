# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Run the API (dev)
uvicorn app.main:app --reload   # http://127.0.0.1:8000, docs at /docs

# Tests
pytest -q
pytest tests/test_extract_api.py -q              # single file
pytest tests/test_extract_api.py::test_name -q   # single test

# Lint
ruff check .

# Docker (full OCR/PDF stack baked in)
docker build -t loci-extract:local .
docker compose up --build -d
```

System binaries used by extraction (already in the Docker image; install on the host for local OCR work): `tesseract-ocr`, `ocrmypdf`, `ghostscript`, `unpaper`, `poppler-utils`. `GET /capabilities` reports which are present at runtime.

Operational env vars: `LOCI_EXTRACT_MAX_UPLOAD_BYTES` (default 25 MiB), `LOCI_EXTRACT_MAX_PDF_PAGES` (default 200), `WEB_CONCURRENCY`.

## Architecture

`loci-extract` is a FastAPI service that turns uploaded documents into a normalized JSON payload. It is intentionally **ephemeral** — the upload is written to a temp file, extracted, then deleted; persistence is the caller's responsibility.

There are three layers, and the design intent (see README "Product goals") is to keep layer 1 stable and full-fidelity while layers 2/3 evolve:

1. **Canonical extraction** (`app/extractors/`, `app/router.py`) — parser-first, deterministic. `router.choose_extractor` dispatches by filename/mime to one of `PdfExtractor`, `DocxExtractor`, `XlsxExtractor`, `ImageOcrExtractor`, `PlainTextExtractor`. Every extractor returns the same `ExtractionPayload` shape (`app/schemas.py`): `raw_text` + `segments` + `chunks` + `extra` + `extraction` (status/warnings/ocr_used).
2. **Classification** (`app/classification/`) — multiple classifier strategies dispatched by `app/classification/routing.py::classify_with_profile()`. Strategies: `"rules"` (text pattern matching via `rules.py`), `"layout"` (PP-Structure GPU layout analysis via `layout.py`), `"donut-irs"` (Donut IRS tax classifier via `donut_classifier.py`, 28 IRS form types), `"auto"` (tries donut → layout → rules, first confident result wins). The `classify_document()` function in `rules.py` accepts an optional `page_image` kwarg for the layout path. `ClassificationResult.strategy` is `Literal["rules", "hint", "layout", "donut"]`. Returns a detected doc type from `{w2, 1099-nec, receipt, tax_return_package, financial_statement, unknown}`.
3. **Structured interpretation** (`app/structured/`) — `structured.router.build_structured_document` dispatches on the classification to per-form modules (`w2.py`, `form_1099_nec.py`, `receipt.py`, `tax_return_package.py`, `financial_statement.py`). Sits on top of the raw payload, never replaces it.

The HTTP surface (`app/main.py`) is small: `GET /healthz`, `GET /capabilities`, `POST /extract`, `POST /extract/structured`, `POST /extract/compare`, `POST /identify`. `_extract_payload_from_upload` splits into `_save_upload_tempfile` (size-check + tempfile write) and `_extract_from_path` (calls `extract_file` with a strategy/backend and runs `_enrich_payload_extra`), so `/extract/compare` can write the upload once and loop pipelines over the same path. `_enrich_payload_extra` adds stable provenance fields (`segment_count`, `warning_codes`, `ocr_quality_summary`, `ocr_evidence_snippets`, etc.) before returning. `/extract/structured` runs classification and the structured router on top.

### PDF extraction is the most complex path

`PdfExtractor` is the only extractor with a multi-strategy pipeline. It tries, in order: PyMuPDF text layer → pdfplumber for table segments → Tesseract OCR per page when no text layer or `ocr_strategy=always` → OCRmyPDF when the parser produces "garbage" (e.g. `(cid:...)` glyphs, high control-char density). Per-page provenance lands in `extra.page_provenance` with `source ∈ {parser, ocr, parser_fallback, none}`. `extra.result_source`, `extra.ocr_backend`, and `extra.ocrmypdf_trigger_reason` describe the overall route taken. When touching this code, preserve the page-level provenance fields — downstream consumers (and tests) depend on them.

### OCR backend abstraction

`app/ocr_backends/` provides a pluggable `OcrBackend` Protocol. `get_backend(name)` resolves `"auto" | "tesseract" | "paddleocr"` to a concrete backend with availability fallback. `TesseractBackend` wraps `pytesseract`; `PaddleOCRBackend` is import-guarded and requires CUDA (or `LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU=1`). PaddleOCR uses single-pass mode (no multi-variant preprocessing) with `selected_ocr_pass = "paddleocr_native"`.

**Important for tests**: existing tests monkeypatch `app.extractors.pdf.tesseract_available` and `app.extractors.image_ocr.tesseract_available` (the local import refs). The extractors use these local refs for the `auto`/`tesseract` availability check, and only use `get_backend()` for explicit non-default backends like `paddleocr`. This preserves existing monkeypatch targets — do not refactor the extractors to use `get_backend()` for availability on the default path.

### OCR strategy & quality

`ocr_strategy` controls **whether** to use OCR (`auto | always | never`). `ocr_backend` controls **which** backend to use (`auto | tesseract | paddleocr`). Both are request-level Form fields on `/extract` and `/extract/structured`. `extra["ocr_backend"]` records the backend actually used; `extra["ocr_backend_requested"]` records what was requested. The structured pipeline is **deliberately conservative**: OCR-backed or parser-fallback tax pages add review reasons, missing required fields force `requires_human_review=true`, and pages with no recovered text also force review. Do not "fix" this by lowering the bar — accuracy on tax/financial docs is a stated product goal and the existing tests assert these flags.

### LLM enrichment

`app/llm/` provides a thin OpenAI-compatible client (`LlmClient`) for optional structured extraction enhancement. Controlled by env vars (`LOCI_EXTRACT_LLM_ENABLED`, `LOCI_EXTRACT_LLM_BASE_URL`, `LOCI_EXTRACT_LLM_MODEL`). The first consumer is financial-statement section labeling — `_maybe_enrich_sections_with_llm` in `app/structured/financial_statement.py` assigns sections to unclassified line items via LLM. Gated by `enable_llm_enrichment` Form field on `/extract/structured` AND `get_llm_client() is not None`. **Never raises** — failures degrade to the rule-based result with `llm_enrichment.applied = False` in the structured fields.

### VLM rendering and two-pass extraction

VLM paths in `app/extractors/pdf.py` render pages at `fitz.Matrix(3, 3)` (≈216 DPI) and cap the longest image side at 2560px (module constants `_VLM_RENDER_DPI`, `_VLM_MAX_IMAGE_DIM`). Previous settings (144 DPI / 1568px) were squashing small numerals on W-2 Box 12 / 1040 line items.

`vlm_extract_page` in `app/extractors/vlm.py` defaults to a **two-pass** extraction when `LOCI_EXTRACT_VLM_TWO_PASS` is enabled (default on, set to `0` to disable). Pass 1: verbatim text OCR + document-type classification. Pass 2: schema-guided structured field extraction, seeded with pass-1 text as context. Pass 2 only runs for `_TWO_PASS_DOC_TYPES` (`w2`, `1099-nec`, `tax_return_package`, `financial_statement`) — generic pages stop after pass 1 to avoid doubling latency. `_FORM_LAYOUT_HINTS` carry per-form box/layout descriptions so the model has spatial priors, and `_DOC_TYPE_PROMPTS` specify exact JSON schemas to produce. Trace records `pass1` and `pass2` sub-entries under the `vlm` key when two-pass ran.

### VLM trace and pipeline comparison

When `ocr_strategy=vlm` or `vlm_hybrid`, `app/extractors/pdf.py` populates `extra.vlm_trace` with per-page stage info: `parser_chars`, `parser_looked_like_garbage`, `verify{usable, reason, confidence, ms}`, `stage_selected ∈ {parser, vlm, none}`, `vlm{attempt, response_chars, parsed_ok, had_fields, image_size, render_dpi, ms}`, `final_chars`, `ms`. Totals include `parser_used`, `vlm_used`, `verify_calls`, `total_ms`. This is the primary signal for diagnosing why VLM output is poor — check it before changing render DPI or image max-dim. `vlm_extract_page` and `verify_text_quality` in `app/extractors/vlm.py` accept an optional `trace=dict` kwarg that records which of the three fallback attempts (`structured`, `plain_text_json`, `raw_fallback`, `none`) won.

`POST /extract/compare` accepts one file plus repeated `pipelines=<name>` form fields and runs the same upload through each. Valid names are defined in `COMPARE_PIPELINE_SPECS` (`parser`, `ocr_tesseract`, `ocr_paddle`, `force_image_tesseract`, `force_image_paddle`, `vlm_pure`, `vlm_hybrid`). Pipelines whose backend is unavailable on the host are returned as `{ok: false, available: false}` rather than failing the whole run. `/capabilities` includes a `compare.available_pipelines` list derived from `detect_compare_pipelines()`. The UI "Compare Pipelines" job renders each pipeline's raw-text preview, the `vlm_trace` table, and a Diff Text tool for two pipelines — this is the debugging workflow for recognition quality.

### Extraction profiles

`app/profiles/` provides named extraction profiles that bundle OCR backend, classifier strategy, and enrichment settings per document family. Profiles are YAML files (`general.yaml`, `tax.yaml`, `financial.yaml`, `receipt.yaml`) loaded lazily by `app/profiles/loader.py`. Callers select a profile via the `extraction_profile` Form field on `/extract` and `/extract/structured`. Explicit Form field values override profile defaults (sentinel-based: overridable fields default to `None`, profile value wins when caller sends nothing). No profile = backward-compatible behavior. `app/classification/routing.py::classify_with_profile()` dispatches to the right classifier chain based on the profile's `classifier.strategy` setting. `/capabilities` lists available profiles and classifier model availability.

### Schemas are the contract

`app/schemas.py` defines `ExtractionPayload`, `Segment`, `Chunk`, `StructuredExtractionResponse`, etc. These are returned verbatim to API callers, so adding/removing fields is a public-API change. Prefer adding to `extra` (a free-form dict) for new metadata. `ClassificationResult.strategy` is `Literal["rules", "hint", "layout", "donut"]`.

## Tests

Tests live in `tests/` and use pytest + httpx. They cover the API endpoints, individual extractors, classification rules, and the structured per-form pipelines (`test_structured_*.py`). OCR-dependent tests gracefully skip or assert structured warnings when binaries are missing — match that pattern when adding new OCR-related tests.

## CI / deployment

CI runs on a self-hosted runner labeled `suru-devops` / `loci-extract` (see `.github/workflows/`). The runner is CPU-only — all GPU code paths must skip cleanly in CI. Workflows publish images to `ghcr.io/sudobot99/loci-extract` and deploy via `compose.prod.yaml` on `SURU-DEVOPS`.

Two Docker images are published: the CPU image (`:main` tag, from `Dockerfile`) and the GPU image (`:gpu` tag, from `Dockerfile.gpu`). `compose.gpu.yaml` is a standalone compose file that runs the GPU image with NVIDIA runtime. `compose.gpu-passthrough.yaml` is the legacy overlay that just adds `--gpus` env vars to whatever image is set. GPU env vars: `LOCI_EXTRACT_DEFAULT_OCR_BACKEND`, `LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU`, `LOCI_EXTRACT_FORCE_CUDA`.

## Further reading

`docs/architecture.md`, `docs/tax-ingestion.md`, `docs/review-workflow.md`, `docs/local-models.md` contain longer-form design notes — read these before making non-trivial changes to the relevant subsystem.
