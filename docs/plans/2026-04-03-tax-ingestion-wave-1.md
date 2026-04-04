# Tax ingestion wave 1 implementation plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a first-wave tax-ingestion pipeline on top of `loci-extract` that supports classification and structured extraction for W-2, 1099-NEC, receipts, and 1040 package summaries while preserving raw extraction as the canonical source artifact.

**Architecture:** Keep `loci-extract` parser-first and stateless. Add a second API path that accepts the same uploaded files, runs the existing `/extract` flow internally, classifies the document, applies a type-specific structured extractor, normalizes and validates fields, and returns a combined JSON payload with review metadata. Raw extraction remains first-class and is always returned or embedded.

**Tech Stack:** FastAPI, Pydantic v2, existing extractor modules, deterministic regex/rule-based classification, schema-driven structured extraction, pytest.

---

## Scope for wave 1

Implement only these structured document families:

1. `w2`
2. `1099-nec`
3. `receipt`
4. `tax_return_package` (1040 package summary only, not full schedule-level line extraction)

Out of scope for wave 1:

- 1099-MISC / 1099-INT / 1099-DIV extraction logic
- schedule-level structured extraction beyond package summary
- persistent storage inside `loci-extract`
- background job queue
- fine-tuning or LLM-dependent extraction
- full field-level OCR confidence scoring per token

---

## Desired API shape

Add a new endpoint:

- `POST /extract/structured`

It should accept the same multipart form inputs as `/extract` plus optional structured-extraction controls.

### Request form fields

- `file` — required upload
- `include_chunks` — bool, default `true`
- `ocr_strategy` — `auto|always|never`, default `auto`
- `doc_type_hint` — optional string; if provided and supported, skip or bias classification
- `mask_pii` — bool, default `true`

### Response shape

Return a combined payload like:

```json
{
  "document_id": "...",
  "classification": {
    "doc_type": "w2",
    "confidence": 0.98,
    "strategy": "rules",
    "matched_signals": ["form w-2", "wage and tax statement"]
  },
  "raw_extraction": {
    "document_id": "...",
    "metadata": {},
    "extraction": {},
    "raw_text": "...",
    "segments": [],
    "chunks": [],
    "extra": {}
  },
  "structured": {
    "document_type": "w2",
    "schema_version": "1.0",
    "fields": {},
    "review": {
      "requires_human_review": false,
      "review_reasons": [],
      "missing_fields": [],
      "validation_errors": []
    }
  },
  "extra": {
    "mask_pii": true
  }
}
```

Use nested `raw_extraction` instead of flattening the existing payload into the new response.

---

## File map

### New modules to create

- `app/classification/__init__.py`
- `app/classification/base.py`
- `app/classification/rules.py`
- `app/structured/__init__.py`
- `app/structured/base.py`
- `app/structured/router.py`
- `app/structured/common.py`
- `app/structured/w2.py`
- `app/structured/form_1099_nec.py`
- `app/structured/receipt.py`
- `app/structured/tax_return_package.py`
- `app/normalization.py`
- `app/review.py`
- `tests/test_classification.py`
- `tests/test_structured_extractors.py`
- `tests/test_structured_api.py`

### Existing files to modify

- `app/main.py`
- `app/router.py` only if a reusable extraction helper is needed
- `app/schemas.py`
- `README.md`
- `docs/tax-ingestion.md`

---

## Data model plan

Add new Pydantic models in `app/schemas.py` for the structured path.

### New models to add

```python
class ClassificationResult(BaseModel):
    doc_type: Literal["w2", "1099-nec", "receipt", "tax_return_package", "unknown"]
    confidence: float = 0.0
    strategy: Literal["rules", "hint"] = "rules"
    matched_signals: list[str] = Field(default_factory=list)

class ReviewMetadata(BaseModel):
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)

class StructuredDocument(BaseModel):
    document_type: str
    schema_version: str = "1.0"
    fields: dict[str, Any] = Field(default_factory=dict)
    review: ReviewMetadata = Field(default_factory=ReviewMetadata)

class StructuredExtractionResponse(BaseModel):
    document_id: str
    classification: ClassificationResult
    raw_extraction: ExtractionPayload
    structured: StructuredDocument
    extra: dict[str, Any] = Field(default_factory=dict)
```

Keep wave 1 simple: use `fields: dict[str, Any]` in the generic response, even though internal extractor modules should still build typed dicts carefully.

---

# Task plan

## Task 1: Create classification package skeleton

**Objective:** Add a minimal classification package with a reusable interface.

**Files:**
- Create: `app/classification/__init__.py`
- Create: `app/classification/base.py`
- Create: `app/classification/rules.py`
- Test: `tests/test_classification.py`

**Step 1: Write failing tests**

Add tests for:

- W-2 text classified as `w2`
- 1099-NEC text classified as `1099-nec`
- receipt-like text classified as `receipt`
- 1040-like text classified as `tax_return_package`
- unknown text classified as `unknown`
- `doc_type_hint` overrides rules when valid

Example test skeleton:

```python
from app.classification.rules import classify_document


def test_classify_w2_text() -> None:
    result = classify_document(
        filename="w2.pdf",
        mime_type="application/pdf",
        raw_text="Form W-2 Wage and Tax Statement 2024",
        doc_type_hint=None,
    )
    assert result.doc_type == "w2"
    assert result.strategy == "rules"
```

**Step 2: Run test to verify failure**

Run:

```bash
source .venv/bin/activate
pytest -q tests/test_classification.py
```

Expected: fail because classification modules do not exist.

**Step 3: Implement base classifier interface**

In `app/classification/base.py`, create a small protocol or abstract base if needed, but keep it lightweight.

**Step 4: Implement rule-based classifier**

In `app/classification/rules.py`, implement a deterministic classifier using case-insensitive keyword/rule matching.

Required signals for wave 1:

- `w2`: `form w-2`, `wage and tax statement`
- `1099-nec`: `1099-nec`, `nonemployee compensation`
- `tax_return_package`: `form 1040`, `u.s. individual income tax return`
- `receipt`: presence of `total`, plus either `subtotal`, `tax`, or common receipt markers like `visa`, `mastercard`, `change`, `receipt`

Return `ClassificationResult` with `matched_signals` populated.

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_classification.py
```

**Step 6: Commit**

```bash
git add app/classification app/schemas.py tests/test_classification.py
git commit -m "feat: add rule-based tax document classification"
```

---

## Task 2: Add normalization helpers

**Objective:** Centralize amount/date/masking helpers for structured extractors.

**Files:**
- Create: `app/normalization.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Add tests for helpers:

- parse currency-like strings such as `$12,345.67`
- parse last4 from masked SSN/TIN patterns
- normalize date strings only when obvious
- mask SSN/TIN strings

Example:

```python
from app.normalization import parse_amount, mask_identifier, last4


def test_parse_amount_with_commas_and_dollar_sign() -> None:
    assert parse_amount("$12,345.67") == 12345.67
```

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k normalization
```

**Step 3: Implement helpers**

In `app/normalization.py`, implement only what wave 1 needs:

- `parse_amount(text: str | None) -> float | None`
- `extract_last4(text: str | None) -> str | None`
- `mask_identifier(text: str | None) -> str | None`
- `normalize_whitespace(text: str | None) -> str`
- `find_first_date(text: str) -> str | None` using conservative regex patterns

Do not overbuild tax-specific formatting logic yet.

**Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k normalization
```

**Step 5: Commit**

```bash
git add app/normalization.py tests/test_structured_extractors.py
git commit -m "feat: add normalization helpers for structured extraction"
```

---

## Task 3: Add review metadata helpers

**Objective:** Centralize how review flags, missing fields, and validation errors are produced.

**Files:**
- Create: `app/review.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Add tests covering:

- missing required fields produce `requires_human_review = True`
- OCR-heavy provenance can trigger review reasons
- validation errors populate correctly

Example:

```python
from app.review import build_review_metadata


def test_missing_required_fields_require_review() -> None:
    review = build_review_metadata(
        required_fields={"employee.full_name": None},
        validation_errors=[],
        raw_extra={},
    )
    assert review.requires_human_review is True
    assert "employee.full_name" in review.missing_fields
```

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k review
```

**Step 3: Implement review helper**

Implement a helper like:

```python
def build_review_metadata(
    *,
    required_fields: dict[str, object],
    validation_errors: list[str],
    raw_extra: dict[str, object],
) -> ReviewMetadata:
    ...
```

Rules for wave 1:

- any missing required field -> review required
- any validation error -> review required
- any page in `page_provenance` with `source == "none"` for tax docs -> add review reason
- OCR-only pages do not automatically fail, but should add review reason for tax forms other than receipts

**Step 4: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k review
```

**Step 5: Commit**

```bash
git add app/review.py tests/test_structured_extractors.py
git commit -m "feat: add review metadata helpers for tax ingestion"
```

---

## Task 4: Add structured extraction base/router

**Objective:** Create a small routing layer that maps classified document types to extractor functions.

**Files:**
- Create: `app/structured/__init__.py`
- Create: `app/structured/base.py`
- Create: `app/structured/common.py`
- Create: `app/structured/router.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Add tests for:

- known type routes to correct extractor
- unknown type returns minimal structured payload requiring review

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k router
```

**Step 3: Implement common helpers**

In `app/structured/common.py`, add helpers to:

- search regex in `raw_text`
- pull candidate lines from `segments`
- build minimal fallback structured docs

**Step 4: Implement router**

In `app/structured/router.py`, create a function:

```python
def build_structured_document(
    classification: ClassificationResult,
    raw_payload: ExtractionPayload,
    *,
    mask_pii: bool = True,
) -> StructuredDocument:
    ...
```

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k router
```

**Step 6: Commit**

```bash
git add app/structured tests/test_structured_extractors.py
git commit -m "feat: add structured extraction router"
```

---

## Task 5: Implement W-2 structured extractor

**Objective:** Add deterministic extraction for first-wave W-2 fields.

**Files:**
- Create: `app/structured/w2.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Use synthetic W-2-like raw text fixtures and verify extraction of:

- tax year
- employee full name
- employee SSN last4
- employer name
- employer EIN
- box 1 wages
- box 2 federal withholding
- review metadata for missing required values

Example fixture text:

```python
raw_text = """
Form W-2 Wage and Tax Statement 2024
Employee's social security number XXX-XX-1234
Employer identification number 12-3456789
Employee name John Q Public
Employer name Example Payroll Inc
1 Wages, tips, other compensation 85000.00
2 Federal income tax withheld 12000.00
"""
```

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k w2
```

**Step 3: Implement extractor**

In `app/structured/w2.py`, implement a function that builds a `StructuredDocument` with fields shaped like:

```python
{
    "tax_year": 2024,
    "employee": {...},
    "employer": {...},
    "boxes": {...},
    "state_local_entries": [...],
}
```

Required for wave 1:

- support empty list defaults for repeated sections
- mask SSN if `mask_pii=True`
- only extract starter boxes 1–6 plus empty defaults for others

Do not attempt full W-2 box coverage yet.

**Step 4: Add validation**

Validate at least:

- tax year present or inferable
- employee name present
- employer name present
- box 1 numeric if present

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k w2
```

**Step 6: Commit**

```bash
git add app/structured/w2.py tests/test_structured_extractors.py
git commit -m "feat: add wave-1 W-2 structured extractor"
```

---

## Task 6: Implement 1099-NEC structured extractor

**Objective:** Add deterministic extraction for first-wave 1099-NEC fields.

**Files:**
- Create: `app/structured/form_1099_nec.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Verify extraction of:

- tax year
- recipient name
- recipient TIN last4
- payer name
- payer TIN
- box 1 nonemployee compensation
- box 4 federal withholding

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k nec
```

**Step 3: Implement extractor**

Use deterministic regex and line scanning. Return starter fields only.

**Step 4: Add validation**

Require:

- payer or recipient name present
- box 1 parse success when visible

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k nec
```

**Step 6: Commit**

```bash
git add app/structured/form_1099_nec.py tests/test_structured_extractors.py
git commit -m "feat: add wave-1 1099-nec structured extractor"
```

---

## Task 7: Implement receipt structured extractor

**Objective:** Add deterministic structured extraction for receipts.

**Files:**
- Create: `app/structured/receipt.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Verify extraction of:

- merchant name from first non-empty lines
- transaction date
- subtotal
- tax
- total
- line-item fallback as empty list if unavailable
- review required when total is missing

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k receipt
```

**Step 3: Implement extractor**

Heuristics for wave 1:

- merchant name: first non-generic line
- date: first recognized date pattern
- totals: regex patterns anchored to `subtotal`, `tax`, `tip`, `total`
- payment hint: detect strings like `visa`, `amex`, `mastercard`, `cash`

Do not overpromise line-item extraction in wave 1.

**Step 4: Add validation**

Require at least one of:

- merchant + total
- date + total

Otherwise require human review.

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k receipt
```

**Step 6: Commit**

```bash
git add app/structured/receipt.py tests/test_structured_extractors.py
git commit -m "feat: add wave-1 receipt structured extractor"
```

---

## Task 8: Implement tax return package summary extractor

**Objective:** Add high-level 1040 package summary extraction without full line-level schedule support.

**Files:**
- Create: `app/structured/tax_return_package.py`
- Test: `tests/test_structured_extractors.py`

**Step 1: Write failing tests**

Verify extraction of:

- tax year
- primary form = `1040`
- taxpayer primary name
- filing status if visible
- summary fields when lines are present:
  - total income
  - AGI
  - taxable income
  - total tax
  - refund
  - amount owed
- attached form hints from text markers like `Schedule C`

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k tax_return
```

**Step 3: Implement extractor**

Use page text scanning and conservative line regexes.

Return fields shaped like:

```python
{
    "tax_year": 2024,
    "primary_form": "1040",
    "taxpayer": {...},
    "filing": {...},
    "summary": {...},
    "attached_forms": [...],
    "pages": [...],
}
```

For `pages`, use `raw_payload.extra.get("page_provenance", [])` to create a simple page list with `page_number` and `form_type_hint="1040"` when appropriate.

**Step 4: Add validation**

Require human review if:

- no taxpayer name
- no year
- no core form marker
- page provenance includes `none`

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_extractors.py -k tax_return
```

**Step 6: Commit**

```bash
git add app/structured/tax_return_package.py tests/test_structured_extractors.py
git commit -m "feat: add wave-1 tax return package structured extractor"
```

---

## Task 9: Add `/extract/structured` API endpoint

**Objective:** Expose the new structured tax-ingestion flow via FastAPI.

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_structured_api.py`

**Step 1: Write failing API tests**

Add tests for:

- structured W-2 response
- structured 1099-NEC response
- unknown document response with review required
- `doc_type_hint` override
- `mask_pii=true` masks identifiers
- `include_chunks=false` preserves raw payload with empty chunks

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_api.py
```

**Step 3: Implement reusable internal helper**

In `app/main.py`, factor the current extraction body into a private helper if needed so `/extract` and `/extract/structured` share upload handling.

Suggested helper shape:

```python
def _extract_payload_from_upload(...) -> ExtractionPayload:
    ...
```

**Step 4: Implement endpoint**

Add:

```python
@app.post("/extract/structured")
async def extract_structured(...):
    ...
```

The endpoint should:

1. parse upload and shared form fields
2. call the shared raw extraction helper
3. classify the document
4. build structured output
5. return `StructuredExtractionResponse.model_dump()`

**Step 5: Run tests to verify pass**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_api.py
```

**Step 6: Commit**

```bash
git add app/main.py tests/test_structured_api.py app/classification app/structured app/review.py app/normalization.py app/schemas.py
git commit -m "feat: add structured tax extraction API"
```

---

## Task 10: Add end-to-end regression coverage

**Objective:** Validate the full structured path against representative fixture payloads and review behavior.

**Files:**
- Modify: `tests/test_structured_api.py`
- Modify: `tests/test_structured_extractors.py`

**Step 1: Add missing edge-case tests**

Cover at least:

- OCR-derived page provenance triggers review on W-2/1099/1040
- receipt missing total requires review
- unknown document classification yields `unknown`
- blank document through `/extract/structured` still returns review metadata

**Step 2: Run tests**

```bash
source .venv/bin/activate
pytest -q tests/test_classification.py tests/test_structured_extractors.py tests/test_structured_api.py
```

**Step 3: Run full suite**

```bash
source .venv/bin/activate
pytest -q
```

**Step 4: Commit**

```bash
git add tests
git commit -m "test: add regression coverage for structured tax extraction"
```

---

## Task 11: Document the new endpoint and wave-1 limitations

**Objective:** Update docs so users know how to call the structured endpoint and what it does.

**Files:**
- Modify: `README.md`
- Modify: `docs/tax-ingestion.md`

**Step 1: Update README**

Add:

- `POST /extract/structured` usage examples
- sample W-2 invocation
- sample receipt invocation
- statement that structured extraction is wave-1, deterministic, and review-aware

**Step 2: Update tax ingestion doc**

Add a short “Implemented in wave 1” section once the feature lands.

**Step 3: Run quick sanity tests**

```bash
source .venv/bin/activate
pytest -q tests/test_structured_api.py
```

**Step 4: Commit**

```bash
git add README.md docs/tax-ingestion.md
git commit -m "docs: document structured tax extraction endpoint"
```

---

## Task 12: Final integration review

**Objective:** Verify that the full feature is coherent and safe to merge.

**Files:**
- Review all modified files

**Step 1: Run full test suite**

```bash
source .venv/bin/activate
pytest -q
```

**Step 2: Review the diff**

```bash
git diff --stat origin/main...HEAD
```

**Step 3: Manually verify example flows**

Run local examples against a sample W-2-like text/PDF and a receipt image or text fixture.

**Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: finalize wave-1 tax ingestion integration"
```

---

## Implementation notes and constraints

### 1. Keep raw extraction canonical

The new structured path must embed or reference the full raw extraction payload. Do not build a structured-only endpoint that loses provenance.

### 2. Prefer deterministic parsing first

Use rules, regexes, and conservative heuristics for wave 1. Avoid introducing LLM dependencies into the core extraction path.

### 3. Mask PII by default

For wave 1, `mask_pii` should default to `true`. Use last4 variants for SSN/TIN where possible.

### 4. Review-aware by default

A partial or low-confidence result is still useful if it is explicit about review needs. Do not silently return high-confidence-looking garbage.

### 5. Keep scope tight

Do not add persistent storage, job queues, or support for the entire 1099 family in this wave.

---

## Verification checklist

Before considering the wave complete:

- [ ] `/extract` still behaves exactly as before
- [ ] `/extract/structured` returns combined raw + structured output
- [ ] W-2 extraction works on representative fixture text
- [ ] 1099-NEC extraction works on representative fixture text
- [ ] receipt extraction works on representative fixture text
- [ ] 1040 package summary extraction works on representative fixture text
- [ ] unknown docs return review-required structured output
- [ ] masking is enabled by default
- [ ] provenance-driven review reasons appear when expected
- [ ] full pytest suite passes

---

## Suggested branching / commit strategy

Use small commits per task exactly as listed above. Do not batch the entire wave into one giant commit.

---

## After wave 1

Next likely follow-up phases:

1. add `1099-misc`, `1099-int`, `1099-div`
2. add packet splitting for mixed uploads
3. add field-level evidence snippets
4. add optional persistence adapters
5. add schedule-level structured tax return extraction
6. add evaluation corpus and benchmarking harness
