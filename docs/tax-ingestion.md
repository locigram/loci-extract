# Tax document ingestion with `loci-extract`

## Purpose

This document describes how to use `loci-extract` as the **raw extraction layer** for tax-document workflows, including:

- W-2s
- 1099 forms
- receipts
- tax returns and 1040 packages

The key design decision is:

> `loci-extract` should handle **document ingestion, OCR, parser extraction, structural segmentation, and provenance**, while downstream systems handle **classification, structured field extraction, review, and persistence**.

`loci-extract` is currently an **ephemeral HTTP API**. It accepts a file, extracts content, returns JSON, and deletes the temporary upload. It does **not** currently persist extracted data on its own.

---

## Recommended architecture

For tax workflows, use a staged pipeline instead of trying to do everything in one prompt or one model call.

## Stage 1: Raw ingestion and extraction

Input documents can include:

- born-digital PDFs
- scanned PDFs
- phone photos of receipts
- multi-page tax return packages
- mixed upload bundles containing several unrelated forms

At this stage, call `loci-extract` and keep the full response.

Expected outputs from `loci-extract`:

- `raw_text` — canonical extracted text
- `segments` — page/table/paragraph/sheet structure
- `chunks` — optional retrieval-oriented chunks
- `extra` — provenance, warnings, OCR metadata, page-level source info

This raw payload is the **canonical source artifact** for all downstream processing.

### Why this matters

Do not throw away the raw extraction output after creating structured fields.

You want to preserve:

- reprocessability when schemas change
- auditability for tax-sensitive workflows
- debugging when OCR or field extraction fails
- lineage from structured field -> extracted text -> source page

---

## Stage 2: Document classification

After extraction, classify each document into a normalized tax document type.

Recommended starter taxonomy:

- `w2`
- `1099-nec`
- `1099-misc`
- `1099-int`
- `1099-div`
- `1040`
- `schedule-c`
- `schedule-e`
- `receipt`
- `invoice`
- `bank-statement`
- `supporting-tax-document`
- `unknown`

### Classification inputs

Use all of:

- filename
- MIME type
- `raw_text`
- first-page segments
- page-level provenance
- obvious form markers like `Form W-2`, `1099-NEC`, `Form 1040`, `Schedule C`

### Classification guidance

Start with deterministic classification first:

- regex / keyword rules
- box/line markers
- header patterns
- tax year markers

Then add model-assisted classification only when rules are insufficient.

For example:

- `Form W-2 Wage and Tax Statement` strongly indicates `w2`
- `Nonemployee compensation` + `1099-NEC` strongly indicates `1099-nec`
- `Form 1040 U.S. Individual Income Tax Return` strongly indicates `1040`
- merchant/date/total patterns with retail OCR noise often indicate `receipt`

---

## Stage 3: Type-specific structured extraction

Once classified, run a document-type-specific extractor that converts raw extraction into strict structured JSON.

This stage should be schema-driven.

Recommended strategy:

- W-2 -> W-2 schema
- 1099-NEC -> 1099-NEC schema
- receipts -> receipt schema
- 1040 package -> tax-return package schema + page/section breakdown

### Do not use one giant generic schema

Tax documents have different semantics and different validation rules.

A W-2 extractor should know about:

- employee identity
- employer identity
- wage/tax boxes
- state wage entries

A receipt extractor should know about:

- merchant
- date
- total
- line items
- tax/tip separation

A 1040 extractor should know about:

- taxpayer identity
- filing status
- dependents
- line-level totals
- attached schedules

---

## Stage 4: Normalization and validation

Structured extraction should not be the final step. Normalize and validate before accepting output.

Examples:

- amounts -> decimal-safe normalized numeric values
- dates -> ISO `YYYY-MM-DD` where possible
- tax year -> normalized integer
- SSN/TIN/EIN -> masked or normalized variants
- state amounts -> normalized repeated entries
- filing status -> normalized enum

Validation examples:

- W-2 wages should parse as money
- 1099-NEC nonemployee compensation should be numeric
- receipt total should be >= subtotal
- 1040 summary fields should not contain impossible negative values unless explicitly allowed
- tax year should match recognized form year markers if present

---

## Stage 5: Confidence scoring and human review

Tax-document ingestion should assume **some percentage of outputs require review**.

Do not design for silent full automation first.

Each structured output should include review metadata like:

- `requires_human_review`
- `review_reasons`
- `missing_fields`
- `validation_errors`
- `field_confidence`
- `document_confidence`

### Common review triggers

- OCR-only page with poor text quality
- low text length on expected content-heavy forms
- contradictory amounts
- missing payer/employer identity
- mismatched tax year markers
- unreadable receipt total/date
- multi-document upload not fully split
- forms detected but box values missing

---

## Stage 6: Persistence and downstream storage

`loci-extract` itself is not the storage layer.

Recommended storage pattern:

### Persist raw extraction artifact

Store the full raw extraction response:

- `document_id`
- source file reference or object-storage key
- raw `loci-extract` JSON payload
- ingestion timestamp
- caller/job metadata

### Persist normalized structured artifact separately

Store a second object/table for type-specific structured output:

- classified document type
- schema version
- normalized fields
- confidence/review state
- link back to raw extraction artifact

### Optional retrieval storage

If you also want RAG/search:

- store `chunks` in a vector DB
- store `raw_text` in searchable document storage
- keep source refs and page provenance for audit

### Recommended minimum durable record model

For each document, store at least:

1. original file reference
2. raw extraction payload
3. classified document type
4. structured normalized JSON
5. review status
6. provenance / schema version

---

## Recommended schemas

These are **starter schemas**, not final tax-compliance schemas.

Use them as normalized internal contracts for ingestion.

## W-2 schema

```json
{
  "document_type": "w2",
  "schema_version": "1.0",
  "tax_year": 2024,
  "employee": {
    "full_name": "",
    "ssn_last4": "",
    "address": ""
  },
  "employer": {
    "name": "",
    "ein": "",
    "address": ""
  },
  "boxes": {
    "1_wages_tips_other_comp": 0.0,
    "2_federal_income_tax_withheld": 0.0,
    "3_social_security_wages": 0.0,
    "4_social_security_tax_withheld": 0.0,
    "5_medicare_wages_and_tips": 0.0,
    "6_medicare_tax_withheld": 0.0,
    "7_social_security_tips": 0.0,
    "8_allocated_tips": 0.0,
    "10_dependent_care_benefits": 0.0,
    "11_nonqualified_plans": 0.0,
    "12_codes": [],
    "13_flags": {
      "statutory_employee": false,
      "retirement_plan": false,
      "third_party_sick_pay": false
    },
    "14_other": []
  },
  "state_local_entries": [
    {
      "state": "",
      "employer_state_id": "",
      "state_wages_tips": 0.0,
      "state_income_tax": 0.0,
      "local_wages_tips": 0.0,
      "local_income_tax": 0.0,
      "locality_name": ""
    }
  ],
  "review": {
    "requires_human_review": false,
    "review_reasons": [],
    "missing_fields": [],
    "validation_errors": []
  }
}
```

### W-2 notes

- Store only `ssn_last4` in broad-access normalized records unless full SSN is explicitly required and protected.
- Box 12 and Box 14 should stay structured as repeatable entries.
- State/local entries must be repeatable because some W-2s include multiple rows.

---

## 1099-NEC schema

```json
{
  "document_type": "1099-nec",
  "schema_version": "1.0",
  "tax_year": 2024,
  "recipient": {
    "name": "",
    "tin_last4": "",
    "address": ""
  },
  "payer": {
    "name": "",
    "tin": "",
    "address": ""
  },
  "boxes": {
    "1_nonemployee_compensation": 0.0,
    "4_federal_income_tax_withheld": 0.0,
    "5_state_tax_withheld": 0.0,
    "6_state_payer_number": "",
    "7_state_income": 0.0
  },
  "state_entries": [
    {
      "state": "",
      "state_tax_withheld": 0.0,
      "state_income": 0.0,
      "state_payer_number": ""
    }
  ],
  "review": {
    "requires_human_review": false,
    "review_reasons": [],
    "missing_fields": [],
    "validation_errors": []
  }
}
```

---

## Other 1099 variants

Use a shared family shape where possible, then variant-specific box mappings.

Recommended family types:

- `1099-nec`
- `1099-misc`
- `1099-int`
- `1099-div`

Suggested normalized shared fields:

- `document_type`
- `schema_version`
- `tax_year`
- `payer`
- `recipient`
- `boxes`
- `state_entries`
- `review`

Variant-specific extractors should map only the legally relevant boxes for that form type.

Do not force every 1099 variant into one flat schema if it causes ambiguous box meanings.

---

## Receipt schema

```json
{
  "document_type": "receipt",
  "schema_version": "1.0",
  "merchant": {
    "name": "",
    "address": "",
    "phone": ""
  },
  "transaction": {
    "date": "",
    "time": "",
    "currency": "USD",
    "payment_method_hint": "",
    "receipt_number": ""
  },
  "amounts": {
    "subtotal": 0.0,
    "tax": 0.0,
    "tip": 0.0,
    "fees": 0.0,
    "discount": 0.0,
    "total": 0.0
  },
  "line_items": [
    {
      "description": "",
      "quantity": 1,
      "unit_price": 0.0,
      "line_total": 0.0,
      "category_hint": ""
    }
  ],
  "review": {
    "requires_human_review": false,
    "review_reasons": [],
    "missing_fields": [],
    "validation_errors": []
  }
}
```

### Receipt notes

Receipts are usually less standardized than tax forms.

Expect more review cases for:

- date ambiguity
- total vs subtotal confusion
- OCR errors in merchant names
- weak line-item extraction
- handwritten tips

Receipts should generally have more aggressive confidence thresholds than W-2/1099 forms.

---

## 1040 / tax return package schema

Do **not** treat an entire return package as one flat record.

A tax return package should be represented as:

1. package-level summary
2. document-level sections/forms
3. line-level extracted values where supported

### Recommended package summary schema

```json
{
  "document_type": "tax_return_package",
  "schema_version": "1.0",
  "tax_year": 2024,
  "primary_form": "1040",
  "taxpayer": {
    "primary_name": "",
    "secondary_name": "",
    "ssn_last4": "",
    "address": ""
  },
  "filing": {
    "filing_status": "",
    "dependents_count": 0,
    "has_spouse": false
  },
  "summary": {
    "total_income": 0.0,
    "adjusted_gross_income": 0.0,
    "taxable_income": 0.0,
    "total_tax": 0.0,
    "total_payments": 0.0,
    "refund": 0.0,
    "amount_owed": 0.0
  },
  "attached_forms": [
    "1040",
    "schedule-1",
    "schedule-c"
  ],
  "pages": [
    {
      "page_number": 1,
      "form_type_hint": "1040",
      "section_hint": "identity-and-filing-status"
    }
  ],
  "review": {
    "requires_human_review": false,
    "review_reasons": [],
    "missing_fields": [],
    "validation_errors": []
  }
}
```

### 1040 package guidance

For actual tax returns, you should usually also keep:

- page-level classification
- schedule detection
- line-level extracted values per form/schedule
- structured linkage between summary totals and page sources

Examples of supported subdocuments:

- `1040`
- `schedule-1`
- `schedule-c`
- `schedule-e`
- attached W-2s
- attached 1099s

---

## Page-level provenance

Page-level provenance should be treated as first-class metadata for tax workflows.

For PDFs, `loci-extract` can already return `extra.page_provenance`, with entries like:

```json
[
  {"page_number": 1, "source": "parser", "has_text": true, "text_length": 1800},
  {"page_number": 2, "source": "ocr", "has_text": true, "text_length": 1200},
  {"page_number": 3, "source": "none", "has_text": false, "text_length": 0}
]
```

### How to use provenance in tax pipelines

Use it to:

- decide whether human review is required
- downgrade confidence for OCR-only pages
- detect blank or unreadable pages
- link extracted fields back to the source page
- support audits and correction workflows

### Recommended rule

If a critical field comes from a page with:

- `source = ocr`
- low OCR quality
- short text length
- nearby warning codes

then mark the field or document for review.

---

## OCR quality and review policy

Tax workflows should have explicit review rules for OCR-derived fields.

Suggested policy:

- parser-derived typed forms with consistent values -> low review burden
- OCR-derived typed forms -> medium review burden
- OCR-derived receipts / handwritten docs -> high review burden
- unreadable/mixed/low-text pages -> mandatory review

### Suggested field confidence model

For each extracted field, capture:

- `confidence_score` in `0.0` to `1.0`
- `source_page_numbers`
- `source_strategy` such as `parser` or `ocr`
- `evidence_text` or source snippet

This helps reviewers quickly confirm the extraction.

---

## PII handling and masking

Tax-document ingestion involves highly sensitive data.

Recommended baseline:

- mask SSNs/TINs in broad-access application records
- encrypt original documents and raw payloads at rest
- restrict access to unmasked identifiers
- log all access to tax-return documents
- use retention/deletion rules intentionally

### Suggested masking policy

Store these public-normalized variants unless you have a stronger protected store:

- SSN -> `***-**-1234`
- TIN -> masked + last4
- EIN -> possibly full only in protected records
- addresses -> full only where necessary for business logic

### Important implementation note

Keep a distinction between:

- **protected raw record**
- **normalized application-facing record**

Do not expose full raw PII widely just because the extractor found it.

---

## Tax year and form-version drift

Forms change over time.

Your ingestion system must explicitly track:

- tax year
- form family
- schema version
- extractor version

### Recommended approach

- detect year markers from headers when possible
- preserve year-specific layout hints if used
- version your normalized schemas
- avoid overfitting extractor logic to one year’s exact layout

For example, the same logical W-2 or 1099 field may appear in slightly different visual layouts across years/vendors/scans.

This is another reason to keep raw extraction payloads permanently available for replay.

---

## Splitting multi-document uploads

A single upload may contain:

- several W-2s back-to-back
- a 1040 package plus attachments
- mixed receipts in one PDF
- scanned packets with separator sheets

### Recommended handling

After raw extraction, run a split/detect stage that identifies document boundaries.

Boundary signals may include:

- `Form W-2` on a new page
- `Form 1099-NEC` on a new page
- `Schedule C` start page
- dramatic text/header changes
- barcode/separator page patterns

### Output recommendation

Represent multi-document splitting as:

- one original upload record
- N child extracted document records
- child records linked to page ranges

Do not force a mixed packet into one structured schema.

---

## Suggested storage model

A good durable storage design usually has three layers.

## Layer 1: Source file

- original binary file
- object storage key
- checksum
- upload metadata

## Layer 2: Raw extraction artifact

- full `loci-extract` response
- page provenance
- warnings
- OCR metadata
- extraction version

## Layer 3: Normalized structured artifact

- classified doc type
- schema version
- normalized fields
- field confidence
- review state
- downstream routing tags

This separation lets you:

- rerun structured extraction without re-uploading the original
- improve schemas later
- compare old vs new extraction logic
- support audit workflows cleanly

---

## Evaluation and labeled data collection

If you want the system to improve over time, collect review outcomes systematically.

## What to label

For each processed document, try to capture:

- final document type
- tax year
- corrected structured output
- field-level corrections
- OCR failure notes
- review reason
- whether the document was acceptable without edits

## Error categories to track

Use explicit error categories such as:

- `classification_error`
- `ocr_failure`
- `field_misread`
- `wrong_tax_year`
- `document_split_error`
- `missing_required_field`
- `amount_normalization_error`
- `receipt_total_mismatch`

## Metrics to track

Recommended metrics:

- document classification accuracy
- field extraction exact-match rate
- amount-field accuracy
- date accuracy
- required-field recall
- human-review rate
- correction rate after review
- OCR-only page failure rate

### Important evaluation note

Evaluate by document source/vendor/year, not just overall average.

A system can look “95% accurate” overall while failing badly on:

- a specific payroll provider’s W-2 layout
- one 1099 variant
- low-quality mobile receipt photos
- scanned 1040 packages

---

## Suggested implementation phases

## Phase 1: Tax-ingestion foundation

Build:

- document taxonomy
- deterministic classifier
- schema definitions
- raw artifact persistence
- normalized record persistence

Start with:

- W-2
- 1099-NEC
- 1099-MISC
- receipts
- 1040 package summary

## Phase 2: Review-aware extraction

Add:

- confidence scoring
- review flags
- field evidence snippets
- page-to-field lineage
- PII masking policy enforcement

## Phase 3: Multi-document packets and schedules

Add:

- upload splitting
- 1040 attachment detection
- schedule classification
- package-level linking across child documents

## Phase 4: Evaluation and training data

Add:

- reviewer corrections capture
- evaluation dashboard / batch reports
- regression corpus of known documents
- vendor/year segmentation in metrics

## Phase 5: Optional model-assisted enrichment

Only after deterministic extraction and review loops are stable, consider:

- model-assisted classification
- schema extraction fallback for difficult layouts
- receipt line-item cleanup
- semantic normalization over ambiguous fields

Do not make core OCR/text extraction depend entirely on an LLM.

---

## Suggested codebase additions on top of `loci-extract`

If implementing this inside or adjacent to `loci-extract`, likely next modules are:

- `app/classification/` for document typing
- `app/structured/` for type-specific JSON extraction
- `app/normalization/` for money/date/identifier normalization
- `app/review/` for confidence/review policy
- `app/tax_schemas/` for internal typed models

Example targets:

- `app/tax_schemas/w2.py`
- `app/tax_schemas/form_1099_nec.py`
- `app/tax_schemas/receipt.py`
- `app/tax_schemas/tax_return_package.py`

---

## Recommended starting scope

If you want to prove this workflow quickly, start with these four document families:

1. W-2
2. 1099-NEC
3. receipts
4. 1040 package summary

That gives you:

- one highly structured wage form
- one structured contractor form
- one messy OCR-heavy category
- one complex multi-page return package category

It is a strong first benchmark without trying to solve the entire IRS universe at once.

---

## Wave 1 status

Wave 1 is now implemented with a review-aware structured endpoint:

- `POST /extract/structured`

Current supported structured types:

- `w2`
- `1099-nec`
- `receipt`
- `tax_return_package`

Current behavior is intentionally conservative for OCR-heavy tax documents:

- parser/OCR provenance from the raw extraction layer is preserved
- image OCR and PDF OCR now use multi-pass preprocessing with best-pass selection
- OCR-backed tax documents are surfaced with review reasons instead of silent confidence
- low OCR score / weak OCR evidence can trigger review explicitly
- masking is enabled by default in structured output
- the full raw extraction payload remains embedded for audit and reprocessing
- structured outputs now include lightweight evidence snippets for key extracted fields

This is the correct starting posture for tax ingestion: preserve provenance, normalize deterministically, and require human review whenever OCR quality might compromise trust.

## Bottom line

The right way to support tax-document ingestion is:

1. preserve the full raw extraction artifact
2. classify document type explicitly
3. use strict schemas per form family
4. normalize and validate fields
5. require review when provenance/quality is weak
6. persist raw + normalized outputs separately
7. collect corrections as labeled data for future improvement

That architecture will scale much better than trying to “train on tax forms” before you have stable extraction, schemas, and review loops.
