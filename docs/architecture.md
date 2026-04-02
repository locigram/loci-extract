# loci-extract architecture

## Positioning

`loci-extract` is a standalone document ingestion and extraction service for ephemeral processing workflows.

It is designed to be consumed by multiple systems, including Locigram, but it does not assume Locigram-specific schemas.

## Core principles

1. **Raw extraction is first-class**
   - Preserve canonical extracted text for replay and reprocessing.
2. **Derived artifacts are secondary**
   - Chunks, summaries, and structured projections are derived from canonical extraction.
3. **Best tool per file type**
   - Route documents through specialized extractors instead of forcing one parser for everything.
4. **Normalized output contract**
   - Every extraction path returns the same top-level response shape.
5. **Ephemeral by default**
   - Temporary local processing, stateless API, optional durable job mode later.

## Initial extractor routing

- PDF -> PyMuPDF
- DOCX -> python-docx
- XLSX -> openpyxl
- Images -> Tesseract
- Plain text / markdown / JSON / CSV -> text decoder

## Planned next steps

- OCR fallback for scanned PDFs
- Better image preprocessing before OCR
- Table extraction and table-to-text normalization
- Async job API for slow OCR workloads
- Configurable extraction profiles per caller
- Optional local LLM post-processing using local model endpoints
