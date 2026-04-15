"""loci-extract — local-first tax document extraction.

Library entry points:
    from loci_extract import extract_document, extract_batch, ExtractionOptions
    from loci_extract.schema import Extraction, Document
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "__version__",
]

# Re-exports (deferred to avoid circular imports during package init):
from loci_extract.core import ExtractionOptions, extract_batch, extract_document  # noqa: E402
from loci_extract.schema import Document, Extraction  # noqa: E402

__all__ += [
    "ExtractionOptions",
    "extract_document",
    "extract_batch",
    "Document",
    "Extraction",
]
