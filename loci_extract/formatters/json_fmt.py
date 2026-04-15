"""JSON formatter — pretty-printed Extraction."""

from __future__ import annotations

from loci_extract.schema import Extraction


def format_extraction(extraction: Extraction) -> str:
    return extraction.model_dump_json(indent=2)


__all__ = ["format_extraction"]
