"""Output formatters — json | csv | lacerte | txf.

Use ``format_extraction(extraction, fmt)`` to dispatch by name.
"""

from __future__ import annotations

from loci_extract.schema import Extraction

from . import csv_fmt, json_fmt, lacerte_fmt, txf_fmt

_DISPATCH = {
    "json": json_fmt.format_extraction,
    "csv": csv_fmt.format_extraction,
    "lacerte": lacerte_fmt.format_extraction,
    "txf": txf_fmt.format_extraction,
}


def format_extraction(extraction: Extraction, fmt: str) -> str:
    if fmt not in _DISPATCH:
        raise ValueError(f"Unknown format {fmt!r}. Valid: {sorted(_DISPATCH)}")
    return _DISPATCH[fmt](extraction)


__all__ = ["format_extraction"]
