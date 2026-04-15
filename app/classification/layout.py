"""Layout-based document classification using PaddleOCR PP-Structure.

This module is import-guarded: it returns None immediately when CUDA is unavailable
or PP-Structure cannot be imported. It never raises into the hot path.

The layout classifier runs on page images and infers document type from
region labels (text/title/table/figure/list) using histogram heuristics.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from app.schemas import StructuredDocType

logger = logging.getLogger("loci.classification.layout")

_layout_engine = None


def _pp_structure_available() -> bool:
    try:
        from paddleocr import PPStructure  # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_is_available() -> bool:
    from app.capabilities import cuda_available
    return cuda_available()["available"]


def _get_layout_engine():
    """Lazy-init the PP-Structure layout engine (singleton)."""
    global _layout_engine
    if _layout_engine is None:
        from paddleocr import PPStructure
        _layout_engine = PPStructure(
            table=False,
            ocr=False,
            layout=True,
            show_log=False,
        )
    return _layout_engine


@dataclass
class LayoutClassification:
    doc_type: StructuredDocType
    confidence: float
    strategy: str = "layout"
    matched_signals: list[str] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)


# Region-label histogram rules for document type inference
_LAYOUT_RULES: list[dict[str, Any]] = [
    {
        "doc_type": "w2",
        "signals": ["many_small_text_regions", "structured_form_layout"],
        "min_text_regions": 8,
        "max_table_regions": 1,
        "title_keywords": ["w-2", "wage"],
    },
    {
        "doc_type": "1099-nec",
        "signals": ["many_small_text_regions", "structured_form_layout"],
        "min_text_regions": 6,
        "max_table_regions": 1,
        "title_keywords": ["1099", "nonemployee"],
    },
    {
        "doc_type": "financial_statement",
        "signals": ["dominant_table_layout", "structured_rows"],
        "min_table_regions": 1,
        "min_text_regions": 3,
        "title_keywords": ["balance sheet", "financial", "account"],
    },
    {
        "doc_type": "receipt",
        "signals": ["narrow_layout", "few_regions"],
        "max_text_regions": 15,
        "max_table_regions": 2,
        "title_keywords": ["total", "subtotal", "receipt"],
    },
]


def _classify_from_regions(
    regions: list[dict[str, Any]],
    *,
    filename: str = "",
) -> LayoutClassification | None:
    """Apply heuristic rules over PP-Structure region labels."""
    region_types = [r.get("type", "").lower() for r in regions]
    text_count = sum(1 for t in region_types if t == "text")
    table_count = sum(1 for t in region_types if t == "table")

    # Extract title text if available
    title_texts = " ".join(
        str(r.get("res", "")).lower()
        for r in regions
        if r.get("type", "").lower() == "title"
    )
    lower_filename = filename.lower()

    best_match: LayoutClassification | None = None
    best_confidence = 0.0

    for rule in _LAYOUT_RULES:
        confidence = 0.0
        signals: list[str] = []

        # Check title keywords
        keyword_hits = sum(
            1 for kw in rule.get("title_keywords", [])
            if kw in title_texts or kw in lower_filename
        )
        if keyword_hits > 0:
            confidence += 0.3 * keyword_hits
            signals.append(f"title_keyword_match:{keyword_hits}")

        # Check region count heuristics
        min_text = rule.get("min_text_regions", 0)
        max_text = rule.get("max_text_regions", 999)
        min_table = rule.get("min_table_regions", 0)
        max_table = rule.get("max_table_regions", 999)

        if min_text <= text_count <= max_text:
            confidence += 0.2
            signals.append(f"text_regions:{text_count}")
        if min_table <= table_count <= max_table:
            confidence += 0.15
            signals.append(f"table_regions:{table_count}")

        # Bonus for structured form signals
        if rule["doc_type"] in ("w2", "1099-nec") and text_count >= 8 and table_count <= 1:
            confidence += 0.15
            signals.append("structured_form_layout")
        if rule["doc_type"] == "financial_statement" and table_count >= 1:
            confidence += 0.2
            signals.append("dominant_table_layout")

        if confidence > best_confidence and confidence >= 0.4:
            best_confidence = confidence
            best_match = LayoutClassification(
                doc_type=rule["doc_type"],
                confidence=min(confidence, 0.95),
                strategy="layout",
                matched_signals=signals,
                regions=[
                    {"type": r.get("type"), "bbox": r.get("bbox")}
                    for r in regions
                ],
            )

    return best_match


def classify_layout(
    image: Image.Image | Path,
    *,
    page_number: int = 1,
    filename: str = "",
) -> LayoutClassification | None:
    """Classify a page image using PP-Structure layout analysis.

    Returns None when CUDA is missing, PP-Structure is not installed,
    or layout classification is disabled via env var. Never raises.
    """
    # Check env override
    if os.getenv("LOCI_EXTRACT_LAYOUT_CLASSIFIER", "auto").strip().lower() == "off":
        return None

    if not _cuda_is_available():
        return None

    if not _pp_structure_available():
        return None

    try:
        import numpy as np

        engine = _get_layout_engine()

        if isinstance(image, Path):
            image = Image.open(image)

        img_array = np.array(image.convert("RGB"))
        result = engine(img_array)

        if not result:
            return None

        return _classify_from_regions(result, filename=filename)

    except Exception as exc:
        logger.warning("Layout classification failed: %s", exc)
        return None
