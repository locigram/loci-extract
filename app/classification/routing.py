"""Profile-aware classifier routing.

Dispatches to the appropriate classification strategy based on the extraction
profile's classifier config. Falls back to the rule-based classifier when
GPU-dependent classifiers are unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image

from app.classification.rules import classify_document
from app.schemas import ClassificationResult

if TYPE_CHECKING:
    from app.profiles.schema import ExtractionProfile

logger = logging.getLogger("loci.classification.routing")


def classify_with_profile(
    *,
    profile: ExtractionProfile | None,
    filename: str,
    mime_type: str,
    raw_text: str,
    doc_type_hint: str | None = None,
    page_image: Image.Image | None = None,
) -> ClassificationResult:
    """Classify a document using the profile's classifier strategy.

    When no profile is provided, delegates to classify_document() directly
    (backward-compatible behavior).
    """
    # No profile: existing behavior
    if profile is None:
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
            doc_type_hint=doc_type_hint,
            page_image=page_image,
        )

    # Hint always wins regardless of strategy
    if doc_type_hint:
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
            doc_type_hint=doc_type_hint,
        )

    strategy = profile.classifier.strategy
    threshold = profile.classifier.confidence_threshold
    model_name = profile.classifier.model_name

    if strategy == "rules":
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
        )

    if strategy == "layout":
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
            page_image=page_image,
        )

    if strategy == "donut-irs":
        result = _try_donut(page_image, model_name=model_name, threshold=threshold)
        if result is not None:
            return result
        # Fallback to rules
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
        )

    if strategy == "vlm":
        result = _try_vlm_classify(profile, page_image, threshold=threshold)
        if result is not None:
            return result
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
        )

    if strategy == "auto":
        # Try VLM first (if profile has vlm config and image available)
        if page_image is not None and profile.vlm.base_url:
            result = _try_vlm_classify(profile, page_image, threshold=threshold)
            if result is not None:
                return result

        # Try donut-irs (if model_name is set and image available)
        if model_name and page_image is not None:
            result = _try_donut(page_image, model_name=model_name, threshold=threshold)
            if result is not None:
                return result

        # Try layout (PP-Structure)
        if page_image is not None:
            result = classify_document(
                filename=filename,
                mime_type=mime_type,
                raw_text=raw_text,
                page_image=page_image,
            )
            if result.strategy in ("layout", "hint") and result.confidence >= threshold:
                return result

        # Fallback to rules
        return classify_document(
            filename=filename,
            mime_type=mime_type,
            raw_text=raw_text,
        )

    # Unknown strategy — fall through to rules
    logger.warning("Unknown classifier strategy: %s, falling back to rules", strategy)
    return classify_document(
        filename=filename,
        mime_type=mime_type,
        raw_text=raw_text,
    )


def _try_vlm_classify(
    profile: ExtractionProfile,
    page_image: Image.Image | None,
    *,
    threshold: float,
) -> ClassificationResult | None:
    """Attempt VLM-based classification; returns None if unavailable or low confidence."""
    if page_image is None:
        return None
    try:
        from app.extractors.vlm import vlm_classify
        from app.llm.config import client_from_config, get_vlm_client

        # Use profile-specific VLM endpoint, or fall back to global VLM endpoint
        client = None
        if profile.vlm.base_url and profile.vlm.model:
            client = client_from_config(profile.vlm.model_dump())
        if client is None:
            client = get_vlm_client()
        if client is None:
            return None

        result = vlm_classify(client, page_image)
        if result is not None and result.confidence >= threshold:
            return result
        return None
    except Exception as exc:
        logger.warning("VLM classification failed: %s", exc)
        return None


def _try_donut(
    page_image: Image.Image | None,
    *,
    model_name: str | None,
    threshold: float,
) -> ClassificationResult | None:
    """Attempt Donut classification; returns None if unavailable or low confidence."""
    if page_image is None:
        return None
    try:
        from app.classification.donut_classifier import classify_with_donut

        return classify_with_donut(
            page_image,
            model_name=model_name or "hsarfraz/donut-irs-tax-docs-classifier",
            confidence_threshold=threshold,
        )
    except Exception as exc:
        logger.warning("Donut classifier failed: %s", exc)
        return None
