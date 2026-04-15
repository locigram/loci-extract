"""Donut-based IRS tax document classifier.

Uses hsarfraz/donut-irs-tax-docs-classifier (74M params, MIT) to classify
28 IRS document types from page images. Import-guarded — returns None when
transformers/torch are not installed or CUDA is unavailable.
"""

from __future__ import annotations

import logging
import re
import threading

from PIL import Image

from app.schemas import ClassificationResult, StructuredDocType

logger = logging.getLogger("loci.classification.donut")

_DEFAULT_MODEL = "hsarfraz/donut-irs-tax-docs-classifier"
_IMAGE_SIZE = (1920, 2560)  # width x height per model card

# Map Donut output labels to StructuredDocType values
_LABEL_TO_DOC_TYPE: dict[str, StructuredDocType] = {
    "Form W-2 Wage and Tax Statement": "w2",
    "1040 U.S. Individual Income Tax Return": "tax_return_package",
    "1040-NR U.S. Nonresident Alien Income Tax Return": "tax_return_package",
    "1040 SCHEDULE 1 Additional Income and Adjustments to Income": "tax_return_package",
    "1040 SCHEDULE 2 Additional Taxes": "tax_return_package",
    "1040 SCHEDULE 3 Additional Credits and Payments": "tax_return_package",
    "1040 SCHEDULE 8812 Credits for Qualifying Children and Other Dependents": "tax_return_package",
    "1040 SCHEDULE A Itemized Deductions": "tax_return_package",
    "1040 SCHEDULE B Interest and Ordinary Dividends": "tax_return_package",
    "1040 SCHEDULE C Profit or Loss From Business": "tax_return_package",
    "1040 SCHEDULE D Capital Gains and Losses": "tax_return_package",
    "1040 SCHEDULE E Supplemental Income and Loss": "tax_return_package",
    "1040 SCHEDULE SE Self-Employment Tax": "tax_return_package",
    "1040-NR SCHEDULE OI Other Information": "tax_return_package",
    "Form 1125-A Cost of Goods Sold": "financial_statement",
    "Form 8949 Sales and Other Dispositions of Capital Assets": "tax_return_package",
    "Form 8959 Additional Medicare Tax": "tax_return_package",
    "Form 8960 Net Investment Income Tax": "tax_return_package",
    "Form 8995 Qualified Business Income Deduction Simplified Computation": "tax_return_package",
    "Form 8995-A SCHEDULE A Specified Service Trades or Businesses": "tax_return_package",
}

# Allowlist of trusted HuggingFace model names for Donut classification.
# Only models on this list can be loaded — prevents arbitrary model injection via profile YAML.
_ALLOWED_MODELS = {
    "hsarfraz/donut-irs-tax-docs-classifier",
    "naver-clova-ix/donut-base-finetuned-rvlcdip",
}

_HF_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$")

# Singleton state
_processor = None
_model = None
_device = None
_lock = threading.Lock()


def _donut_available() -> bool:
    """Check if transformers and torch are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _validate_model_name(model_name: str) -> bool:
    """Validate model name against allowlist and format pattern."""
    if model_name in _ALLOWED_MODELS:
        return True
    if not _HF_MODEL_PATTERN.match(model_name):
        logger.warning("Rejected model name with invalid format: %s", model_name)
        return False
    logger.warning(
        "Model '%s' not in allowlist. Add it to _ALLOWED_MODELS in donut_classifier.py to use it.",
        model_name,
    )
    return False


def _load_model(model_name: str = _DEFAULT_MODEL):
    """Lazy-load the Donut classifier model and processor (thread-safe singleton)."""
    global _processor, _model, _device

    if _model is not None:
        return

    if not _validate_model_name(model_name):
        return

    with _lock:
        if _model is not None:
            return

        import torch
        from torch import nn
        from transformers import DonutProcessor, DonutSwinModel, DonutSwinPreTrainedModel

        class DonutForImageClassification(DonutSwinPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = config.num_labels
                self.swin = DonutSwinModel(config)
                self.dropout = nn.Dropout(0.5)
                self.classifier = nn.Linear(self.swin.num_features, config.num_labels)

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                outputs = self.swin(pixel_values)
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits

        _processor = DonutProcessor.from_pretrained(model_name)
        _model = DonutForImageClassification.from_pretrained(model_name)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()


def classify_with_donut(
    image: Image.Image | None,
    *,
    model_name: str = _DEFAULT_MODEL,
    confidence_threshold: float = 0.75,
) -> ClassificationResult | None:
    """Classify a document image using the Donut IRS classifier.

    Returns None when:
    - transformers/torch not installed
    - image is None
    - classification confidence is below threshold
    - any error occurs

    Never raises.
    """
    if image is None:
        return None

    if not _donut_available():
        return None

    try:
        import torch

        _load_model(model_name)

        # Resize to model's expected dimensions
        img = image.convert("RGB").resize(_IMAGE_SIZE, Image.Resampling.LANCZOS)

        with torch.no_grad():
            pixel_values = _processor(img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(_device)
            outputs = _model(pixel_values)

            # Get confidence via softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted_idx = torch.max(probabilities, 1)
            confidence = float(max_prob.cpu().numpy()[0])
            label_idx = int(predicted_idx.cpu().numpy()[0])

        predicted_label = _model.config.id2label.get(label_idx, "unknown")
        doc_type = _LABEL_TO_DOC_TYPE.get(predicted_label, "unknown")

        if confidence < confidence_threshold:
            logger.debug(
                "Donut classification below threshold: %s (%.2f < %.2f)",
                predicted_label, confidence, confidence_threshold,
            )
            return None

        return ClassificationResult(
            doc_type=doc_type,
            confidence=round(confidence, 4),
            strategy="donut",
            matched_signals=[f"donut_label:{predicted_label}"],
        )

    except Exception as exc:
        logger.warning("Donut classification failed: %s", exc)
        return None
