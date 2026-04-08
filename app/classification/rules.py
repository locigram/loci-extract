from __future__ import annotations

from app.schemas import ClassificationResult, StructuredDocType

SUPPORTED_DOC_TYPES: tuple[StructuredDocType, ...] = (
    "w2",
    "1099-nec",
    "receipt",
    "tax_return_package",
    "financial_statement",
    "unknown",
)


def _normalized(text: str) -> str:
    return " ".join((text or "").lower().split())


def classify_document(
    *,
    filename: str,
    mime_type: str,
    raw_text: str,
    doc_type_hint: str | None = None,
) -> ClassificationResult:
    hint = (doc_type_hint or "").strip().lower()
    if hint in SUPPORTED_DOC_TYPES and hint != "unknown":
        return ClassificationResult(
            doc_type=hint,
            confidence=1.0,
            strategy="hint",
            matched_signals=[f"doc_type_hint:{hint}"],
        )

    text = _normalized(raw_text)
    lower_name = (filename or "").lower()
    matched_signals: list[str] = []

    def has(*signals: str) -> bool:
        return any(signal in text for signal in signals)

    if has("form w-2", "wage and tax statement"):
        if "form w-2" in text:
            matched_signals.append("form w-2")
        if "wage and tax statement" in text:
            matched_signals.append("wage and tax statement")
        return ClassificationResult(doc_type="w2", confidence=0.98, strategy="rules", matched_signals=matched_signals)

    if has("1099-nec", "nonemployee compensation"):
        if "1099-nec" in text:
            matched_signals.append("1099-nec")
        if "nonemployee compensation" in text:
            matched_signals.append("nonemployee compensation")
        return ClassificationResult(doc_type="1099-nec", confidence=0.98, strategy="rules", matched_signals=matched_signals)

    if has("form 1040", "u.s. individual income tax return"):
        if "form 1040" in text:
            matched_signals.append("form 1040")
        if "u.s. individual income tax return" in text:
            matched_signals.append("u.s. individual income tax return")
        return ClassificationResult(
            doc_type="tax_return_package",
            confidence=0.97,
            strategy="rules",
            matched_signals=matched_signals,
        )

    financial_signals = [signal for signal in ("balance sheet", "account number", "account name", "accounting basis", "liabilities & capital") if signal in text]
    if "balance sheet" in financial_signals and len(financial_signals) >= 2:
        return ClassificationResult(doc_type="financial_statement", confidence=0.93, strategy="rules", matched_signals=financial_signals)

    receipt_signals = [signal for signal in ("subtotal", "tax", "total", "receipt", "visa", "mastercard", "amex", "change") if signal in text]
    if "total" in receipt_signals and len(receipt_signals) >= 2:
        return ClassificationResult(doc_type="receipt", confidence=0.9, strategy="rules", matched_signals=receipt_signals)
    if any(token in lower_name for token in ("receipt", "invoice")) and "total" in text:
        return ClassificationResult(doc_type="receipt", confidence=0.75, strategy="rules", matched_signals=["filename", "total"])
    if mime_type.startswith("image/") and "total" in text:
        return ClassificationResult(doc_type="receipt", confidence=0.72, strategy="rules", matched_signals=["image_mime", "total"])

    return ClassificationResult(doc_type="unknown", confidence=0.0, strategy="rules", matched_signals=[])
