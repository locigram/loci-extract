from __future__ import annotations

from typing import Any

import pytesseract
from PIL import Image, ImageFilter, ImageOps


def score_ocr_text(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    alnum_count = sum(1 for char in cleaned if char.isalnum())
    line_count = len([line for line in cleaned.splitlines() if line.strip()])
    words = [word for word in cleaned.replace("\n", " ").split() if word.strip()]
    unique_words = {word.lower() for word in words}
    long_word_count = sum(1 for word in words if len(word) >= 4)

    score = 0.0
    score += alnum_count * 1.2
    score += line_count * 6.0
    score += len(words) * 2.0
    score += len(unique_words) * 1.5
    score += long_word_count * 2.5

    if len(cleaned) < 12:
        score -= 10.0
    if len(unique_words) <= 1 and len(words) > 3:
        score -= 8.0

    return round(max(score, 0.0), 2)


def _base_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def build_ocr_variants(image: Image.Image) -> list[dict[str, Any]]:
    base = _base_image(image)
    gray = ImageOps.autocontrast(base.convert("L"))

    soft_upscale = gray.resize((gray.width * 2, gray.height * 2))
    threshold_180 = gray.point(lambda px: 255 if px > 180 else 0)
    threshold_180_upscale = threshold_180.resize((threshold_180.width * 2, threshold_180.height * 2))
    threshold_160 = gray.point(lambda px: 255 if px > 160 else 0)
    threshold_160_sharpen = threshold_160.filter(ImageFilter.SHARPEN)
    threshold_160_upscale = threshold_160_sharpen.resize((threshold_160_sharpen.width * 2, threshold_160_sharpen.height * 2))
    sharpen_upscale = gray.filter(ImageFilter.SHARPEN).resize((gray.width * 2, gray.height * 2))

    variants = [
        {
            "name": "soft_upscale",
            "image": soft_upscale,
            "steps": ["exif_transpose", "grayscale", "autocontrast", "upscale_2x"],
        },
        {
            "name": "threshold_180_upscale",
            "image": threshold_180_upscale,
            "steps": ["exif_transpose", "grayscale", "autocontrast", "threshold_180", "upscale_2x"],
        },
        {
            "name": "threshold_160_sharpen_upscale",
            "image": threshold_160_upscale,
            "steps": [
                "exif_transpose",
                "grayscale",
                "autocontrast",
                "threshold_160",
                "sharpen",
                "upscale_2x",
            ],
        },
        {
            "name": "sharpen_upscale",
            "image": sharpen_upscale,
            "steps": ["exif_transpose", "grayscale", "autocontrast", "sharpen", "upscale_2x"],
        },
    ]

    for variant in variants:
        variant_image = variant["image"]
        setattr(variant_image, "_ocr_pass_name", variant["name"])
    return variants


def extract_best_ocr_result(image: Image.Image) -> dict[str, Any]:
    variants = build_ocr_variants(image)
    pass_results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for variant in variants:
        text = pytesseract.image_to_string(variant["image"]).strip()
        score = score_ocr_text(text)
        pass_result = {
            "name": variant["name"],
            "steps": variant["steps"],
            "text": text,
            "score": score,
            "text_length": len(text),
            "processed_mode": variant["image"].mode,
        }
        pass_results.append(pass_result)
        if best_result is None or pass_result["score"] > best_result["score"]:
            best_result = pass_result

    best_result = best_result or {
        "name": "none",
        "steps": [],
        "text": "",
        "score": 0.0,
        "text_length": 0,
        "processed_mode": image.mode,
    }

    return {
        "text": best_result["text"],
        "score": best_result["score"],
        "selected_pass": best_result["name"],
        "processed_mode": best_result["processed_mode"],
        "preprocessing": best_result["steps"],
        "ocr_passes": [
            {
                "name": item["name"],
                "score": item["score"],
                "text_length": item["text_length"],
                "processed_mode": item["processed_mode"],
                "steps": item["steps"],
            }
            for item in pass_results
        ],
    }
