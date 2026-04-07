from __future__ import annotations

from statistics import median
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


def _build_variant(name: str, image: Image.Image, steps: list[str], *, rotation_degrees: int = 0) -> dict[str, Any]:
    setattr(image, "_ocr_pass_name", name)
    setattr(image, "_ocr_rotation", rotation_degrees)
    return {
        "name": name,
        "image": image,
        "steps": steps,
        "rotation_degrees": rotation_degrees,
    }



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
        _build_variant(
            "soft_upscale",
            soft_upscale,
            ["exif_transpose", "grayscale", "autocontrast", "upscale_2x"],
        ),
        _build_variant(
            "threshold_180_upscale",
            threshold_180_upscale,
            ["exif_transpose", "grayscale", "autocontrast", "threshold_180", "upscale_2x"],
        ),
        _build_variant(
            "threshold_160_sharpen_upscale",
            threshold_160_upscale,
            [
                "exif_transpose",
                "grayscale",
                "autocontrast",
                "threshold_160",
                "sharpen",
                "upscale_2x",
            ],
        ),
        _build_variant(
            "sharpen_upscale",
            sharpen_upscale,
            ["exif_transpose", "grayscale", "autocontrast", "sharpen", "upscale_2x"],
        ),
    ]

    for rotation_degrees in (90, 270):
        rotated = soft_upscale.rotate(rotation_degrees, expand=True)
        variants.append(
            _build_variant(
                f"soft_upscale_rot{rotation_degrees}",
                rotated,
                ["exif_transpose", "grayscale", "autocontrast", "upscale_2x", f"rotate_{rotation_degrees}"],
                rotation_degrees=rotation_degrees,
            )
        )
    return variants



def _cluster_positions(values: list[float], tolerance: float) -> list[float]:
    anchors: list[float] = []
    for value in sorted(values):
        for idx, anchor in enumerate(anchors):
            if abs(anchor - value) <= tolerance:
                anchors[idx] = round((anchor + value) / 2, 2)
                break
        else:
            anchors.append(round(value, 2))
    return anchors



def _extract_table_candidates(image: Image.Image) -> list[dict[str, Any]]:
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    texts = data.get("text") or []
    if not texts:
        return []

    words: list[dict[str, Any]] = []
    for idx, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        left = int(data.get("left", [0])[idx] or 0)
        top = int(data.get("top", [0])[idx] or 0)
        width = int(data.get("width", [0])[idx] or 0)
        height = int(data.get("height", [0])[idx] or 0)
        conf_raw = data.get("conf", ["-1"])[idx]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        words.append(
            {
                "text": text,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "center_y": top + (height / 2 if height else 0),
                "conf": conf,
            }
        )

    if len(words) < 4:
        return []

    heights = [word["height"] for word in words if word["height"] > 0]
    row_tolerance = max(12.0, (median(heights) * 0.8) if heights else 12.0)
    rows: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["left"])):
        for row in rows:
            if abs(row["center_y"] - word["center_y"]) <= row_tolerance:
                row["words"].append(word)
                row["center_y"] = (row["center_y"] + word["center_y"]) / 2
                break
        else:
            rows.append({"center_y": word["center_y"], "words": [word]})

    dense_rows = []
    for row in rows:
        ordered_words = sorted(row["words"], key=lambda item: item["left"])
        if len(ordered_words) < 2:
            continue
        dense_rows.append(ordered_words)

    if len(dense_rows) < 2:
        return []

    widths = [word["width"] for row in dense_rows for word in row if word["width"] > 0]
    column_tolerance = max(24.0, (median(widths) * 1.5) if widths else 24.0)
    anchors = _cluster_positions([word["left"] for row in dense_rows for word in row], column_tolerance)
    if len(anchors) < 2:
        return []

    rendered_rows: list[str] = []
    populated_rows = 0
    for row in dense_rows:
        cells = [""] * len(anchors)
        for word in row:
            nearest_index = min(range(len(anchors)), key=lambda idx: abs(anchors[idx] - word["left"]))
            cells[nearest_index] = (cells[nearest_index] + " " + word["text"]).strip()
        non_empty = [cell for cell in cells if cell]
        if len(non_empty) < 2:
            continue
        populated_rows += 1
        rendered_rows.append(" | ".join(cell for cell in cells if cell))

    if populated_rows < 2:
        return []

    return [
        {
            "text": "\n".join(rendered_rows).strip(),
            "row_count": populated_rows,
            "column_count": len(anchors),
            "detection_method": "ocr_word_grid",
        }
    ]



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
            "rotation_degrees": variant["rotation_degrees"],
            "image": variant["image"],
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
        "rotation_degrees": 0,
        "image": image,
        "text": "",
        "score": 0.0,
        "text_length": 0,
        "processed_mode": image.mode,
    }
    table_candidates = _extract_table_candidates(best_result["image"])

    return {
        "text": best_result["text"],
        "score": best_result["score"],
        "selected_pass": best_result["name"],
        "selected_rotation": best_result["rotation_degrees"],
        "processed_mode": best_result["processed_mode"],
        "preprocessing": best_result["steps"],
        "table_candidates": table_candidates,
        "ocr_passes": [
            {
                "name": item["name"],
                "score": item["score"],
                "text_length": item["text_length"],
                "processed_mode": item["processed_mode"],
                "rotation_degrees": item["rotation_degrees"],
                "steps": item["steps"],
            }
            for item in pass_results
        ],
    }
