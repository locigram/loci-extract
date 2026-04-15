"""VLM path — send rendered page images to a multimodal OpenAI-compatible endpoint.

Used when ``--vision`` is passed or when the default pipeline hits an image
page with ``vision=True`` in options. Renders the PDF pages to PNG, encodes
each as a base64 data URL, and sends one chat call per page with the same
SYSTEM_PROMPT used in the text path. The returned text is concatenated and
parsed by ``llm.parse_extraction``.

For multimodal models (Qwen3-VL, llava, minicpm-v) hosted on llama.cpp /
Ollama / vLLM, the ``image_url`` content part with ``data:image/png;base64,...``
is the standard way to deliver images through the OpenAI-compat API.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger("loci_extract.vision")


def _render_and_encode(pdf_path: Path, page: int, dpi: int, max_dim: int, tmp_dir: Path) -> str:
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page,
        last_page=page,
        output_folder=str(tmp_dir),
    )
    if not images:
        raise RuntimeError(f"pdf2image returned no images for page {page}")
    img = images[0].convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vision_extract_pages(
    client,
    pdf_path: str | Path,
    pages: list[int],
    *,
    vision_model: str,
    system_prompt: str,
    dpi: int = 300,
    max_dim: int = 2048,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[int, str]:
    """Ask the VLM to read each page image. Returns ``{page: raw_text}``.

    The prompt tells the model to output readable text (not JSON yet) — JSON
    structuring happens in a downstream text-LLM call via ``llm.parse_extraction``.
    This keeps the vision step focused on OCR-style transcription and makes the
    structured pass cheaper/more reliable.
    """
    if not pages:
        return {}

    out: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="loci-extract-vision-") as tmp:
        tmp_dir = Path(tmp)
        for page in pages:
            try:
                b64 = _render_and_encode(Path(pdf_path), page, dpi, max_dim, tmp_dir)
            except Exception as exc:
                logger.warning("Vision render failed for page %d: %s", page, exc)
                out[page] = ""
                continue
            try:
                response = client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Transcribe ALL visible text on this tax document page verbatim. "
                                        "Preserve line breaks, field labels, dollar amounts, and any "
                                        "non-standard codes. Do not summarize. Return plain text only."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                            ],
                        },
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                out[page] = response.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("Vision call failed for page %d: %s", page, exc)
                out[page] = ""
    return out


__all__ = ["vision_extract_pages"]
