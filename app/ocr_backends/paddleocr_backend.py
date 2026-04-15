"""PaddleOCR GPU backend — import-guarded, lazy-initialized.

Returns unavailable when paddleocr is not installed or CUDA is not present
(unless LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU=1 is set).
"""

from __future__ import annotations

import os
import threading

from PIL import Image

from app.ocr_backends import OcrBackendResult


def _paddleocr_importable() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


def _cuda_runtime_available() -> bool:
    """Check if CUDA is usable for PaddleOCR."""
    try:
        import paddle

        return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False


class PaddleOCRBackend:
    _instance = None
    _lock = threading.Lock()

    @property
    def name(self) -> str:
        return "paddleocr"

    def is_available(self) -> bool:
        if not _paddleocr_importable():
            return False
        if os.getenv("LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU", "").strip() in ("1", "true", "yes"):
            return True
        return _cuda_runtime_available()

    def _get_model(self):
        """Lazy-init the PaddleOCR model handle (thread-safe singleton)."""
        if PaddleOCRBackend._instance is None:
            with PaddleOCRBackend._lock:
                if PaddleOCRBackend._instance is None:
                    from paddleocr import PaddleOCR

                    use_gpu = _cuda_runtime_available()
                    PaddleOCRBackend._instance = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        use_gpu=use_gpu,
                        show_log=False,
                    )
        return PaddleOCRBackend._instance

    def run(
        self,
        image: Image.Image,
        *,
        variant_name: str | None = None,
        rotation: int = 0,
    ) -> OcrBackendResult:
        import numpy as np

        from app.ocr_backends import WordBox

        model = self._get_model()
        img_array = np.array(image)
        result = model.ocr(img_array, cls=True)

        lines: list[str] = []
        words: list[WordBox] = []
        confidences: list[float] = []

        for page_result in result or []:
            for entry in page_result or []:
                box, (text, conf) = entry
                lines.append(text)
                confidences.append(conf)

                # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                left = int(min(xs))
                top = int(min(ys))
                width = int(max(xs) - left)
                height = int(max(ys) - top)
                words.append(WordBox(text=text, left=left, top=top, width=width, height=height, conf=conf))

        full_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return OcrBackendResult(
            text=full_text,
            confidence=avg_conf,
            words=words,
            raw={"paddle_result": result},
        )
