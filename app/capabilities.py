from __future__ import annotations

import os
import subprocess
from shutil import which


def command_available(name: str) -> bool:
    return which(name) is not None


def tesseract_available() -> bool:
    return command_available("tesseract")


def ocrmypdf_available() -> bool:
    return command_available("ocrmypdf")


def ghostscript_available() -> bool:
    return command_available("gs")


def _paddleocr_importable() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> dict[str, object]:
    """Probe CUDA availability through multiple providers.

    Returns a dict with: available (bool), provider (str|None),
    device_count (int), driver (str|None).
    """
    # Allow env override for testing
    force = os.getenv("LOCI_EXTRACT_FORCE_CUDA", "").strip().lower()
    if force in ("1", "true", "yes"):
        return {"available": True, "provider": "env_override", "device_count": 1, "driver": None}
    if force in ("0", "false", "no"):
        return {"available": False, "provider": "env_override", "device_count": 0, "driver": None}

    # Try torch
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "provider": "torch",
                "device_count": torch.cuda.device_count(),
                "driver": None,
            }
    except (ImportError, Exception):
        pass

    # Try paddle
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return {
                "available": True,
                "provider": "paddle",
                "device_count": paddle.device.cuda.device_count(),
                "driver": None,
            }
    except (ImportError, Exception):
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
            return {
                "available": True,
                "provider": "nvidia-smi",
                "device_count": len(gpu_lines),
                "driver": None,
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return {"available": False, "provider": None, "device_count": 0, "driver": None}


def _detect_gpu_providers() -> dict[str, bool]:
    providers: dict[str, bool] = {}
    try:
        import torch  # noqa: F401
        providers["torch"] = True
    except ImportError:
        providers["torch"] = False
    try:
        import paddle  # noqa: F401
        providers["paddle"] = True
    except ImportError:
        providers["paddle"] = False
    providers["paddleocr"] = _paddleocr_importable()
    try:
        import transformers  # noqa: F401
        providers["transformers"] = True
    except ImportError:
        providers["transformers"] = False
    return providers


def _detect_ocr_backends() -> dict[str, object]:
    tess = tesseract_available()
    paddle = _paddleocr_importable()
    cuda = cuda_available()
    paddle_usable = paddle and (
        cuda["available"]
        or os.getenv("LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU", "").strip() in ("1", "true", "yes")
    )
    if paddle_usable:
        default_backend = "paddleocr"
    elif tess:
        default_backend = "tesseract"
    else:
        default_backend = None
    return {
        "tesseract": tess,
        "paddleocr": paddle_usable,
        "default": default_backend,
    }


def _donut_classifier_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return cuda_available()["available"]
    except ImportError:
        return False


def _vlm_endpoint_available() -> bool:
    try:
        from app.llm.config import get_vlm_client
        return get_vlm_client() is not None
    except Exception:
        return False


def _detect_llm_endpoints() -> dict[str, object]:
    try:
        from app.llm.config import list_endpoints
        return list_endpoints()
    except Exception:
        return {}


def _detect_profiles() -> dict[str, object]:
    from app.profiles import list_profiles

    return {
        "available": list_profiles(),
        "classifier_models": {
            "donut-irs": _donut_classifier_available(),
            "layout": _paddleocr_importable() and cuda_available()["available"],
            "vlm": _vlm_endpoint_available(),
            "rules": True,
        },
    }


def detect_compare_pipelines() -> dict[str, object]:
    """Return which /extract/compare pipeline names run on this host.

    Used by both the capabilities endpoint and the compare endpoint to skip
    pipelines that lack their backend (no tesseract, no GPU, no VLM, etc.).
    """
    tess = tesseract_available()
    paddle_ok = _paddleocr_importable() and (
        cuda_available()["available"]
        or os.getenv("LOCI_EXTRACT_ALLOW_PADDLEOCR_CPU", "").strip() in ("1", "true", "yes")
    )
    vlm = _vlm_endpoint_available()
    available: list[str] = ["parser"]
    if tess:
        available += ["ocr_tesseract", "force_image_tesseract"]
    if paddle_ok:
        available += ["ocr_paddle", "force_image_paddle"]
    if vlm:
        available += ["vlm_pure", "vlm_hybrid"]
    return {
        "available_pipelines": available,
        "backends": {
            "tesseract": tess,
            "paddleocr": paddle_ok,
            "vlm": vlm,
        },
    }


def detect_capabilities() -> dict[str, object]:
    return {
        "ocr": {
            "tesseract": tesseract_available(),
            "ocrmypdf": ocrmypdf_available(),
            "ghostscript": ghostscript_available(),
        },
        "pdf": {
            "pdftoppm": command_available("pdftoppm"),
            "pdfinfo": command_available("pdfinfo"),
            "pymupdf": True,
        },
        "documents": {
            "docx": True,
            "xlsx": True,
            "images": True,
        },
        "gpu": {
            "cuda": cuda_available(),
            "providers": _detect_gpu_providers(),
        },
        "ocr_backends": _detect_ocr_backends(),
        "profiles": _detect_profiles(),
        "llm_endpoints": _detect_llm_endpoints(),
        "compare": detect_compare_pipelines(),
    }
