"""LLM/VLM endpoint configuration.

Supports multiple named endpoints configured via environment variables or profile config.
The default endpoint uses LOCI_EXTRACT_LLM_* env vars. Additional endpoints can be
configured via LOCI_EXTRACT_VLM_* env vars or passed at runtime from profile settings.

Env vars (default LLM endpoint):
  LOCI_EXTRACT_LLM_ENABLED   — "true" to enable (default "false")
  LOCI_EXTRACT_LLM_BASE_URL  — OpenAI-compatible endpoint base URL
  LOCI_EXTRACT_LLM_MODEL     — model name
  LOCI_EXTRACT_LLM_TIMEOUT   — request timeout in seconds (default 8)
  LOCI_EXTRACT_LLM_API_KEY   — optional API key

Env vars (VLM endpoint):
  LOCI_EXTRACT_VLM_ENABLED   — "true" to enable (default "false")
  LOCI_EXTRACT_VLM_BASE_URL  — OpenAI-compatible endpoint (e.g. Ollama at http://localhost:11434)
  LOCI_EXTRACT_VLM_MODEL     — model name (e.g. "qwen3-vl:8b")
  LOCI_EXTRACT_VLM_TIMEOUT   — request timeout in seconds (default 30)
  LOCI_EXTRACT_VLM_API_KEY   — optional API key
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from app.llm.client import LlmClient

logger = logging.getLogger("loci.llm.config")

_registry: dict[str, LlmClient] = {}
_lock = threading.Lock()
_defaults_loaded = False


def _load_env_endpoint(prefix: str, default_timeout: float = 8.0) -> LlmClient | None:
    """Load an endpoint from env vars with the given prefix."""
    enabled = os.getenv(f"{prefix}_ENABLED", "false").strip().lower()
    if enabled not in ("1", "true", "yes"):
        return None
    base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
    if not base_url:
        return None
    model = os.getenv(f"{prefix}_MODEL", "").strip()
    if not model:
        return None
    timeout_raw = os.getenv(f"{prefix}_TIMEOUT", str(default_timeout)).strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = default_timeout
    api_key = os.getenv(f"{prefix}_API_KEY", "").strip() or None
    return LlmClient(base_url=base_url, model=model, timeout=timeout, api_key=api_key)


def _ensure_defaults_loaded() -> None:
    """Load default LLM and VLM endpoints from env vars (once)."""
    global _defaults_loaded
    if _defaults_loaded:
        return
    with _lock:
        if _defaults_loaded:
            return
        # Default LLM endpoint (text-only enrichment)
        llm = _load_env_endpoint("LOCI_EXTRACT_LLM", default_timeout=8.0)
        if llm:
            _registry["default"] = llm
        # Default VLM endpoint (vision extraction)
        vlm = _load_env_endpoint("LOCI_EXTRACT_VLM", default_timeout=30.0)
        if vlm:
            _registry["vlm"] = vlm
        _defaults_loaded = True


def register_endpoint(name: str, client: LlmClient) -> None:
    """Register a named endpoint at runtime."""
    with _lock:
        _registry[name] = client


def get_llm_client(name: str = "default") -> LlmClient | None:
    """Get a named LLM client. Returns None if not configured."""
    _ensure_defaults_loaded()
    return _registry.get(name)


def get_vlm_client() -> LlmClient | None:
    """Shorthand for get_llm_client('vlm')."""
    return get_llm_client("vlm")


def client_from_config(config: dict[str, Any]) -> LlmClient | None:
    """Create an LlmClient from a profile config dict.

    Config shape: {"base_url": "...", "model": "...", "timeout": 30, "api_key": "..."}
    Returns None if base_url or model is missing.
    """
    base_url = str(config.get("base_url", "")).strip()
    model = str(config.get("model", "")).strip()
    if not base_url or not model:
        return None
    return LlmClient(
        base_url=base_url,
        model=model,
        timeout=float(config.get("timeout", 30)),
        api_key=str(config.get("api_key", "")).strip() or None,
    )


def list_endpoints() -> dict[str, dict[str, str]]:
    """List configured endpoints (for /capabilities)."""
    _ensure_defaults_loaded()
    return {
        name: {"base_url": c.base_url, "model": c.model}
        for name, c in _registry.items()
    }


def reset_llm_client() -> None:
    """Reset all state (for testing)."""
    global _defaults_loaded
    with _lock:
        _registry.clear()
        _defaults_loaded = False
