"""Profile loader — reads YAML files from app/profiles/ with lazy caching."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import yaml

from app.profiles.schema import ExtractionProfile

logger = logging.getLogger("loci.profiles")

_PROFILES_DIR = Path(__file__).parent
_cache: dict[str, ExtractionProfile] | None = None
_lock = threading.Lock()


def _load_all() -> dict[str, ExtractionProfile]:
    """Read and validate all .yaml files in the profiles directory."""
    profiles: dict[str, ExtractionProfile] = {}
    for yaml_path in sorted(_PROFILES_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(yaml_path.read_text())
            if not isinstance(raw, dict) or "name" not in raw:
                logger.warning("Skipping invalid profile file: %s", yaml_path.name)
                continue
            profile = ExtractionProfile(**raw)
            profiles[profile.name] = profile
        except Exception as exc:
            logger.warning("Failed to load profile %s: %s", yaml_path.name, exc)
    return profiles


def _ensure_loaded() -> dict[str, ExtractionProfile]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load_all()
    return _cache


def get_profile(name: str) -> ExtractionProfile | None:
    """Return a cached profile by name, or None if not found."""
    return _ensure_loaded().get(name)


def list_profiles() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(_ensure_loaded().keys())


def reset_cache() -> None:
    """Reset the profile cache (for testing)."""
    global _cache
    _cache = None
