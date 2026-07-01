# ==========================================================
# Configuration loader
# ----------------------------------------------------------
# Loads settings from config.yaml, applies optional overrides
# from a .env file / environment variables, and exposes them as
# a nested, attribute-accessible object (cfg.model.path, ...).
#
# The loader is defensive: a missing config.yaml, a missing key,
# or a malformed file all fall back to the built-in DEFAULTS so
# the app can always start.
# ==========================================================

import os
import copy
from types import SimpleNamespace

try:
    import yaml
except ImportError:  # PyYAML is optional at import time; we degrade gracefully.
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ----------------------------------------------------------
# Built-in defaults — mirror config.yaml so the app works even
# if the YAML file is absent.
# ----------------------------------------------------------
DEFAULTS = {
    "app": {"title": "Lab Report OCR • AI Vision", "page_icon": "🔬"},
    "model": {"path": "models/best.pt", "default_confidence": 0.25},
    "ocr": {
        "default_engine": "Auto (Tesseract → EasyOCR)",
        "tesseract_psm": 6,
        "easyocr_languages": ["en"],
        "easyocr_gpu": False,
    },
    "preprocess": {
        "enabled": False,
        "grayscale": True,
        "denoise": True,
        "adaptive_threshold": False,
        "sharpen": False,
        "resize_min_dim": 1000,
    },
    "paths": {"temp_dir": "temp", "outputs_dir": "outputs", "logs_dir": "outputs/logs"},
    "cleanup": {"temp_max_age_hours": 24},
    "logging": {
        "level": "INFO",
        "file": "outputs/logs/app.log",
        "max_bytes": 1048576,
        "backup_count": 5,
    },
    "pdf": {"render_dpi": 200},
    "batch": {"max_files": 25},
}

# Whitelisted environment overrides → (section, key, caster).
_ENV_OVERRIDES = {
    "LABOCR_MODEL_PATH": ("model", "path", str),
    "LABOCR_OCR_ENGINE": ("ocr", "default_engine", str),
    "LABOCR_LOG_LEVEL": ("logging", "level", str),
    "LABOCR_TEMP_MAX_AGE_HOURS": ("cleanup", "temp_max_age_hours", float),
}


def _deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data):
    """Apply whitelisted environment-variable overrides in place."""
    for env_key, (section, key, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            data.setdefault(section, {})[key] = caster(raw)
        except (ValueError, TypeError):
            # Ignore un-castable overrides rather than crash.
            continue
    return data


def _to_namespace(data):
    """Convert a (possibly nested) dict into SimpleNamespace tree."""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in data.items()})
    return data


def load_config(path="config.yaml"):
    """Load configuration and return a nested SimpleNamespace.

    Resolution order (lowest → highest precedence):
      1. Built-in DEFAULTS
      2. config.yaml (if present and parseable)
      3. .env / environment variables (whitelisted keys only)
    """
    file_data = {}
    if yaml is not None and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                file_data = yaml.safe_load(fh) or {}
        except Exception:
            file_data = {}

    merged = _deep_merge(DEFAULTS, file_data)

    # Load .env (if python-dotenv available) before reading env vars.
    if load_dotenv is not None:
        load_dotenv()
    merged = _apply_env_overrides(merged)

    return _to_namespace(merged)


def ensure_dirs(cfg):
    """Create the temp / outputs / logs directories declared in config."""
    for d in (cfg.paths.temp_dir, cfg.paths.outputs_dir, cfg.paths.logs_dir):
        os.makedirs(d, exist_ok=True)
