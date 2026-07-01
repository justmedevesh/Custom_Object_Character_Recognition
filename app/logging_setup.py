# ==========================================================
# Logging setup
# ----------------------------------------------------------
# Provides a single configured logger shared across the app.
# Writes to both the console and a rotating file (outputs/logs/
# app.log). Streamlit re-runs the script top-to-bottom on every
# interaction, so we guard against attaching duplicate handlers.
# ==========================================================

import os
import json
import logging
from logging.handlers import RotatingFileHandler

_LOGGER_NAME = "labocr"
_CONFIGURED = False


def get_logger(cfg=None):
    """Return the shared application logger, configuring it once.

    ``cfg`` is the object returned by config.load_config(); when
    omitted, sensible defaults are used.
    """
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)

    if _CONFIGURED:
        return logger

    # ── Resolve settings (with fallbacks) ──
    level_name = getattr(getattr(cfg, "logging", None), "level", "INFO") if cfg else "INFO"
    log_file = (
        getattr(getattr(cfg, "logging", None), "file", "outputs/logs/app.log")
        if cfg else "outputs/logs/app.log"
    )
    max_bytes = getattr(getattr(cfg, "logging", None), "max_bytes", 1048576) if cfg else 1048576
    backup_count = getattr(getattr(cfg, "logging", None), "backup_count", 5) if cfg else 5

    logger.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── Rotating file handler (best-effort) ──
    try:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception as exc:  # Never let logging setup crash the app.
        logger.warning("File logging unavailable: %s", exc)

    _CONFIGURED = True
    logger.info("Logger initialised (level=%s, file=%s)", level_name, log_file)
    return logger


def log_run_record(logs_dir, record):
    """Append one structured run record as a line to runs.jsonl.

    ``record`` is a JSON-serialisable dict (timestamp, filename,
    detections, OCR text, timings, engine, ...). Best-effort: any
    IO error is swallowed so it never interrupts the UI.
    """
    try:
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, "runs.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path
    except Exception:
        return None
