# ==========================================================
# OCR engine layer
# ----------------------------------------------------------
# Text cleaning + multi-engine extraction (Tesseract / EasyOCR
# / Auto fallback). This module is Streamlit-free so it can be
# unit-tested in isolation. The EasyOCR reader is *injected* by
# the caller (app.py caches it via st.cache_resource) to keep
# this module free of UI dependencies.
# ==========================================================

import re

import cv2
import numpy as np
from PIL import Image

# Optional OCR backends — imported defensively so the app still
# runs (with reduced capability) when one is not installed.
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr  # noqa: F401  (only needed to know availability)
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False


# Engine label constants (kept identical to the original UI strings).
ENGINE_TESSERACT = "Tesseract"
ENGINE_EASYOCR = "EasyOCR"
ENGINE_AUTO = "Auto (Tesseract → EasyOCR)"


# ----------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------

# Conservative substitutions for common OCR / unicode artifacts.
_OCR_REPLACEMENTS = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "‘": "'",   # curly quotes
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
}


def clean_ocr_text(text):
    """Normalise OCR output: strip artifacts, collapse whitespace/newlines."""
    if not text:
        return ""
    text = text.replace(" ", " ")  # non-breaking space → normal space
    for bad, good in _OCR_REPLACEMENTS.items():
        text = text.replace(bad, good)
    # Drop non-printable / control characters (keep basic ASCII range).
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Collapse any run of whitespace (spaces, tabs, newlines) to one space.
    text = re.sub(r"\s+", " ", text)
    # Trim stray noise punctuation often left at crop edges.
    text = text.strip(" \t\r\n|_~^`")
    return text.strip()


# ----------------------------------------------------------
# Engine availability + reader factory
# ----------------------------------------------------------

def available_engines():
    """Return the list of engine labels usable given installed backends."""
    engines = []
    if PYTESSERACT_AVAILABLE:
        engines.append(ENGINE_TESSERACT)
    if EASYOCR_AVAILABLE:
        engines.append(ENGINE_EASYOCR)
    if PYTESSERACT_AVAILABLE and EASYOCR_AVAILABLE:
        engines.append(ENGINE_AUTO)
    return engines


def make_easyocr_reader(languages=None, gpu=False):
    """Construct an EasyOCR reader. Caller is responsible for caching."""
    if not EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR is not installed.")
    return easyocr.Reader(languages or ["en"], gpu=gpu)


# ----------------------------------------------------------
# Engine runners
# ----------------------------------------------------------

def _ocr_tesseract(crop_rgb, psm=6):
    """Run Tesseract on an RGB crop with grayscale + Otsu preprocessing."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_crop = Image.fromarray(thresh)
    return pytesseract.image_to_string(pil_crop, config=f"--psm {psm}")


def _ocr_easyocr(crop_rgb, reader):
    """Run EasyOCR on an RGB crop; join detected fragments into one string."""
    if reader is None:
        raise RuntimeError("EasyOCR reader was not provided.")
    fragments = reader.readtext(crop_rgb, detail=0, paragraph=True)
    return " ".join(fragments)


def run_ocr(crop_rgb, engine, easyocr_reader=None, tesseract_psm=6):
    """Extract text from a crop using the requested engine.

    Returns ``(clean_text, success, used_engine)``. In Auto mode,
    Tesseract is tried first and EasyOCR is used as a fallback
    whenever Tesseract errors out or returns no text.

    ``success`` is True only when non-empty text was extracted.
    """
    if crop_rgb is None or getattr(crop_rgb, "size", 0) == 0:
        return "", False, "—"

    has_tess = PYTESSERACT_AVAILABLE
    has_easy = EASYOCR_AVAILABLE and easyocr_reader is not None
    want_tess_first = engine in (ENGINE_TESSERACT, ENGINE_AUTO)
    allow_easy_fallback = engine in (ENGINE_EASYOCR, ENGINE_AUTO)

    # 1. Primary attempt — Tesseract (unless EasyOCR was explicitly chosen).
    if want_tess_first and has_tess:
        try:
            text = clean_ocr_text(_ocr_tesseract(crop_rgb, psm=tesseract_psm))
            if text:
                return text, True, ENGINE_TESSERACT
        except Exception as exc:
            if not (allow_easy_fallback and has_easy):
                return f"[OCR Error: {exc}]", False, ENGINE_TESSERACT

    # 2. EasyOCR — either the selected engine or an automatic fallback.
    if allow_easy_fallback and has_easy:
        try:
            text = clean_ocr_text(_ocr_easyocr(crop_rgb, easyocr_reader))
            return text, bool(text), ENGINE_EASYOCR
        except Exception as exc:
            return f"[OCR Error: {exc}]", False, ENGINE_EASYOCR

    # 3. Nothing produced text (engine ran empty, or backend unavailable).
    if want_tess_first and has_tess:
        return "", False, ENGINE_TESSERACT
    return "", False, "—"
