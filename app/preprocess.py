# ==========================================================
# Image preprocessing for improved OCR accuracy
# ----------------------------------------------------------
# A small, composable set of cv2/numpy operations. Each step is
# optional and toggled from the UI / config. All functions take
# and return an RGB ndarray so they can be chained freely and the
# downstream OCR/preview code stays unchanged.
# ==========================================================

import cv2
import numpy as np


def to_grayscale(img_rgb):
    """Convert to grayscale but return a 3-channel RGB image.

    Keeping 3 channels means every downstream consumer (OCR,
    st.image, PIL save) works without special-casing.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def denoise(img_rgb):
    """Edge-preserving denoise (good for scanner/JPEG noise)."""
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)


def adaptive_threshold(img_rgb):
    """Binarise using a local adaptive threshold; return RGB."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def sharpen(img_rgb):
    """Apply an unsharp-mask style sharpening kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img_rgb, -1, kernel)


def resize_min_dim(img_rgb, min_dim):
    """Upscale so the shortest side is at least ``min_dim`` pixels.

    Only ever enlarges (never shrinks) — small crops benefit most
    from upscaling before OCR. ``min_dim<=0`` disables resizing.
    """
    if not min_dim or min_dim <= 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return img_rgb
    shortest = min(h, w)
    if shortest >= min_dim:
        return img_rgb
    scale = min_dim / float(shortest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess_image(img_rgb, options):
    """Apply the enabled preprocessing steps in a sensible order.

    ``options`` is a mapping (or namespace) with boolean flags:
    grayscale, denoise, adaptive_threshold, sharpen, and an int
    ``resize_min_dim``. Order matters: resize → denoise → sharpen
    → grayscale/threshold.
    """
    if img_rgb is None or getattr(img_rgb, "size", 0) == 0:
        return img_rgb

    def opt(name, default=False):
        if isinstance(options, dict):
            return options.get(name, default)
        return getattr(options, name, default)

    out = img_rgb
    out = resize_min_dim(out, opt("resize_min_dim", 0))
    if opt("denoise"):
        out = denoise(out)
    if opt("sharpen"):
        out = sharpen(out)
    # Threshold subsumes grayscale; otherwise apply grayscale alone.
    if opt("adaptive_threshold"):
        out = adaptive_threshold(out)
    elif opt("grayscale"):
        out = to_grayscale(out)
    return out
