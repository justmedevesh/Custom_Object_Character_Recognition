# ==========================================================
# PDF helpers
# ----------------------------------------------------------
# Convert an uploaded PDF into a list of RGB page images using
# PyMuPDF (fitz). PyMuPDF ships as a self-contained wheel, so no
# system poppler install is required (unlike pdf2image).
# ==========================================================

import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False


def pdf_bytes_to_images(pdf_bytes, dpi=200, max_pages=None):
    """Render each PDF page to an RGB numpy array.

    Returns a list of ``(page_index, image_rgb)`` tuples with
    ``page_index`` starting at 1. Raises RuntimeError if PyMuPDF
    is not installed so the caller can surface a clear message.
    """
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError(
            "PyMuPDF (pymupdf) is not installed — cannot read PDF files."
        )

    # Zoom factor: 72 is the PDF's base DPI.
    zoom = float(dpi) / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for page_index in range(total):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            # pix.samples is a flat RGB buffer (alpha=False → 3 channels).
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # safety: drop alpha if present
                img = img[:, :, :3]
            pages.append((page_index + 1, np.ascontiguousarray(img)))
    return pages
