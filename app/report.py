# ==========================================================
# Structured lab-report mapping
# ----------------------------------------------------------
# Maps raw YOLO class names onto human-friendly report fields and
# builds the tidy "one row per detection" table that drives the
# editable grid and every export. Pure / Streamlit-free.
# ==========================================================

from collections import OrderedDict

# Raw YOLO class  ->  structured report field.
CLASS_TO_FIELD = {
    "Name": "Patient Name",
    "Test_Name": "Test Name",
    "Test_Asked": "Test Category",
    "Technology": "Technology",
    "Value": "Value",
    "Unit": "Unit",
    "Reference_Range": "Reference Range",
    "Clinical_Conditions": "Clinical Significance",
    "Ref_By": "Referred By",
}

# Preferred display / export ordering of fields.
FIELD_ORDER = [
    "Patient Name",
    "Referred By",
    "Test Category",
    "Technology",
    "Test Name",
    "Value",
    "Unit",
    "Reference Range",
    "Clinical Significance",
]

# Fields that should logically have a single value per report.
SINGLETON_FIELDS = {"Patient Name", "Referred By", "Test Category", "Technology"}


def map_field(class_name):
    """Return the friendly field name for a raw class (fallback: the class)."""
    return CLASS_TO_FIELD.get(class_name, class_name)


def build_report_rows(detections, source="", page=1):
    """Build tidy report rows from a list of detection dicts.

    Each detection dict is expected to have: class, confidence,
    bbox, ocr_text (already cleaned), and optionally ocr_engine.
    Detections should already be in reading order.

    Returns a list of ordered dicts — one row per detection — with
    stable column names used across the UI and all exports.
    """
    rows = []
    for i, det in enumerate(detections, start=1):
        cls = det.get("class", "")
        bbox = det.get("bbox", [0, 0, 0, 0])
        rows.append(OrderedDict([
            ("uid", f"{source}|{page}|{i}"),
            ("Source", source),
            ("Page", page),
            ("Order", i),
            ("Field", map_field(cls)),
            ("Detected Class", cls),
            ("Text", det.get("ocr_text", "")),
            ("Confidence", det.get("confidence", 0.0)),
            ("OCR Engine", det.get("ocr_engine", "")),
            ("Bbox", det.get("bbox", [])),
            ("x1", bbox[0] if len(bbox) > 0 else 0),
            ("y1", bbox[1] if len(bbox) > 1 else 0),
            ("x2", bbox[2] if len(bbox) > 2 else 0),
            ("y2", bbox[3] if len(bbox) > 3 else 0),
        ]))
    return rows


def summarise_fields(rows):
    """Derive a best-effort key-value summary of singleton fields.

    For fields expected to be unique (Patient Name, Referred By,
    Test Category, Technology) pick the highest-confidence non-empty
    text. Returns an ordered dict {field: text}. Repeating
    measurement fields are intentionally excluded — they belong in
    the tidy table, not the summary.
    """
    best = OrderedDict()
    best_conf = {}
    for row in rows:
        field = row.get("Field", "")
        if field not in SINGLETON_FIELDS:
            continue
        text = (row.get("Text", "") or "").strip()
        if not text or text.startswith("["):
            continue
        conf = float(row.get("Confidence", 0.0) or 0.0)
        if field not in best or conf > best_conf.get(field, -1):
            best[field] = text
            best_conf[field] = conf

    # Emit in canonical order.
    summary = OrderedDict()
    for field in FIELD_ORDER:
        if field in best:
            summary[field] = best[field]
    return summary
