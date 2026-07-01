# ==========================================================
# Export builders
# ----------------------------------------------------------
# Turn the (possibly user-edited) report rows into downloadable
# bytes: CSV, JSON, Excel (.xlsx) and a combined ZIP. All inputs
# are plain lists of dicts so this module is Streamlit-free and
# unit-testable. Excel uses pandas + openpyxl.
# ==========================================================

import io
import csv
import json
import zipfile

# Columns surfaced in the flat exports (internal helper keys dropped).
EXPORT_COLUMNS = [
    "Source", "Page", "Order", "Field", "Detected Class",
    "Text", "Confidence", "OCR Engine", "x1", "y1", "x2", "y2",
]


def rows_to_records(rows):
    """Project full report rows down to the public export columns."""
    records = []
    for row in rows:
        records.append({col: row.get(col, "") for col in EXPORT_COLUMNS})
    return records


def records_to_csv_bytes(records):
    """Serialise records to UTF-8 CSV bytes."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    for rec in records:
        writer.writerow({col: rec.get(col, "") for col in EXPORT_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def records_to_json_bytes(records, summary=None, meta=None):
    """Serialise records (+ optional summary/meta) to pretty JSON bytes."""
    payload = {
        "meta": meta or {},
        "summary": summary or {},
        "results": records,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def records_to_excel_bytes(records, summary=None):
    """Serialise records to an .xlsx workbook (Results + Summary sheets).

    Requires pandas + openpyxl. Raises ImportError if unavailable so
    the caller can disable the Excel button gracefully.
    """
    import pandas as pd  # local import keeps module importable without pandas

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df = pd.DataFrame(records, columns=EXPORT_COLUMNS)
        df.to_excel(writer, sheet_name="OCR Results", index=False)

        if summary:
            summary_df = pd.DataFrame(
                [{"Field": k, "Value": v} for k, v in summary.items()]
            )
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer.getvalue()


def excel_available():
    """True when pandas + an Excel writer engine are importable."""
    try:
        import pandas  # noqa: F401
        import openpyxl  # noqa: F401
        return True
    except ImportError:
        return False


def build_zip(records, summary=None, meta=None, images=None, crops=None):
    """Build a combined ZIP archive in memory.

    ``images``: list of (filename, png_bytes) — e.g. annotated pages.
    ``crops``:  list of (filename, png_bytes) — individual crops.
    Always includes CSV + JSON, and Excel when the engine is present.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ocr_results.csv", records_to_csv_bytes(records))
        zf.writestr("ocr_results.json", records_to_json_bytes(records, summary, meta))
        if excel_available():
            try:
                zf.writestr("ocr_results.xlsx", records_to_excel_bytes(records, summary))
            except Exception:
                pass  # Excel is a nice-to-have inside the ZIP.
        for fname, data in (images or []):
            zf.writestr(fname, data)
        for fname, data in (crops or []):
            zf.writestr(f"crops/{fname}", data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
