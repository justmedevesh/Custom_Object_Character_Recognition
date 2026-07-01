# ==========================================================
# Processing pipeline — the keystone
# ----------------------------------------------------------
# process_one() runs the full compute path for a SINGLE page:
#   predict → reading-order sort → (optional preprocess) → crop
#   → OCR → annotate → build report rows + timings.
#
# Because both the single-image and batch flows call this exact
# function, "single image" is simply "a batch of one" and the two
# code paths can never drift apart. It returns a plain dict
# ("bundle") of results; all Streamlit rendering happens in app.py.
# ==========================================================

import time

import cv2
import numpy as np

import ocr as ocr_mod
import preprocess as preprocess_mod
from detect import reading_order_indices
from report import build_report_rows, summarise_fields


def _annotate(result, show_labels=True, show_conf=True):
    """Return the YOLO annotated image as an RGB ndarray."""
    plotted = result.plot(labels=show_labels, conf=show_conf)
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)


def process_one(
    image_rgb,
    model,
    *,
    source="image",
    page=1,
    confidence=0.25,
    ocr_engine=ocr_mod.ENGINE_AUTO,
    easyocr_reader=None,
    tesseract_psm=6,
    preprocess_options=None,
    show_labels=True,
    show_conf=True,
):
    """Process a single RGB image and return a result bundle.

    The bundle keys:
      source, page, image_rgb, annotated_rgb,
      detections (list of dicts in reading order, each with
        index/class/confidence/bbox/crop/ocr_text/ocr_success/ocr_engine),
      report_rows, summary,
      num_detections, classes (Counter-like dict),
      yolo_time, ocr_time, total_time,
      ocr_success_count, ocr_fail_count,
      error (None unless prediction failed)
    """
    total_start = time.perf_counter()
    bundle = {
        "source": source,
        "page": page,
        "image_rgb": image_rgb,
        "annotated_rgb": image_rgb,
        "detections": [],
        "report_rows": [],
        "summary": {},
        "num_detections": 0,
        "classes": {},
        "yolo_time": 0.0,
        "ocr_time": 0.0,
        "total_time": 0.0,
        "ocr_success_count": 0,
        "ocr_fail_count": 0,
        "error": None,
    }

    # ── 1. YOLO detection (BGR for ultralytics/cv2 consistency) ──
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        yolo_start = time.perf_counter()
        results = model.predict(source=image_bgr, conf=confidence, verbose=False)
        bundle["yolo_time"] = time.perf_counter() - yolo_start
    except Exception as exc:
        bundle["error"] = f"Detection failed: {exc}"
        bundle["total_time"] = time.perf_counter() - total_start
        return bundle

    result = results[0]
    boxes = result.boxes
    bundle["annotated_rgb"] = _annotate(result, show_labels, show_conf)

    num = len(boxes)
    bundle["num_detections"] = num
    if num == 0:
        bundle["total_time"] = time.perf_counter() - total_start
        return bundle

    # ── 2. Sort detections in natural reading order ──
    bboxes = [list(map(int, b.xyxy[0].tolist())) for b in boxes]
    order = reading_order_indices(bboxes)

    h, w = image_rgb.shape[:2]
    detections = []
    class_counts = {}
    for new_idx, orig_i in enumerate(order, start=1):
        box = boxes[orig_i]
        cls_name = result.names[int(box.cls)]
        conf_val = round(float(box.conf) * 100, 1)
        x1, y1, x2, y2 = bboxes[orig_i]
        # Clamp to image bounds.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image_rgb[y1:y2, x1:x2]

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        detections.append({
            "index": new_idx,
            "class": cls_name,
            "confidence": conf_val,
            "bbox": [x1, y1, x2, y2],
            "crop": crop,
            "ocr_text": "",
            "ocr_success": False,
            "ocr_engine": "",
        })

    bundle["classes"] = class_counts

    # ── 3. OCR each crop (with optional preprocessing) ──
    ocr_start = time.perf_counter()
    for det in detections:
        crop = det["crop"]
        if preprocess_options is not None and _preprocess_enabled(preprocess_options):
            ocr_input = preprocess_mod.preprocess_image(crop, preprocess_options)
        else:
            ocr_input = crop
        text, success, used_engine = ocr_mod.run_ocr(
            ocr_input, ocr_engine,
            easyocr_reader=easyocr_reader, tesseract_psm=tesseract_psm,
        )
        det["ocr_text"] = text if text else "[No text detected]"
        det["ocr_success"] = success
        det["ocr_engine"] = used_engine
    bundle["ocr_time"] = time.perf_counter() - ocr_start

    bundle["detections"] = detections
    bundle["ocr_success_count"] = sum(1 for d in detections if d["ocr_success"])
    bundle["ocr_fail_count"] = num - bundle["ocr_success_count"]

    # ── 4. Structured report rows + summary ──
    rows = build_report_rows(detections, source=source, page=page)
    bundle["report_rows"] = rows
    bundle["summary"] = summarise_fields(rows)

    bundle["total_time"] = time.perf_counter() - total_start
    return bundle


def _preprocess_enabled(options):
    """True if the preprocessing master switch is on."""
    if isinstance(options, dict):
        return bool(options.get("enabled", False))
    return bool(getattr(options, "enabled", False))


def bundle_log_record(bundle, run_id, ocr_engine_selected):
    """Build a JSON-serialisable per-page log record from a bundle."""
    return {
        "run_id": run_id,
        "source": bundle.get("source"),
        "page": bundle.get("page"),
        "num_detections": bundle.get("num_detections"),
        "classes": bundle.get("classes"),
        "ocr_engine_selected": ocr_engine_selected,
        "yolo_time_s": round(bundle.get("yolo_time", 0.0), 4),
        "ocr_time_s": round(bundle.get("ocr_time", 0.0), 4),
        "total_time_s": round(bundle.get("total_time", 0.0), 4),
        "ocr_success": bundle.get("ocr_success_count"),
        "ocr_fail": bundle.get("ocr_fail_count"),
        "error": bundle.get("error"),
        "detections": [
            {
                "order": d["index"],
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "ocr_text": d["ocr_text"],
                "ocr_engine": d["ocr_engine"],
            }
            for d in bundle.get("detections", [])
        ],
    }
