# ==========================================================
# Detection helpers
# ----------------------------------------------------------
# Pure, Streamlit-free helpers for ordering detections in
# natural reading order and summarising model metadata. The
# heavy YOLO model object is created/cached in app.py and passed
# in, keeping this module unit-testable.
# ==========================================================

import os


def reading_order_indices(bboxes, row_tol_ratio=0.6):
    """Return indices of ``bboxes`` in natural reading order.

    Reading order = top-to-bottom by row, then left-to-right
    within each row. Boxes are grouped into horizontal "bands"
    whose vertical centres fall within ``row_tol_ratio`` × the
    median box height of one another; bands are emitted top-down
    and items inside a band are sorted by left edge (x1).

    ``bboxes`` is a list of [x1, y1, x2, y2]. Returns a list of
    integer indices into the original list.
    """
    n = len(bboxes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    heights = [max(1.0, float(b[3] - b[1])) for b in bboxes]
    median_h = sorted(heights)[len(heights) // 2]
    tol = median_h * row_tol_ratio
    centers_y = [(float(b[1]) + float(b[3])) / 2.0 for b in bboxes]

    # Process boxes top-to-bottom so bands are created in order.
    by_y = sorted(range(n), key=lambda i: centers_y[i])

    bands = []  # each: {"anchor": cy_of_first_item, "items": [idx, ...]}
    for i in by_y:
        placed = False
        for band in bands:
            if abs(centers_y[i] - band["anchor"]) <= tol:
                band["items"].append(i)
                placed = True
                break
        if not placed:
            bands.append({"anchor": centers_y[i], "items": [i]})

    order = []
    for band in bands:
        order.extend(sorted(band["items"], key=lambda i: float(bboxes[i][0])))
    return order


def model_info(model, model_path):
    """Build a metadata dict for the model-info panel.

    Returns name, class list, class count, task and library
    version. Inference time is tracked separately by the caller.
    """
    info = {
        "name": os.path.basename(model_path),
        "path": model_path,
        "classes": [],
        "num_classes": 0,
        "version": "unknown",
        "task": "detect",
    }
    try:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            classes = [names[k] for k in sorted(names)]
        elif names is not None:
            classes = list(names)
        else:
            classes = []
        info["classes"] = classes
        info["num_classes"] = len(classes)
        info["task"] = getattr(model, "task", "detect") or "detect"
    except Exception:
        pass

    try:
        import ultralytics
        info["version"] = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        pass

    return info
