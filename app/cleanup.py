# ==========================================================
# Temp-file housekeeping
# ----------------------------------------------------------
# Removes stale files from the temp directory so it does not grow
# unbounded across sessions. Called once at app startup.
# ==========================================================

import os
import time


def clean_old_temp(temp_dir, max_age_hours=24, logger=None):
    """Delete files in ``temp_dir`` older than ``max_age_hours``.

    Returns the number of files removed. Best-effort: per-file
    errors are logged (if a logger is given) and skipped.
    """
    if not temp_dir or not os.path.isdir(temp_dir):
        return 0

    cutoff = time.time() - float(max_age_hours) * 3600.0
    removed = 0

    for entry in os.listdir(temp_dir):
        path = os.path.join(temp_dir, entry)
        try:
            if not os.path.isfile(path):
                continue
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except Exception as exc:
            if logger:
                logger.warning("Could not remove temp file %s: %s", path, exc)

    if logger and removed:
        logger.info("Cleaned %d stale temp file(s) from %s", removed, temp_dir)
    return removed
