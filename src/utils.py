"""Shared utilities: resampling, the Einthoven residual check and the submission table."""
import numpy as np


# ----------------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------------

def resample_to_rows(series: np.ndarray, n_rows: int) -> np.ndarray:
    """Linear resampling of an extracted per-column series to the required
    number of submission rows (`floor(fs*2.5)` / `floor(fs*10)`)."""
    series = np.asarray(series, dtype=np.float64)
    if len(series) == 0:
        return np.zeros(n_rows)
    return np.interp(np.linspace(0, 1, n_rows), np.linspace(0, 1, len(series)), series)


# ----------------------------------------------------------------------------
# Einthoven law utilities
# ----------------------------------------------------------------------------

def einthoven_residual(gt: 'pd.DataFrame'):  # noqa: F821  (pd imported below)
    """Free consistency check on ground-truth (or extracted) limb leads.

    Returns dict with rms of II, residual II - I - III, and the augmented-lead
    sum aVR + aVL + aVF. For TRUE signals these are ~0 by construction; for our
    extractions any residual is pure digitization error.
    """
    import pandas as pd  # local import to keep this module dependency-light at top
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import extract_lead_signal  # noqa: E402

    I = extract_lead_signal(gt, 'I'); III = extract_lead_signal(gt, 'III'); II = extract_lead_signal(gt, 'II')
    n = min(len(I), len(III), len(II))
    resid = II[:n] - I[:n] - III[:n]
    avr = extract_lead_signal(gt, 'aVR'); avl = extract_lead_signal(gt, 'aVL'); avf = extract_lead_signal(gt, 'aVF')
    m = min(len(avr), len(avl), len(avf))
    return {
        'rms_II': float(np.sqrt((II[:n] ** 2).mean())),
        'rms_resid_III': float(np.sqrt((resid ** 2).mean())),
        'rms_augmented_sum': float(np.sqrt(((avr[:m] + avl[:m] + avf[:m]) ** 2).mean())),
    }


# ----------------------------------------------------------------------------
# Kaggle submission
# ----------------------------------------------------------------------------

def submission_frame(test: 'pd.DataFrame', preds: dict = None) -> 'pd.DataFrame':  # noqa: F821
    """Build the submission table for Kaggle from test.csv.

    test:  one row per image and lead, with columns id, lead, number_of_rows.
    preds: {(image id, lead): 1-D array in mV}. A lead that is missing, has the
           wrong length or holds NaN/inf is written as zeros, so the file is
           always valid for the metric.
    Returns the columns id ("{image id}_{row}_{lead}") and value.
    """
    import pandas as pd
    preds = preds or {}
    ids, values = [], []
    for image_id, lead, n in zip(test["id"], test["lead"], test["number_of_rows"]):
        v = np.asarray(preds.get((image_id, lead), np.zeros(n)), dtype=np.float64)
        if len(v) != n or not np.isfinite(v).all():
            v = np.zeros(n)
        ids.extend(f"{image_id}_{k}_{lead}" for k in range(n))
        values.append(v)
    return pd.DataFrame({"id": ids, "value": np.concatenate(values) if values else np.zeros(0)})
