"""Shared utilities: resampling, Einthoven residual check, small plotting helpers."""
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
