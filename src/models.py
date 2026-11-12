"""Baseline models and their evaluation on records with ground truth.

Both baselines are rule-based, so there is nothing to train:
  * all zeros     predict 0 mV for every sample (the floor of the metric);
  * classical CV  read the trace from the image with image processing only
                  (the implementation is in baseline_cv.py).

The evaluate_* functions score a model on a list of record ids with the
official metric and return one row per image, so both assignments use the
same code for their numbers.
"""
import numpy as np
import pandas as pd

from .data import ROOT, extract_lead_signal, load_record_gt
from .metric import LEADS, competition_score_from_linear, record_linear_snr
from .baseline_cv import digitize_image, run_validation

IMAGE_TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


# ----------------------------------------------------------------------------
# Baseline 1: all zeros
# ----------------------------------------------------------------------------

def predict_zeros(gt: pd.DataFrame) -> dict:
    """A flat 0 mV line for every lead, as long as that lead's true series."""
    return {lead: np.zeros(len(extract_lead_signal(gt, lead))) for lead in LEADS}


def evaluate_zeros(record_ids, data_dir=ROOT / "data") -> pd.DataFrame:
    """Score the all-zeros prediction. One row per record: record, ratio (linear SNR), db."""
    fs_of = _sampling_rates(data_dir)
    rows = []
    for rec in record_ids:
        gt = load_record_gt(str(data_dir), rec)
        ratio = record_linear_snr(gt, predict_zeros(gt), fs_of[int(rec)])
        rows.append({"record": int(rec), "ratio": ratio, "db": 10 * np.log10(max(ratio, 1e-12))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Baseline 2: classical computer vision
# ----------------------------------------------------------------------------

def predict_classical_cv(img_rgb: np.ndarray, fs: int):
    """Digitize one RGB image. Returns ({lead: mV array}, status per lead),
    or (None, reason) when no grid or no signal rows are found."""
    return digitize_image(img_rgb, fs)


def evaluate_classical_cv(record_ids, image_types=IMAGE_TYPES, out_csv="cv_scores.csv",
                          verbose=False) -> pd.DataFrame:
    """Run the classical CV baseline on every available image of the records.

    One row per image: record, variant (image type), ratio (linear SNR), db, status.
    Leads that could not be read are scored as zeros. The table is also saved
    to `out_csv` in the repo root (git-ignored).
    """
    return run_validation(record_ids, variants=image_types, out_csv=out_csv, verbose=verbose)


# ----------------------------------------------------------------------------
# Shared
# ----------------------------------------------------------------------------

def score(ratios) -> float:
    """Competition score in dB from per-image linear SNR ratios (mean first, then dB)."""
    return competition_score_from_linear(ratios)


def _sampling_rates(data_dir) -> dict:
    train = pd.read_csv(f"{data_dir}/train.csv")
    return dict(zip(train.id, train.fs))
