"""
Official PhysioNet ECG signal extraction metric — vendored verbatim from the public
Kaggle metric kernel:
https://www.kaggle.com/code/metric/physionet-ecg-signal-extraction-metric/

Local-evaluation adapter added at the bottom:
  - record_linear_snr(): score ONE record when we HAVE its ground truth
  - competition_score(): final score from a list of per-record linear SNRs

Competition scoring semantics (important):
  * per record: signal/noise powers are summed over the 12 leads -> ONE linear ratio
    (capped at PERFECT_SCORE = 1e6)
  * final score = 10*log10(mean(linear ratios)), floored at -384
  * an all-zeros submission scores exactly 0 dB (ratio = 1 per record)
"""
from typing import Tuple

import numpy as np
import pandas as pd

import scipy.optimize
import scipy.signal


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAX_TIME_SHIFT = 0.2
PERFECT_SCORE = 1e6


class ParticipantVisibleError(Exception):
    pass


def compute_power(label: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    if label.ndim != 1 or prediction.ndim != 1:
        raise ParticipantVisibleError('Inputs must be 1-dimensional arrays.')
    finite_mask = np.isfinite(prediction)
    if not np.any(finite_mask):
        raise ParticipantVisibleError("The 'prediction' array contains no finite values (all NaN or inf).")

    prediction[~np.isfinite(prediction)] = 0
    noise = label - prediction
    p_signal = np.sum(label**2)
    p_noise = np.sum(noise**2)
    return p_signal, p_noise


def compute_snr(signal: float, noise: float) -> float:
    if noise == 0:
        # Perfect reconstruction
        snr = PERFECT_SCORE
    elif signal == 0:
        snr = 0
    else:
        snr = min((signal / noise), PERFECT_SCORE)
    return snr


def align_signals(label: np.ndarray, pred: np.ndarray, max_shift: float = float('inf')) -> np.ndarray:
    if np.any(~np.isfinite(label)):
        raise ParticipantVisibleError('values in label should all be finite')
    if np.sum(np.isfinite(pred)) == 0:
        raise ParticipantVisibleError('prediction can not all be infinite')

    # Initialize the reference and digitized signals.
    label_arr = np.asarray(label, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)

    label_mean = np.mean(label_arr)
    pred_mean = np.mean(pred_arr)

    label_arr_centered = label_arr - label_mean
    pred_arr_centered = pred_arr - pred_mean

    # Compute the correlation between the reference and digitized signals and locate the maximum correlation.
    correlation = scipy.signal.correlate(label_arr_centered, pred_arr_centered, mode='full')

    n_label = np.size(label_arr)
    n_pred = np.size(pred_arr)

    lags = scipy.signal.correlation_lags(n_label, n_pred, mode='full')
    valid_lags_mask = (lags >= -max_shift) & (lags <= max_shift)

    max_correlation = np.nanmax(correlation[valid_lags_mask])
    all_max_indices = np.flatnonzero(correlation == max_correlation)
    best_idx = min(all_max_indices, key=lambda i: abs(lags[i]))
    time_shift = lags[best_idx]
    start_padding_len = max(time_shift, 0)
    pred_slice_start = max(-time_shift, 0)
    pred_slice_end = min(n_label - time_shift, n_pred)
    end_padding_len = max(n_label - n_pred - time_shift, 0)
    aligned_pred = np.concatenate((np.full(start_padding_len, np.nan), pred_arr[pred_slice_start:pred_slice_end], np.full(end_padding_len, np.nan)))

    def objective_func(v_shift):
        return np.nansum((label_arr - (aligned_pred - v_shift)) ** 2)

    if np.any(np.isfinite(label_arr) & np.isfinite(aligned_pred)):
        results = scipy.optimize.minimize_scalar(objective_func, method='Brent')
        vertical_shift = results.x
        aligned_pred -= vertical_shift
    return aligned_pred


def _calculate_image_score(group: pd.DataFrame) -> float:
    """Helper function to calculate the total SNR score for a single image group."""

    unique_fs_values = group['fs'].unique()
    if len(unique_fs_values) != 1:
        raise ParticipantVisibleError('Sampling frequency should be consistent across each ecg')
    sampling_frequency = unique_fs_values[0]
    if sampling_frequency != int(len(group[group['lead'] == 'II']) / 10):
        raise ParticipantVisibleError('The sequence_length should be sampling frequency * 10s')
    sum_signal = 0
    sum_noise = 0
    for lead in LEADS:
        sub = group[group['lead'] == lead]
        label = sub['value_true'].values
        pred = sub['value_pred'].values

        aligned_pred = align_signals(label, pred, int(sampling_frequency * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(label, aligned_pred)
        sum_signal += p_signal
        sum_noise += p_noise
    return compute_snr(sum_signal, sum_noise)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Official entry point (kept verbatim; see metric kernel for the full docstring)."""
    for df in [solution, submission]:
        if row_id_column_name not in df.columns:
            raise ParticipantVisibleError(f"'{row_id_column_name}' column not found in DataFrame.")
        if df['value'].isna().any():
            raise ParticipantVisibleError('NaN exists in solution/submission')
        if not np.isfinite(df['value']).all():
            raise ParticipantVisibleError('Infinity exists in solution/submission')

    submission = submission[['id', 'value']]
    merged_df = pd.merge(solution, submission, on=row_id_column_name, suffixes=('_true', '_pred'))
    merged_df['image_id'] = merged_df[row_id_column_name].str.split('_').str[0]
    merged_df['row_id'] = merged_df[row_id_column_name].str.split('_').str[1].astype('int64')
    merged_df['lead'] = merged_df[row_id_column_name].str.split('_').str[2]
    merged_df.sort_values(by=['image_id', 'row_id', 'lead'], inplace=True)
    image_scores = merged_df.groupby('image_id').apply(_calculate_image_score, include_groups=False)
    return max(float(10 * np.log10(image_scores.mean())), -384)


# ---------------------------------------------------------------------------
# Local-evaluation adapter (Assignment 1 harness)
# ---------------------------------------------------------------------------

# Data-structure discovery (verified on records 1006427285 & 1006867983):
# the GT csv has 10s of rows (= lead II length). The 11 short leads (2.5 s each)
# are NOT contiguous in their column: each lead's samples sit in the row-block
# corresponding to its vertical position on the PAPER:
#   block 0: I, II, III | block 1: aVR, aVL, aVF | block 2: V1..V3 | block 3: V4..V6
# Lead II is populated across all rows (the 10s rhythm strip).
PAPER_BLOCK = {'I': 0, 'II': 0, 'III': 0, 'aVR': 1, 'aVL': 1, 'aVF': 1,
               'V1': 2, 'V2': 2, 'V3': 2, 'V4': 3, 'V5': 3, 'V6': 3}


def extract_lead_signal(gt: pd.DataFrame, lead: str) -> np.ndarray:
    """The lead's own 2.5s (or 10s for II) series, taken from its finite span."""
    v = gt[lead].values
    finite = np.flatnonzero(np.isfinite(v))
    if len(finite) == 0:
        return np.zeros(0)
    runs = np.split(finite, np.where(np.diff(finite) != 1)[0] + 1)
    run = max(runs, key=len)          # longest contiguous finite run
    return v[run[0]:run[-1] + 1].astype(np.float64)


def load_record_gt(data_dir: str, record_id) -> pd.DataFrame:
    """Ground-truth time series of one record: columns = 12 leads."""
    return pd.read_csv(f"{data_dir}/train/{record_id}/{record_id}.csv")


def record_linear_snr(gt: pd.DataFrame, preds: dict, fs: int, return_aligned: bool = False):
    """
    Score ONE record with the official logic, when ground truth is available.

    Args:
        gt:     DataFrame with one column per lead (index = sample index).
        preds:  dict lead -> 1-D np.array. Length should be floor(fs*10) for 'II'
                and floor(fs*2.5) otherwise; align_signals tolerates mismatches.
        fs:     sampling frequency of the record.
    Returns:
        linear SNR ratio for this record (1.0 = zeros-level, 1e6 cap = perfect),
        plus (optionally) the aligned predictions per lead for diagnostics.
    """
    sum_signal, sum_noise = 0.0, 0.0
    aligned_all = {}
    for lead in LEADS:
        # Each lead's truth = its own contiguous series (see PAPER_BLOCK note).
        label = extract_lead_signal(gt, lead)
        if len(label) == 0:
            label = np.zeros(1)
        pred = np.asarray(preds.get(lead, np.zeros(len(label))), dtype=np.float64)
        aligned = align_signals(label, pred, int(fs * MAX_TIME_SHIFT))
        p_signal, p_noise = compute_power(label, aligned.copy())
        sum_signal += p_signal
        sum_noise += p_noise
        if return_aligned:
            aligned_all[lead] = aligned
    snr = compute_snr(sum_signal, sum_noise)
    return (snr, aligned_all) if return_aligned else snr


def competition_score_from_linear(record_linear_snrs) -> float:
    """Official aggregation: dB of the mean of per-record linear ratios, floored at -384."""
    vals = np.asarray(list(record_linear_snrs), dtype=np.float64)
    return max(float(10.0 * np.log10(vals.mean())), -384.0)


def n_samples(fs: int, lead: str) -> int:
    """Required number of prediction rows: floor(fs*10) for lead II, floor(fs*2.5) otherwise."""
    return int(np.floor(fs * (10.0 if lead == 'II' else 2.5)))
