"""Baseline 2 — Classical computer-vision extraction (no machine learning).

Pipeline (image-only, no learned components):
  1. detect the ECG grid (red-ish lines on colour variants, dark full-span lines on B/W)
     -> big-square pitch in px -> calibration: px/s (big sq = 0.2 s) and px/mV (0.5 mV)
  2. fixed paper layout (from the ecg-image-kit renderer, verified empirically):
       row A: I  aVR V1 V4      row B: II aVL V2 V5
       row C: III aVF V3 V6     row D: rhythm strip = lead II, 10 s full width
     plot area starts at x_gap = floor(((x_max - 10 s)/2)/0.2)*0.2 seconds.
  3. per cell: per-column centroid of the trace mask (dark & not reddish),
     column-to-column run tracking; NaN columns interpolated.
  4. resample to the required number of samples (floor(fs*2.5) / floor(fs*10)).
  5. extraction failure -> zeros fallback (0 dB) for that lead.

Outputs a per-(record, variant) score table (linear ratios) + summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_record_gt
from metric import record_linear_snr, competition_score_from_linear, n_samples

LEAD_CELLS = {  # paper row (A,B,C,D) -> [(lead, col), ...]
    "A": [("I", 0), ("aVR", 1), ("V1", 2), ("V4", 3)],
    "B": [("II", 0), ("aVL", 1), ("V2", 2), ("V5", 3)],
    "C": [("III", 0), ("aVF", 1), ("V3", 2), ("V6", 3)],
    "D": [("II", None)],  # full-width strip, handled separately
}


def find_line_groups(mask_1d_bool, min_len=1):
    idx = np.flatnonzero(mask_1d_bool)
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups if len(g) >= min_len]


def detect_pitch_autocorr(profile):
    """Sub-pixel fine-grid pitch via autocorrelation peak + parabolic refinement."""
    p = profile.astype(np.float64)
    p = p - p.mean()
    a = np.correlate(p, p, mode='full')[len(p)-1:]
    a = a / (a[0] + 1e-12)
    lo, hi = 4, 60
    seg = a[lo:hi]
    k = int(np.argmax(seg)) + lo
    if 1 <= k < len(a)-1:
        den = a[k-1] - 2*a[k] + a[k+1]
        delta = 0.0 if abs(den) < 1e-12 else 0.5*(a[k-1] - a[k+1])/den
        delta = float(np.clip(delta, -0.49, 0.49))
    else:
        delta = 0.0
    return k + delta


def geometry_from_image(img_u8):
    """Estimate calibration from the grid.

    Fine grid = 1 mm squares (0.04 s / 0.1 mV); big = 5 fine (0.2 s / 0.5 mV).
    Pitch via autocorrelation (sub-pixel). Plot-area origin anchored on the
    trace's right edge: plot spans exactly 10 s ending at the trace's last column.
    """
    r = img_u8[..., 0].astype(np.int16); g = img_u8[..., 1].astype(np.int16); b = img_u8[..., 2].astype(np.int16)
    reddish = (r > g + 20) & (r > b + 20)
    gray = img_u8.astype(np.float32).mean(2)
    dark = gray < 110
    H, W = dark.shape

    grid_mask = reddish if reddish.mean() > 0.002 else dark

    pitch_x = detect_pitch_autocorr(grid_mask.sum(0))
    pitch_y = detect_pitch_autocorr(grid_mask.sum(1))
    if pitch_x is None or pitch_y is None or not (3 < pitch_x < 80) or not (3 < pitch_y < 80):
        return None
    if pitch_x >= 15:
        pitch_x /= 5.0
    if pitch_y >= 15:
        pitch_y /= 5.0
    px_per_s = pitch_x / 0.04
    px_per_mV = pitch_y / 0.1

    # right edge of the drawn trace (rows A-D live in the lower ~3/4 of the paper)
    trace = dark & ~reddish
    band_rows = trace[int(H*0.3):int(H*0.99)]
    cols_with_trace = np.flatnonzero(band_rows.sum(0) > 2)
    cols_with_trace = cols_with_trace[cols_with_trace < W - 15]   # exclude right border
    if len(cols_with_trace) < 100:
        return None
    right_edge = float(cols_with_trace.max())
    x0_px = right_edge - 10.0 * px_per_s
    return dict(pitch_x=pitch_x, pitch_y=pitch_y, px_per_s=px_per_s,
                px_per_mV=px_per_mV, x0_px=x0_px, cell_px=2.5 * px_per_s)


def trace_runs_in_column(mask2d, y0, y1, x):
    col = mask2d[y0:y1, x]
    idx = np.flatnonzero(col)
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    return [(float(g.mean() + y0), len(g), int(g[0] + y0)) for g in groups]


def extract_cell(black, y0, y1, x0, x1, px_per_s):
    """Per-column centroid with simple run tracking. Returns (mv_series, t_s) or None."""
    width = x1 - x0
    ys = np.full(width, np.nan)
    prev_y = None
    # seed: median y of runs in columns with exactly one run
    seeds = []
    for k, x in enumerate(range(x0, x1)):
        runs = trace_runs_in_column(black, y0, y1, x)
        if len(runs) == 1:
            seeds.append(runs[0][0])
    if len(seeds) < width * 0.2:
        return None
    prev_y = float(np.median(seeds))

    for k, x in enumerate(range(x0, x1)):
        runs = trace_runs_in_column(black, y0, y1, x)
        if not runs:
            continue
        if len(runs) == 1 or prev_y is None:
            run = min(runs, key=lambda rr: abs(rr[0] - (prev_y if prev_y is not None else rr[0])))
        else:
            run = min(runs, key=lambda rr: abs(rr[0] - prev_y))
        ys[k] = run[0]
        prev_y = run[0]

    fin = np.flatnonzero(np.isfinite(ys))
    if len(fin) < width * 0.3:
        return None
    t = (np.arange(x0, x1) - x0) / px_per_s
    mv = np.interp(np.arange(width), fin, ys[fin])
    return mv, t


def digitize_image(img_u8, fs):
    """Full 12-lead extraction from one image. Returns dict lead -> mv array (or None)."""
    geom = geometry_from_image(img_u8)
    if geom is None:
        return None, "no grid"
    r = img_u8[..., 0].astype(np.int16); g = img_u8[..., 1].astype(np.int16); b = img_u8[..., 2].astype(np.int16)
    reddish = (r > g + 20) & (r > b + 20)
    gray = img_u8.astype(np.float32).mean(2)
    black = (gray < 110) & ~reddish

    H, W = black.shape
    if reddish.mean() < 0.002:
        # B/W variant: the grid is dark too -> remove full-span lines (the grid)
        vert = black.sum(0) > 0.75 * H
        horiz = black.sum(1) > 0.75 * W
        black = black & ~vert[None, :] & ~horiz[:, None]
    # --- row detection via histogram peaks of trace pixels (robust to rows that touch) ---
    x0_plot = max(int(geom["x0_px"]) - 5, 0)
    black = black.copy()
    black[:, :x0_plot] = False
    colcnt = black.sum(0)
    clean_cols = colcnt < 0.30 * H
    hist = (black * clean_cols[None, :]).sum(1).astype(float) if clean_cols.any() else black.sum(1).astype(float)
    k = np.ones(15) / 15
    hs = np.convolve(hist, k, mode='same')
    from scipy.signal import find_peaks
    peaks0, _ = find_peaks(hs, distance=50, prominence=5)
    peaks0 = [int(p) for p in peaks0 if p > 0.2 * H]          # drop the patient header
    if len(peaks0) < 4:
        return None, f"peaks<4 ({len(peaks0)})"
    diffs = np.diff(peaks0)
    row_d = diffs[(diffs > 120) & (diffs < 450)]
    pitch0 = float(np.median(row_d)) if len(row_d) else 260.0
    peaks, _ = find_peaks(hs, distance=int(0.65 * pitch0), prominence=5)
    peaks = [int(p) for p in peaks if p > 0.2 * H]
    if len(peaks) < 4:
        return None, f"peaks2<4 ({len(peaks)})"
    centers = peaks[-4:]                                      # rows top->bottom = A..D
    pitch = float(np.median(np.diff(centers))) if len(centers) > 2 else pitch0
    rows = {nm: (max(0, int(c - 0.45 * pitch)), min(H, int(c + 0.45 * pitch)))
            for nm, c in zip("ABC", centers[:3])}
    rows["D"] = (max(0, int(centers[3] - 0.45 * pitch)), H - 12)   # strip: run to paper bottom

    x0 = int(geom["x0_px"]); cell = geom["cell_px"]
    preds, meta = {}, {}
    for row_name, leads in LEAD_CELLS.items():
        y0, y1 = rows[row_name]
        pad = 15
        y0p, y1p = max(0, y0 - pad), min(H - 12, y1 + pad)   # avoid the paper border
        if row_name == "D":
            xs0, xs1 = x0, x0 + int(10 * geom["px_per_s"])
            out = extract_cell(black, y0p, y1p, xs0, min(xs1, W), geom["px_per_s"])
            if out is None:
                preds["II"] = None; meta["II"] = "strip failed"
                continue
            mv, t = out
            mv_mv = -(mv - np.nanmedian(mv)) / geom["px_per_mV"]
            nreq = int(np.floor(fs * 10.0))
            tt = np.linspace(0, t[-1], nreq)
            preds["II"] = np.interp(tt, t, mv_mv)
            meta["II"] = "strip"
        else:
            for lead, ci in leads:
                xs0 = int(x0 + ci * cell); xs1 = int(x0 + (ci + 1) * cell)
                out = extract_cell(black, y0p, y1p, xs0, min(xs1, W), geom["px_per_s"])
                if out is None:
                    preds[lead] = None; meta[lead] = "cell failed"
                    continue
                mv, t = out
                mv_mv = -(mv - np.nanmedian(mv)) / geom["px_per_mV"]
                nreq = int(np.floor(fs * 2.5))
                tt = np.linspace(0, t[-1], nreq)
                preds[lead] = np.interp(tt, t, mv_mv)
                meta[lead] = "ok"
    return {k: v for k, v in preds.items() if v is not None}, meta


def run_validation(val_ids_in, variants=('0001','0003','0004','0005','0006','0009','0010','0011','0012'), out_csv='reports/cv_baseline_scores.csv', verbose=True):
    val_ids = [int(x) for x in val_ids_in]
    train_csv = pd.read_csv(ROOT / "data" / "train.csv")
    fs_of = dict(zip(train_csv.id, train_csv.fs))
    variants = list(variants)

    rows_out = []
    for rec in val_ids:
        gt = load_record_gt(str(ROOT / "data"), rec)
        fs = fs_of[rec]
        for v in variants:
            img_path = ROOT / "data" / "train" / str(rec) / f"{rec}-{v}.png"
            if not img_path.exists():
                continue
            from PIL import Image
            img = np.asarray(Image.open(img_path).convert("RGB"))
            preds, meta = digitize_image(img, fs)
            if preds is None:
                ratio = 1.0; status = f"fallback-zeros ({meta})"
                full = {L: np.zeros(len(gt[L].dropna())) for L in gt.columns}
            else:
                full = {L: (preds[L] if L in preds else np.zeros(len(gt[L].dropna())))
                        for L in gt.columns}
                ratio = record_linear_snr(gt, full, fs)
                status = "ok" if all(m == "ok" for m in meta.values()) else json.dumps(meta)
            rows_out.append(dict(record=rec, variant=v, ratio=ratio,
                                 db=10 * np.log10(max(ratio, 1e-12)), status=status[:120]))
            if verbose: print(f"{rec} {v}: {10*np.log10(max(ratio,1e-12)):7.2f} dB  {status[:80]}", flush=True)

    df = pd.DataFrame(rows_out)
    out = ROOT / out_csv
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    if verbose: print("\n=== per-variant mean record-dB ===")
    if verbose: print(df.groupby('variant').db.mean().round(2).to_string())
    if verbose: print(f'\n=== FINAL competition-style score (all rows): {competition_score_from_linear(df.ratio):.3f} dB ===')
    if verbose: print(f'saved -> {out}')
    return df


