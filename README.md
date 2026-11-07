<!-- README status: DRAFT for member review — fill the placeholders marked (…) before push -->

# GROUP43_CENG313 — ECG Image Digitization (PhysioNet–Kaggle)

Course project on the closed Kaggle competition
["PhysioNet — Digitization of ECG Images"](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)
(Oct 2025 – Jan 2026). **Task:** reconstruct 12-lead ECG time series (mV) from images of
paper ECGs — color/BW scans, phone photos, stains, mold, laptop-screen photographs.

## Members

| Name | Student ID | GitHub username |
|------|-----------|-----------------|
| Mehmet Emircan Küllücek | 24118080060 | @mmilesmorales |
| Aleksei Khlopkov | 23118080404 | @13cide |

## Dataset

- **Inputs:** 977 training records, each with a ground-truth 12-lead time series
  (`train/<id>/<id>.csv`, sampling rates 250–1025 Hz) and **9 imaging variants**
  (clean render, color/BW scans, phone photos of prints, laptop-screen photos,
  stained/soaked prints, damaged prints, mold).
- **Target:** for each test image, the mV time series per lead — `floor(fs · 2.5 s)`
  samples for the 11 short leads, `floor(fs · 10 s)` for lead II.
- Local subset used here: a fixed record-level validation split (90/10, seed 0);
  the notebook fetches a 3.5 GB subset of it via the Kaggle API (see below).

## Approach (Assignment 1)

1. **EDA** — dataset statistics, the 9 imaging variants, and the paper's organizational
   constants (lead rows/columns, pixel calibration ≈ 197 px/s, ≈ 79 px/mV) verified
   against ground truth.
2. **Baseline 1 — all-zeros:** every submitted value = 0.0 (metric anchor ≈ 0 dB).
3. **Baseline 2 — classical CV digitization (from scratch, no ML):** grid-line
   autocorrelation → pixel calibration → fixed paper-layout cell extraction →
   per-column trace tracking → resampling → scoring.
4. **Evaluation:** the official competition metric, vendored
   (`src/metric.py`), with a record-level local validation split.

## Results (local validation subset, official metric)

| Baseline | Score |
|---|---|
| All-zeros | 0.254 dB |
| Classical CV | 1.016 dB |

On clean renders the CV baseline reaches up to **+15.4 dB** per record (mean +6.95),
while on real-world artifact variants most records fall below the zeros floor —
quantifying the domain gap that motivates learned rectification approaches.

## How to run the notebook

This repository is **private** — running the notebook requires collaborator access.

1. Open `assignment1_eda_baseline.ipynb` in **Google Colab** — best via
   [Open in Colab](https://colab.research.google.com/github/mmilesmorales/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb).
   Colab clones the private repository automatically for your logged-in collaborator
   GitHub account.
2. Cell 0 then downloads the dataset (~2.5 GB) from our shared Drive link — no Kaggle
   credentials involved. If you uploaded only the notebook instead, cell 0 asks once for
   a GitHub token (hidden input, used for a single clone, never stored).
3. Run all cells. Requires no GPU.

## Repository layout

```
├── assignment1_eda_baseline.ipynb    ← Assignment-1 deliverable (EDA + baselines)
├── src/
│   ├── data.py        dataset loading, record-level split, lead extraction
│   ├── metric.py      official competition metric (vendored) + local adapter
│   ├── baseline_cv.py classical-CV digitization baseline
│   └── utils.py       resampling, Einthoven-residual check, plotting
└── README.md
```

## Progress log

- 2026-10-24: project scaffold; competition research
- 2026-10-25: EDA; dataset-organization analysis; SNR harness; baselines 1–2 implemented
- 2026-11-02: results consolidated; community-baseline comparison documented
- (…) upcoming: Assignment-2 work
