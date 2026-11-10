# GROUP43_CENG313: ECG image digitization

CENG313 course project on the Kaggle competition
[PhysioNet – Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)
(Oct 2025 – Jan 2026, now closed).

**Task:** given a picture of a printed 12-lead ECG (clean render, scan, phone photo, photo of a screen,
stained or damaged paper), recover the 12 voltage curves in mV.

## Members

| Name | Student ID | GitHub |
|---|---|---|
| Mehmet Emircan Küllücek | 24118080060 | @mmilesmorales |
| Aleksei Khlopkov | 23118080404 | @13cide |

## Data

- 977 training records. Each has a ground-truth CSV (12 leads, 10 s, sampled at 250–1025 Hz)
  and 9 images of the same ECG (types `0001` … `0012`, `0001` is the clean render).
- The competition is closed, so we evaluate locally. Records are split 90/10 **by record**
  (seed 0), so all images of one ECG stay on one side.
- Our subset (about 2.5 GB, on Google Drive) holds the validation records: ground truth for 98
  records and all 9 images for the first 30. All scores below use those 30 records (270 images).

## Assignment 1: EDA and baselines

Notebook: [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb)

- **EDA:** sampling rates and exact row counts, how the ground truth maps onto the printed page,
  amplitude and energy per lead, Einthoven's law in the ground truth, and the nine image types.
- **Metric tests:** how the official SNR metric reacts to offset, delay, noise, amplitude and
  time-scale errors.
- **Layout check:** the clean-page template (196.85 px/s, 78.74 px/mV, fixed row positions) is
  confirmed by drawing the ground truth onto the image and measuring how much lands on ink.
- **Baseline 1, all zeros:** predict 0 mV everywhere.
- **Baseline 2, classical CV (no learning):** grid detection → calibration → fixed lead cells →
  column-by-column trace following → resampling.

### Results (official metric, 30 validation records)

| Method | Score (dB) |
|---|---|
| All zeros | 0.262 |
| Classical CV, all 9 image types | 1.016 |
| Classical CV, clean `0001` images only | 8.26 |
| *Reference, not ours:* community pretrained pipeline ([link](https://github.com/Yashjamine1207/physionet-ecg-image-digitization)) | *17.989* |

The classical baseline works on clean renders but is worse than zeros on most scans and photos,
because those pages are shifted, rotated or warped relative to the template. Details per image
type are in section 7 of the notebook.

## Assignment 2: proposed solution

Notebook: [`assignment2_solution.ipynb`](assignment2_solution.ipynb) (in progress).

## How to run

**Google Colab (recommended, no GPU needed)**
1. Open the notebook from GitHub in Colab, e.g.
   [Assignment 1 in Colab](https://colab.research.google.com/github/mmilesmorales/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb).
2. Add a Colab secret named `GITHUB_TOKEN` (key icon in the left sidebar) with a GitHub token
   that can read this private repo, and allow the notebook to access it.
3. Run all cells. The first cell clones the repo and downloads the data subset (first run only).

**Locally**
```
pip install -r requirements.txt
jupyter notebook   # start from the repo root
```
The first cell downloads the data into `data/` if it is not there yet.

## Repository layout

```
├── assignment1_eda_baseline.ipynb   Assignment 1: EDA and baselines
├── assignment2_solution.ipynb       Assignment 2: our solution (in progress)
├── requirements.txt
└── src/
    ├── data.py         dataset download, record-level split, ground-truth loading
    ├── metric.py       official competition metric (vendored) + per-record helpers
    ├── layout.py       clean-page template and the ink hit-rate check
    ├── baseline_cv.py  classical CV baseline
    └── utils.py        resampling, Einthoven residuals
```
`data/` is git-ignored.
