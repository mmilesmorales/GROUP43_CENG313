# GROUP43_CENG313: ECG image digitization

CENG313 (Introduction to Data Science) course project on the Kaggle competition
[PhysioNet – Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization)
(Oct 2025 – Jan 2026, closed; late submissions are still scored).

**Task:** given a picture of a printed 12-lead ECG (clean render, scan, phone photo, photo of a screen,
stained or damaged paper), recover the 12 voltage curves in mV.

## Group

**Group ID: 43**

| Name | Student ID | GitHub |
|---|---|---|
| Mehmet Emircan Küllücek | 24118080060 | @mmilesmorales |
| Aleksei Khlopkov | 23118080404 | @13cide |

## Data

- 977 training records. Each has a ground-truth CSV (12 leads, 10 s, sampled at 250–1025 Hz)
  and 9 images of the same ECG (types `0001` … `0012`, `0001` is the clean render).
  The hidden test set (about 1000 images) is only available inside a Kaggle notebook.
- For local work, records are split 90/10 **by record** (seed 0), so all images of one ECG stay
  on one side.
- The full data are 85.75 GB. We use a subset (about 2.5 GB) with the validation records:
  ground truth for 98 records and all 9 images for the first 30. Local scores use those 30 records
  (270 images).

**Getting the data.** The notebooks download our subset automatically into `data/` (from the
team's Google Drive, with `gdown`). The full data can be downloaded from the
[competition data page](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data)
after accepting the competition rules. `data/` is never committed.

## Assignment 1: EDA and baseline

Notebook: [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb)

1. **Problem definition:** task, target (one mV sequence per lead) and the official SNR metric.
2. **Data overview:** shapes, types, missing values and duplicates of all files.
3. **EDA:** sampling rates and row counts, the signals and their energy per lead, Einthoven's law,
   the nine image types, how the metric reacts to errors, and a check of the clean-page template.
4. **Preprocessing:** no imputation (the empty cells are page layout), outliers kept, trace mask
   and scale from the grid, resampling; each decision measured with the metric.
5. **Baselines:** all zeros, and a classical computer-vision digitizer (no learning).
6. **Kaggle submission:** late-submission scores and why they differ from the local ones.
7. **Conclusion:** key findings and next steps.

### Results — local validation & Kaggle late submissions

Late submissions (not on the leaderboard; the competition is closed). Score screenshot:
[`submissions/submissions.png`](submissions/submissions.png).

| Method | Local validation, 30 records (dB) | Kaggle private (dB) | Kaggle public (dB) |
|---|---|---|---|
| All zeros | 0.262 | 0.106 | 0.094 |
| Classical CV, all 9 image types | 1.023 | -4.311 | -4.073 |
| Classical CV, clean `0001` images only | 8.26 | – | – |
| *Reference, not ours:* public pretrained pipeline by hengck23 ([Kaggle notebook](https://www.kaggle.com/code/hengck23/demo-submission)) | *17.989* | – | – |

The classical baseline works on clean renders but is worse than zeros on most scans and photos,
because those pages are shifted, rotated or warped relative to the template. Its local average is
carried by the few clean pages, which is why it drops below zero on the hidden test. For scale:
the competition winners reached about 23 dB on the private test set.
## How to run

The repository is public, so no account or token is needed. The data are downloaded
automatically by the first cell of each notebook (our ~2.5 GB subset, from the team's Google Drive)
into `data/`; this takes a few minutes the first time and is skipped afterwards.

**Option 1: Google Colab (easiest, no GPU needed)**
1. Open [Assignment 1 in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb).
2. Runtime → Run all. The first cell clones this repo and downloads the data.

The whole notebook runs in a few minutes; the classical CV baseline (section 5.2) is the slowest cell.

**Option 2: locally (Python 3.10 or newer)**
```
git clone https://github.com/ceng313-2026/GROUP43_CENG313.git
cd GROUP43_CENG313
pip install -r requirements.txt
jupyter notebook assignment1_eda_baseline.ipynb
```
Start Jupyter from the repo root (the notebooks import code from `src/`), then run all cells.
If the automatic download fails, download the
[data subset zip](https://drive.google.com/file/d/15VNuF-QR-IiCfIl4lqVwbvO-lLehtckv/view)
manually and unzip it into `data/`, so that `data/train.csv` exists.

**Just reading:** the notebooks are committed with all outputs, so they can be read on GitHub
without running anything. If GitHub cannot display one, use
[nbviewer](https://nbviewer.org/github/ceng313-2026/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb).

## Kaggle submission history

| Date | Kernel | Submission | Public (dB) | Private (dB) |
|---|---|---|---|---|
| 2026-11-11 | `physionet-late-submission-zeros` | all-zeros baseline | 0.094 | 0.106 |
| 2026-11-11 | `physionet-cv-submission` | classical CV baseline | -4.073 | -4.311 |

Screenshot: [`submissions/submissions.png`](submissions/submissions.png).

## Progress log

| Date | Contributor | Work |
|---|---|---|
| 2026-10-24 | Emircan | project scaffold, Kaggle API setup, competition research, official metric vendored into `src/` |
| 2026-10-25 | Emircan | EDA: dataset structure, nine image variants, paper layout and calibration verified against ground truth; SNR harness validated; all-zeros and classical-CV baselines implemented |
| 2026-11-02 | Emircan | results consolidated; community pretrained pipeline (hengck23) run as external reference on the same validation cohort (17.989 dB) |
| 2026-11-07/11 | Emircan | dataset subset packaged (2.5 GB) and published on Drive for reproducible notebook runs; Kaggle late submissions tested, accepted and completed for both baselines (scores in the table above); score screenshot added to `submissions/` |
| 2026-11-09–10 | Aleksei | `requirements.txt`, shared dataset downloader, `src/layout.py`, notebook restructure with per-plot interpretations, README rewrite |
| 2026-11-12 | Aleksei | repository transferred into the course organization `ceng313-2026`; course-structure alignment (`__init__.py`, `models.py`, `submission_frame`); notebook sections numbered per the course template; Kaggle scores merged into the README results |

## Repository layout

```
├── README.md
├── requirements.txt
├── .gitignore                       excludes data/
├── assignment1_eda_baseline.ipynb   Assignment 1: EDA and baseline
├── assignment2_solution.ipynb       Assignment 2: our solution (in progress)
├── src/
│   ├── __init__.py
│   ├── data.py         dataset download, record-level split, ground-truth loading
│   ├── models.py       the two baselines and their evaluation with the metric
│   ├── utils.py        resampling, Einthoven residuals, Kaggle submission table
│   ├── metric.py       official competition metric (vendored) + per-record helpers
│   ├── layout.py       clean-page template and the ink hit-rate check
│   └── baseline_cv.py  classical CV digitizer used by models.py
└── submissions/        Kaggle score screenshots
```
