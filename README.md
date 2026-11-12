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

## Current Kaggle score

Late submissions (not on the leaderboard). Screenshot: [`submissions/submissions.png`](submissions/submissions.png).

| Submission | Private score (dB) | Public score (dB) |
|---|---|---|
| Classical CV baseline | -4.311 | -4.073 |
| All zeros | **0.106** | 0.094 |

Our best score so far is the all-zeros baseline. The competition winners reached about 23 dB.

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

### Results (official metric)

| Method | Local validation, 30 records (dB) | Kaggle private (dB) |
|---|---|---|
| All zeros | 0.262 | 0.106 |
| Classical CV, all 9 image types | 1.016 | -4.311 |
| Classical CV, clean `0001` images only | 8.26 | – |
| *Reference, not ours:* public pretrained pipeline by hengck23 ([Kaggle notebook](https://www.kaggle.com/code/hengck23/demo-submission)) | *17.989* | – |

The classical baseline works on clean renders but is worse than zeros on most scans and photos,
because those pages are shifted, rotated or warped relative to the template. Its local average is
carried by the few clean pages, which is why it drops below zero on the hidden test.

## Assignment 2: proposed solution

Notebook: [`assignment2_solution.ipynb`](assignment2_solution.ipynb) (in progress, due 30 Dec 2026).

## How to run

**Google Colab (recommended, no GPU needed)**
1. Open the notebook in Colab:
   [Assignment 1 in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb).
2. Add a Colab secret named `GITHUB_TOKEN` (key icon in the left sidebar) with a GitHub token that
   can read this private repo, and allow the notebook to access it.
3. Run all cells. The first cell clones the repo and downloads the data subset (first run only).

**Locally**
```
pip install -r requirements.txt
jupyter notebook   # start from the repo root
```
The first cell downloads the data into `data/` if it is not there yet.

If GitHub cannot display a notebook, open it in
[nbviewer](https://nbviewer.org/github/ceng313-2026/GROUP43_CENG313/blob/main/assignment1_eda_baseline.ipynb)
or Colab.

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
