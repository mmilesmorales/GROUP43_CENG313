"""Dataset helpers: splits, record GT loading, lead-signal extraction.

Assignment-1 layout: the notebook imports everything from here instead of
re-implementing pieces in cells.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Team copy of the dataset subset on Google Drive (public link, ~2.5 GB zip).
DATA_DRIVE_ID = "15VNuF-QR-IiCfIl4lqVwbvO-lLehtckv"


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------

def download_dataset(dest=ROOT / "data", drive_id: str = DATA_DRIVE_ID) -> Path:
    """Download and unpack the dataset subset into `dest`.

    Does nothing if `dest/train.csv` already exists. The zip may hold the files
    at its top level or inside a `data/` folder; both layouts work.
    """
    import shutil
    import tempfile
    import zipfile

    dest = Path(dest)
    if (dest / "train.csv").exists():
        return dest

    import gdown
    zip_path = dest.parent / "data_subset.zip"
    if not zip_path.exists():
        gdown.download(id=drive_id, output=str(zip_path), quiet=False)
    if not zip_path.exists():
        raise RuntimeError("dataset download failed: check the Drive link and your connection")

    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        inner = Path(tmp) / "data" if (Path(tmp) / "data" / "train.csv").exists() else Path(tmp)
        for item in inner.iterdir():
            if not (dest / item.name).exists():
                shutil.move(str(item), str(dest / item.name))
    zip_path.unlink()
    return dest


# ----------------------------------------------------------------------------
# Record-level split (train/val), reproducible
# ----------------------------------------------------------------------------

def make_split(val_frac: float = 0.10, seed: int = 0, write: bool = True):
    """Split the 977 training records 90/10 BY RECORD (all 9 images of a record
    stay together), reproducible with seed 0."""
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr))
    n_val = int(round(val_frac * len(tr)))
    val_ids = sorted(tr.id.values[perm[:n_val]].tolist())
    train_ids = sorted(tr.id.values[perm[n_val:]].tolist())
    if write:
        for name, ids in [("split_val_ids.txt", val_ids), ("split_train_ids.txt", train_ids)]:
            (ROOT / "data" / name).write_text("\n".join(map(str, ids)))
    return train_ids, val_ids


def load_val_ids(n: int = None) -> list:
    ids = [int(x) for x in (ROOT / "data" / "split_val_ids.txt").read_text().split()]
    return ids[:n] if n else ids


# ----------------------------------------------------------------------------
# Ground truth utilities
# ----------------------------------------------------------------------------

def load_record_gt(data_dir, record_id) -> pd.DataFrame:
    """Ground-truth time series of one record: columns = 12 leads.

    NOTE (data-structure discovery): the GT csv stores 10s of rows (= lead II
    length).  The 11 short leads hold their own 2.5s series in their finite span.
    Use extract_lead_signal() for the per-lead series.
    """
    return pd.read_csv(f"{data_dir}/train/{record_id}/{record_id}.csv")


def extract_lead_signal(gt: pd.DataFrame, lead: str) -> np.ndarray:
    """A lead's own contiguous 2.5 s series (10 s for II), from its finite span."""
    v = gt[lead].values
    finite = np.flatnonzero(np.isfinite(v))
    if len(finite) == 0:
        return np.zeros(0)
    runs = np.split(finite, np.where(np.diff(finite) != 1)[0] + 1)
    run = max(runs, key=len)          # longest contiguous finite run
    return v[run[0]:run[-1] + 1].astype(np.float64)


def fs_of_record(train_csv: pd.DataFrame, record_id) -> int:
    row = train_csv.loc[train_csv.id == record_id]
    return int(row.fs.iloc[0])
