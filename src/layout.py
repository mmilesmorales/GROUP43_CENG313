"""Paper layout of the clean ECG renders (image type 0001, 2200 x 1700 px).

The numbers below were measured on our clean images and are checked in the
Assignment 1 notebook (section 5): drawing the ground truth at these positions
puts almost every sample on dark ink.
"""
import numpy as np

PX_PER_S = 196.85        # 25 mm/s at 200 dpi (1 mm = 7.874 px)
PX_PER_MV = 78.74        # 10 mm/mV
X_START = 118            # x of the first sample of each row
ZERO_Y = (709, 992, 1275, 1535)   # 0 mV line of rows 1-4 (feet of the calibration pulse)

# Rows 1-3 show four 2.5 s leads each; row 4 is lead II over the full 10 s.
ROWS = (("I", "aVR", "V1", "V4"),
        ("II", "aVL", "V2", "V5"),
        ("III", "aVF", "V3", "V6"))


def ink_mask(img_rgb: np.ndarray, dark: int = 110, red_margin: int = 20) -> np.ndarray:
    """Boolean mask of trace ink: dark pixels that are not part of the red grid."""
    im = img_rgb.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    reddish = (r > g + red_margin) & (r > b + red_margin)
    return (im.mean(axis=2) < dark) & ~reddish


def lead_positions(signal_mv: np.ndarray, fs: int, row: int, col: int):
    """Pixel (x, y) of every sample of a lead drawn at its template position.

    row 0-2 with col 0-3 for the short leads, row 3 (col 0) for the 10 s lead II.
    """
    t = np.arange(len(signal_mv)) / fs
    x = X_START + col * 2.5 * PX_PER_S + t * PX_PER_S
    y = ZERO_Y[row] - np.asarray(signal_mv) * PX_PER_MV
    return x, y


def ink_hit_rate(ink: np.ndarray, x: np.ndarray, y: np.ndarray, tol: int = 2) -> float:
    """Share of points (x, y) that have ink within +-tol px vertically."""
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    inside = (xi >= 0) & (xi < ink.shape[1]) & (yi >= 0) & (yi < ink.shape[0])
    xi, yi = xi[inside], yi[inside]
    hit = np.zeros(len(xi), bool)
    for d in range(-tol, tol + 1):
        hit |= ink[np.clip(yi + d, 0, ink.shape[0] - 1), xi]
    return float(hit.mean()) if len(hit) else float("nan")
