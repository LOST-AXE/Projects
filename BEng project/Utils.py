"""
Shared functions for all edge map scripts.

Functions:
robust_norm:         percentile-based normalisation to [0,1]
print_edge_metrics:  standardised terminal output for all methods
save_figure:         save figure to outputs folder with subject + method label
get_subject_id:      extract subject ID from a .mat filepath
get_boundary_masks:  build boundary, WM interior, GM interior masks
get_valid_mask:      T1-range based valid mask (no segmentation needed)
load_slice:          load middle axial slice from 3D .mat arrays
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


def get_subject_id(mat_path: str) -> str:
    """Extract subject ID from .mat filepath"""
    basename = os.path.splitext(os.path.basename(mat_path))[0]
    # e.g. "Child01_lsq_fit_16022024_..." -> "Child01"
    return basename.split("_")[0]


def get_valid_mask(T1_slice: np.ndarray, PD_slice: np.ndarray = None,
                   t1_lo: float = 400.0, t1_hi: float = 4500.0) -> np.ndarray:
    """
    Build valid brain mask from T1 range alone.
    Optionally also requires finite PD values.
    """
    mask = np.isfinite(T1_slice) & (T1_slice >= t1_lo) & (T1_slice <= t1_hi)
    if PD_slice is not None:
        mask = mask & np.isfinite(PD_slice)
    return mask


def get_boundary_masks(wm: np.ndarray, gm: np.ndarray,
                       valid: np.ndarray) -> tuple:
    """
    Build boundary, WM, GM masks.
    wm / gm are boolean 2D arrays (used for evaluation only).
    Returns (boundary_valid, wm_interior, gm_interior).
    """
    wm_d = binary_dilation(wm, iterations=2)
    gm_d = binary_dilation(gm, iterations=2)
    boundary = (wm_d & gm) | (gm_d & wm)
    return (boundary & valid,
            wm & valid & ~boundary,
            gm & valid & ~boundary)


def robust_norm(img: np.ndarray, mask: np.ndarray,
                lo: float = 1, hi: float = 99) -> np.ndarray:
    """Normalise img to [0,1] using percentiles inside mask."""
    vals = img[mask]
    if vals.size == 0:
        return np.zeros_like(img)
    vmin, vmax = np.percentile(vals, lo), np.percentile(vals, hi)
    if vmax <= vmin:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def print_edge_metrics(label: str, E: np.ndarray,
                       boundary_valid: np.ndarray,
                       wm_interior: np.ndarray,
                       gm_interior: np.ndarray) -> dict:
    """
    Print standardised edge map metrics to terminal.
    Same format across all three edge map scripts.

    Metrics printed:
    5th, median, 95th percentile for boundary/ WM/ GM
    Boundary/WM contrast ratio

    Returns dict of scalar metric values for downstream use.
    """
    def pct(arr):
        if arr.size == 0:
            return np.nan, np.nan, np.nan
        return (float(np.percentile(arr, 5)),
                float(np.median(arr)),
                float(np.percentile(arr, 95)))

    b5,  b50,  b95  = pct(E[boundary_valid])
    wm5, wm50, wm95 = pct(E[wm_interior])
    gm5, gm50, gm95 = pct(E[gm_interior])
    ratio = b50 / (wm50 + 1e-9)

    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  {'Region':<12} {'5th':>8} {'Median':>8} {'95th':>8}")
    print(f"  {'Boundary':<12} {b5:>8.4f} {b50:>8.4f} {b95:>8.4f}")
    print(f"  {'WM':<12} {wm5:>8.4f} {wm50:>8.4f} {wm95:>8.4f}")
    print(f"  {'GM':<12} {gm5:>8.4f} {gm50:>8.4f} {gm95:>8.4f}")
    print(f"  {'─'*45}")
    print(f"  Boundary / WM contrast ratio: {ratio:.2f}x")
    print(f"{'═'*55}")

    return dict(b5=b5, b50=b50, b95=b95,
                wm5=wm5, wm50=wm50, wm95=wm95,
                gm5=gm5, gm50=gm50, gm95=gm95,
                ratio=ratio)


def save_figure(fig: plt.Figure, mat_path: str,
                method: str, output_dir: str = None) -> str:
    """
    Save figure as <subject_id>_<method>.png in an /outputs folder.

    If output_dir is None, creates an 'outputs' folder in the same
    directory as the .mat file.

    Returns the full path of the saved file.
    """
    subject_id = get_subject_id(mat_path)
    filename   = f"{subject_id}_{method}.png"

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(mat_path), "outputs")

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved → {full_path}")
    return full_path