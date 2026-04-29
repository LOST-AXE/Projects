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
                       valid: np.ndarray,
                       mat_path: str = None) -> tuple:
    """
    Build boundary, WM interior, and GM interior masks.

    If mat_path is provided and SPM25 probability maps exist alongside
    the .mat file, uses probability-based boundary definition.
    Otherwise falls back to gap-based boundary from binary masks.

    Parameters:
    wm: binary WM mask (from mask_wm in .mat file)
    gm: binary GM mask (from mask_gm in .mat file)
    valid: valid brain voxel mask (from get_valid_mask)
    mat_path: path to .mat file — used to locate SPM probability maps

    Outpiut:
    boundary_valid: boundary voxels within valid mask
    wm_interior: pure WM voxels within valid mask
    gm_interior: pure GM voxels within valid mask
    """
    # gap-based boundary from binary masks
    # The binary masks were thresholded at 0.99 in Silas's pipeline.
    # Voxels excluded from both masks are partial volume boundary voxels.

    boundary = valid & ~wm & ~gm
    if boundary.sum() > 0 and (wm.sum() > 0 or gm.sum() > 0):
        # masks exist and gap is meaningful
        print(f"  [utils] Gap boundary: {boundary.sum()} voxels")
        return boundary, wm & valid, gm & valid
    # otherwise fall through to SPM
    else:
    #if 1 == 1:
        if mat_path is None:
            raise RuntimeError(
                "No boundary masks and no mat_path provided for SPM fallback.")
        subject_id = get_subject_id(mat_path)
        data_dir = os.path.dirname(mat_path)
        gm_path = os.path.join(data_dir, f"c1{subject_id}_T1map.nii")
        wm_path = os.path.join(data_dir, f"c2{subject_id}_T1map.nii")

        if os.path.exists(gm_path) and os.path.exists(wm_path):
            try:
                import nibabel as nib

                gm_prob_3d = nib.load(gm_path).get_fdata().astype(np.float32)
                wm_prob_3d = nib.load(wm_path).get_fdata().astype(np.float32)

                # Extract middle slice (same slice as all other processing)
                x = gm_prob_3d.shape[1] // 2 - 40
                gm_prob = gm_prob_3d[:, x, :]
                wm_prob = wm_prob_3d[:, x, :]

                # Strict thresholds 0.9 ensures only confidently
                # pure tissue voxels are used as interior references.
                # A 50/50 mixed voxel has prob ~0.5 for both tissues
                # and would incorrectly pass a 0.5 threshold.
                wm_interior = (wm_prob > 0.9) & valid
                gm_interior = (gm_prob > 0.9) & valid

                # Boundary: meaningful probability of both tissues
                # but not confidently assigned to either
                boundary = valid & ~wm_interior & ~gm_interior

                print(f"  [utils] SPM boundary: {boundary.sum()} voxels  "
                      f"| WM: {wm_interior.sum()}  | GM: {gm_interior.sum()}")

                return boundary, wm_interior, gm_interior

            except Exception as e:
                raise RuntimeError(
                     f" [utils] SPM maps found but failed to load: {e}")

        else:
            raise RuntimeError(
                f"No valid boundary masks found and SPM maps not found at {gm_path}. "
                f"Run save_t1map.py then run_spm_segment.m first.")

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