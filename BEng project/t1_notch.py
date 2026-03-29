"""
T1 notch filter edge map, physics-free baseline.

Method:
Suppress voxels whose T1 falls near the WM or GM peak,
leaving intermediate T1 values (boundary mixtures) bright.

Weight function:
    G_WM = exp(-0.5 * ((T1 - T1_WM) / sigma)^2)
    G_GM = exp(-0.5 * ((T1 - T1_GM) / sigma)^2)
    W = clip(1 - G_WM - G_GM, 0, 1) * envelope

sigma values are tested and compared.
Best sigma (highest boundary/WM ratio) is used for the saved figure.

T1_WM and T1_GM come directly from tissue_library.py
(calibrated from dataset histogram via inspect_t1map.py).
No simulation required unlike other methods.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from tissue_library import TISSUE_PARAMS
from Utils import (get_subject_id, get_valid_mask, get_boundary_masks,
                   print_edge_metrics, save_figure)

# Dataset path
MAT_PATH = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
    "Child01_lsq_fit_16022024_x0_20000_1500.mat"
)

# Sigma values to test
SIGMAS = [75, 100, 125, 150, 200]
# Larger sigma equates to larger values suppressed.

def t1_notch(T1, T1_WM, T1_GM, sigma, valid):
    """Apply Gaussian T1 notch filter. Returns W normalised to [0,1]."""
    g_wm = np.exp(-0.5 * ((T1 - T1_WM) / sigma) ** 2)
    g_gm = np.exp(-0.5 * ((T1 - T1_GM) / sigma) ** 2)
    W = np.clip(1.0 - g_wm - g_gm, 0.0, 1.0)

    # Broad envelope: suppress T1 values outside the WM-GM range
    mid = 0.5 * (T1_WM + T1_GM)
    half = 0.5 * (T1_GM - T1_WM)
    envelope = np.clip(1.0 - ((T1 - mid) / (1.5 * half)) ** 2, 0.0, 1.0)
    W = W * envelope
    W[~valid] = 0.0

    vals = W[valid]
    if vals.size > 0:
        vmax = np.percentile(vals, 99)
        if vmax > 0:
            W = np.clip(W / vmax, 0.0, 1.0)
    return W


def main(mat_path: str = MAT_PATH):

    subject_id = get_subject_id(mat_path)
    print(f"\nT1 notch filter  |  subject: {subject_id}")

    # Tissue T1 values from library (histogram-calibrated)
    T1_WM = TISSUE_PARAMS["white_matter"]["T1"]
    T1_GM = TISSUE_PARAMS["grey_matter"]["T1"]
    print(f"Notch targets: T1_WM={T1_WM:.0f} ms, T1_GM={T1_GM:.0f} ms")

    # Load data
    mat   = sio.loadmat(mat_path)
    T1_3d = mat["T1_soln"].astype(np.float64)
    x  = T1_3d.shape[2] // 2
    T1 = np.clip(T1_3d[x, :, :], 200.0, 30000.0)
    if "mask_wm" in mat and "mask_gm" in mat:
        wm = mat["mask_wm"].astype(np.uint8)[x, :, :] > 0
        gm = mat["mask_gm"].astype(np.uint8)[x, :, :] > 0
    else:
        wm = np.zeros(T1.shape, dtype=bool)
        gm = np.zeros(T1.shape, dtype=bool)
        print("  No WM/GM masks in file — will use SPM probability maps.")

    valid = get_valid_mask(T1)

    # Evaluation masks
    boundary_valid, wm_interior, gm_interior = get_boundary_masks(wm, gm,
                                                                  valid,
                                                                  mat_path)

    # Run all sigmas
    results  = {}
    metrics  = {}
    best_sigma = None
    best_ratio = -np.inf

    for sigma in SIGMAS:
        W = t1_notch(T1, T1_WM, T1_GM, sigma, valid)
        results[sigma] = W
        m = print_edge_metrics(f"T1 notch  sigma={sigma} ms",
                               W, boundary_valid, wm_interior, gm_interior)
        metrics[sigma] = m
        if m["ratio"] > best_ratio:
            best_ratio = m["ratio"]
            best_sigma = sigma

    print(f"\n  Best sigma = {best_sigma} ms  (Boundary/WM = {best_ratio:.2f}x)")

    # Figure
    from scipy.ndimage import binary_dilation
    boundary = ((binary_dilation(wm, iterations=2) & gm) |
                (binary_dilation(gm, iterations=2) & wm))

    n_cols = len(SIGMAS) + 1           # T1 map + one panel per sigma
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    fig.suptitle(
        f"T1 notch filter — {subject_id}\n"
        f"T1_WM={T1_WM:.0f} ms  |  T1_GM={T1_GM:.0f} ms",
        fontsize=12, fontweight="bold")

    # T1 map reference
    im = axes[0].imshow(T1, cmap="gray", vmin=400, vmax=2500)
    #axes[0].contour(boundary.astype(float), levels=[0.5],
                    #colors="white", linewidths=0.5)
    axes[0].set_title("T1 map (reference)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="T1 (ms)")

    for ax, sigma in zip(axes[1:], SIGMAS):
        W     = results[sigma]
        ratio = metrics[sigma]["ratio"]
        star  = " best" if sigma == best_sigma else ""
        ax.imshow(W, cmap="gray", vmin=0, vmax=1)
        #ax.contour(boundary.astype(float), levels=[0.5],
                   #colors="white", linewidths=0.5)
        ax.set_title(f"sigma={sigma} ms{star}\nBoundary/WM = {ratio:.2f}x",
                     fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, mat_path, "t1_notch")
    plt.show()

    # Return best result
    return results, metrics, best_sigma


if __name__ == "__main__":
    main()