"""
t1_notch.py
===========
Simple T1-notch baseline.

Idea:
Suppress voxels near the WM and GM T1 peaks, leaving intermediate T1 values
(boundary mixtures) brighter.

This is a physics-free baseline to compare against flaws_dual_null.py.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from tissue_library import TISSUE_PARAMS


def t1_notch(T1, T1_WM, T1_GM, sigma, valid):
    """Simple symmetric T1 notch filter."""
    g_wm = np.exp(-0.5 * ((T1 - T1_WM) / sigma) ** 2)
    g_gm = np.exp(-0.5 * ((T1 - T1_GM) / sigma) ** 2)

    w = np.clip(1.0 - g_wm - g_gm, 0.0, 1.0)

    # suppress values far outside the WM–GM range
    mid = 0.5 * (T1_WM + T1_GM)
    half = 0.5 * (T1_GM - T1_WM)
    envelope = np.clip(1.0 - ((T1 - mid) / (1.5 * half)) ** 2, 0.0, 1.0)

    w = w * envelope
    w[~valid] = 0.0

    vals = w[valid]
    if vals.size > 0:
        vmax = np.percentile(vals, 99)
        if vmax > 0:
            w = np.clip(w / vmax, 0.0, 1.0)

    return w


def print_stats(label, W, boundary_valid, wm_interior, gm_interior):
    def pct(arr):
        return np.percentile(arr, [5, 50, 95]) if arr.size > 0 else [np.nan] * 3

    med_b = np.median(W[boundary_valid]) if np.any(boundary_valid) else np.nan
    med_wm = np.median(W[wm_interior]) if np.any(wm_interior) else np.nan
    med_gm = np.median(W[gm_interior]) if np.any(gm_interior) else np.nan

    print(f"\n── {label} ─────────────────────────")
    print(f"  W (boundary): {pct(W[boundary_valid])}")
    print(f"  W (WM):       {pct(W[wm_interior])}")
    print(f"  W (GM):       {pct(W[gm_interior])}")
    print(f"  Boundary median = {med_b:.4f}")
    print(f"  WM median       = {med_wm:.4f}")
    print(f"  GM median       = {med_gm:.4f}")
    print(f"  Boundary / WM contrast ratio: {med_b / (med_wm + 1e-9):.2f}x")


def main():
    mat = sio.loadmat(
        "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
        "Child01_lsq_fit_16022024_x0_20000_1500.mat"
    )

    T1_3d = mat["T1_soln"].astype(np.float64)
    wm_3d = mat["mask_wm"].astype(np.uint8)
    gm_3d = mat["mask_gm"].astype(np.uint8)

    x = T1_3d.shape[2] // 2
    T1 = T1_3d[x, :, :]
    wm = wm_3d[x, :, :] > 0
    gm = gm_3d[x, :, :] > 0

    T1 = np.clip(T1, 200.0, 30000.0)
    valid = np.isfinite(T1) & (T1 >= 400.0) & (T1 <= 4500.0)

    T1_WM = TISSUE_PARAMS["white_matter"]["T1"]
    T1_GM = TISSUE_PARAMS["grey_matter"]["T1"]

    print(f"T1_WM = {T1_WM:.0f} ms")
    print(f"T1_GM = {T1_GM:.0f} ms")

    wm_d = binary_dilation(wm, iterations=2)
    gm_d = binary_dilation(gm, iterations=2)
    boundary = (wm_d & gm) | (gm_d & wm)

    boundary_valid = boundary & valid
    wm_interior = wm & valid & ~boundary
    gm_interior = gm & valid & ~boundary

    sigmas = [100.0, 200.0]
    results = {}

    for sigma in sigmas:
        label = f"sigma={int(sigma)} ms"
        W = t1_notch(T1, T1_WM, T1_GM, sigma, valid)
        results[label] = W
        print_stats(label, W, boundary_valid, wm_interior, gm_interior)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im = axes[0].imshow(T1, cmap="gray", vmin=400, vmax=2500)
    axes[0].contour(boundary.astype(float), levels=[0.5], colors="cyan", linewidths=0.5)
    axes[0].set_title("T1 map")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="T1 (ms)")

    for ax, (label, W) in zip(axes[1:], results.items()):
        med_b = np.median(W[boundary_valid]) if np.any(boundary_valid) else np.nan
        med_wm = np.median(W[wm_interior]) if np.any(wm_interior) else np.nan
        ratio = med_b / (med_wm + 1e-9)

        ax.imshow(W, cmap="hot", vmin=0, vmax=1)
        ax.contour(boundary.astype(float), levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"{label}\nBoundary/WM = {ratio:.2f}x")
        ax.axis("off")

    plt.suptitle(
        f"T1 notch baseline\nT1_WM={T1_WM:.0f} ms, T1_GM={T1_GM:.0f} ms",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    main()