"""
Option A: physics-based edge map using INV1 signal cancellation.

Method:
At TI1* (where WM and GM INV1 signals are equal and opposite),
boundary voxels with mixed tissue composition produce a small |INV1|.

Edge term:
    EdgeA = 1 - |INV1_normalised|

Combined with a Sobel spatial gradient:
    score = EdgeA * grad
    threshold(score) -> binary edge

This was the original approach before the dual-null and notch methods.
It is kept as a baseline comparator.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from mp2rage_simulator import MP2RAGESimulator
from tissue_library import PROTOCOLS
from run_simulation import main as get_null_times
from edge_filters import sobel_mag
from Utils import (get_subject_id, get_valid_mask, get_boundary_masks,
                   robust_norm, print_edge_metrics, save_figure)

# Dataset path:
MAT_PATH = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
    "Child01_lsq_fit_16022024_x0_20000_1500.mat"
)

def main(mat_path: str = MAT_PATH):

    subject_id = get_subject_id(mat_path)
    print(f"\nOption A edge map  |  subject: {subject_id}")

    # Get TI1* from run_simulation
    # TI1* is where WM + GM INV1 = 0 (equal and opposite).
    # Close the run_simulation plots to continue.
    print("Getting TI1* from run_simulation...")
    TI_WM, _, _, _, TI1_star = get_null_times()
    print(f"TI1* = {TI1_star:.1f} ms")

    # Load data
    mat   = sio.loadmat(mat_path)
    T1_3d = mat["T1_soln"].astype(np.float64)
    PD_3d = mat["PD_soln"].astype(np.float64)
    wm_3d = mat["mask_wm"].astype(np.uint8)
    gm_3d = mat["mask_gm"].astype(np.uint8)

    x  = T1_3d.shape[2] // 2
    T1 = np.clip(T1_3d[x, :, :], 200.0, 30000.0)
    PD = PD_3d[x, :, :]
    wm = wm_3d[x, :, :] > 0
    gm = gm_3d[x, :, :] > 0

    valid = get_valid_mask(T1, PD)

    PD_scale = np.percentile(PD[valid], 99) if np.any(valid) else np.max(PD)
    PDn      = np.clip(PD / max(PD_scale, 1e-6), 0.0, 2.0)

    # Simulate INV1 at TI1*
    base = PROTOCOLS["protocol_1"].copy()
    gap  = base["TI2"] - base["TI1"]
    proto = base.copy()
    proto["TI1"] = float(TI1_star)
    proto["TI2"] = float(TI1_star + gap)

    sim = MP2RAGESimulator(proto, verbose=False)
    if not sim.timing_is_valid():
        raise ValueError(f"Timing invalid at TI1={TI1_star:.1f} ms")

    INV1 = np.zeros_like(T1, dtype=np.float32)
    ys, zs = np.where(valid)
    for y, z in zip(ys, zs):
        inv1, _ = sim.calculate_signals(
            T1=float(T1[y, z]), PD=float(PDn[y, z]),
            T2star=30.0, B1minus=1.0)
        INV1[y, z] = float(inv1)

    # Edge term
    absINV1   = np.abs(INV1)
    absINV1_n = robust_norm(absINV1, valid)
    EdgeA     = 1.0 - absINV1_n
    EdgeA[~valid] = 0.0

    # Sobel spatial gradient on |INV1|
    grad   = sobel_mag(absINV1, mask=valid)
    grad_s = np.clip(grad / (np.percentile(grad[valid], 99) + 1e-12), 0, 1)
    score  = EdgeA * grad_s
    score[~valid] = 0.0



    # Evaluation masks
    boundary_valid, wm_interior, gm_interior = get_boundary_masks(wm, gm, valid)

    # Metrics
    metrics = print_edge_metrics(
        f"Option A  (TI1*={TI1_star:.0f} ms)",
        score, boundary_valid, wm_interior, gm_interior)

    # Figure
    boundary = binary_dilation(wm, iterations=2) & gm | \
               binary_dilation(gm, iterations=2) & wm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Option A — {subject_id}  |  TI1*={TI1_star:.0f} ms",
                 fontsize=12, fontweight="bold")

    axes[0].imshow(absINV1_n, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(boundary.astype(float), levels=[0.5],
                    colors="white", linewidths=0.5)
    axes[0].set_title("|INV1| normalised")
    axes[0].axis("off")

    axes[1].imshow(EdgeA, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(boundary.astype(float), levels=[0.5],
                    colors="white", linewidths=0.5)
    axes[1].set_title("EdgeA = 1 - |INV1|")
    axes[1].axis("off")

    axes[2].imshow(score, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(boundary.astype(float), levels=[0.5],
                    colors="white", linewidths=0.5)
    axes[2].set_title(f"Score = EdgeA × Sobel\n"
                      f"Boundary/WM = {metrics['ratio']:.2f}x")
    axes[2].axis("off")

    plt.tight_layout()
    save_figure(fig, mat_path, "optionA")
    plt.show()

    return score, metrics


if __name__ == "__main__":
    main()