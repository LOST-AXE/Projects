"""
Dual-null edge term from MP2RAGE inversion images.

Method:
Two inversion times are found from run_simulation.py:
  TI_WM: WM INV1 = 0  (WM nulled, GM survives)  -> Image A
  TI_GM: GM INV1 = 0  (GM nulled, WM survives)  -> Image B

Two candidate edge terms are computed and compared:

  Method 1 - raw minimum:
    E1 = min(|A|, |B|)  then normalised

  Method 2 - weighted minimum (independent normalisation):
    A_n = robust_norm(|A|),  B_n = robust_norm(|B|)
    E2  = min(A_n, B_n)
    Gives equal weight to both nullings regardless of signal magnitude.
    This is the stronger method.

Both methods and metrics returned.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from mp2rage_simulator import MP2RAGESimulator
from tissue_library import PROTOCOLS
from run_simulation import main as get_null_times
from Utils import (get_subject_id, get_valid_mask, get_boundary_masks,
                   robust_norm, print_edge_metrics, save_figure)

# Dataset path
MAT_PATH = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
    "Child01_lsq_fit_16022024_x0_20000_1500.mat"
)


def simulate_inv1_slice(T1_slice, PDn_slice, valid_mask,
                        base_protocol, TI1_target, T2star_fixed=30.0):
    """Simulate INV1 voxel-wise on a 2D slice at a given TI1."""
    gap = base_protocol["TI2"] - base_protocol["TI1"]
    proto = base_protocol.copy()
    proto["TI1"] = float(TI1_target)
    proto["TI2"] = float(TI1_target + gap)

    sim = MP2RAGESimulator(proto, verbose=False)
    if not sim.timing_is_valid():
        raise ValueError(f"Timing invalid at TI1={TI1_target:.1f} ms")

    INV1 = np.zeros_like(T1_slice, dtype=np.float32)
    ys, zs = np.where(valid_mask)
    for y, z in zip(ys, zs):
        inv1, _ = sim.calculate_signals(
            T1=float(T1_slice[y, z]), PD=float(PDn_slice[y, z]),
            T2star=T2star_fixed, B1minus=1.0)
        INV1[y, z] = float(inv1)
    return INV1


def main(mat_path: str = MAT_PATH):

    subject_id = get_subject_id(mat_path)
    print(f"\nFLAWS dual-null  |  subject: {subject_id}")

    # Get null times
    # Close run_simulation plots to continue.
    print("Getting null times from run_simulation...")
    TI_WM, TI_GM, _, _, _ = get_null_times()
    print(f"TI_WM={TI_WM:.1f} ms, TI_GM={TI_GM:.1f} ms")

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

    # Simulate A and B
    base = PROTOCOLS["protocol_1"].copy()

    print("Simulating Image A (WM-nulled)...")
    A = simulate_inv1_slice(T1, PDn, valid, base, TI_WM)

    print("Simulating Image B (GM-nulled)...")
    B = simulate_inv1_slice(T1, PDn, valid, base, TI_GM)

    absA = np.abs(A)
    absB = np.abs(B)

    # Method 1: raw minimum
    E1 = robust_norm(np.minimum(absA, absB), valid)
    E1[~valid] = 0.0

    # Method 2: independently normalised
    A_n = robust_norm(absA, valid);  A_n[~valid] = 0.0
    B_n = robust_norm(absB, valid);  B_n[~valid] = 0.0
    E2_raw = np.minimum(A_n, B_n)
    E2 = robust_norm(E2_raw, valid)
    E2[~valid] = 0.0

    # Evaluation masks
    boundary_valid, wm_interior, gm_interior = get_boundary_masks(wm, gm, valid)

    # Metrics
    m1 = print_edge_metrics("Method 1 — raw min(|A|,|B|)",
                             E1, boundary_valid, wm_interior, gm_interior)
    m2 = print_edge_metrics("Method 2 — weighted min(A_n, B_n)",
                             E2, boundary_valid, wm_interior, gm_interior)

    # Figure (grayscale, one combined figure)
    from scipy.ndimage import binary_dilation
    boundary = ((binary_dilation(wm, iterations=2) & gm) |
                (binary_dilation(gm, iterations=2) & wm))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"FLAWS dual-null — {subject_id}\n"
        f"TI_WM={TI_WM:.0f} ms  |  TI_GM={TI_GM:.0f} ms",
        fontsize=12, fontweight="bold")

    def show(ax, img, title, signed=False):
        if signed:
            v = np.percentile(np.abs(img[valid]), 99)
            ax.imshow(img, cmap="gray", vmin=-v, vmax=v)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        #ax.contour(boundary.astype(float), levels=[0.5],
                   #colors="white", linewidths=0.5)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    show(axes[0, 0], A,   f"Image A — WM nulled\n(TI1={TI_WM:.0f} ms)", signed=True)
    show(axes[0, 1], B,   f"Image B — GM nulled\n(TI1={TI_GM:.0f} ms)", signed=True)
    show(axes[0, 2], A_n, "|A| normalised (A_n)")
    show(axes[1, 0], B_n, "|B| normalised (B_n)")
    show(axes[1, 1], E1,
         f"E1 — raw min(|A|,|B|)\nBoundary/WM = {m1['ratio']:.2f}x")
    show(axes[1, 2], E2,
         f"E2 — weighted min(A_n,B_n)  [best]\nBoundary/WM = {m2['ratio']:.2f}x")

    plt.tight_layout()
    save_figure(fig, mat_path, "flaws_dual_null")
    plt.show()

    # Return both methods
    return E1,m1, E2, m2


if __name__ == "__main__":
    main()