"""
flaws_dual_null.py
==================
Builds and compares two dual-null edge terms from MP2RAGE inversion images.

Pipeline
--------
1. Calls run_simulation.main() to get TI_WM and TI_GM from INV1 zero crossings

2. Loads T1/PD maps and takes the middle slice

3. Simulates INV1 voxel-wise at two inversion times:
     Image A  (TI1 = TI_WM) : WM signal = 0, GM survives
     Image B  (TI1 = TI_GM) : GM signal = 0, WM survives

4. Computes two candidate edge terms and compares them:

   Method 1 — Raw minimum:
     E_raw_min = min(|A|, |B|)
     Problem: |A| and |B| may have different dynamic ranges, so the minimum
     is dominated by whichever image has smaller overall signal magnitude.

   Method 2 — Independently normalised minimum:
     A_n = robust_norm(|A|),  B_n = robust_norm(|B|)
     E_balanced_raw = min(A_n, B_n)
     Each image is normalised to [0,1] on its own scale before combining.
     A WM-like voxel is expected to be low in A_n and relatively high in B_n,
     so the minimum remains low.
     A GM-like voxel is expected to be relatively high in A_n and low in B_n,
     so the minimum remains low.
     A boundary voxel is expected to be moderate in both, so the minimum
     remains non-zero.
     This gives both nullings a more comparable contribution regardless of
     their absolute signal magnitudes.

5. Both E terms are candidate physics-grounded edge emphasis terms for:
     I_enh = I_base + lambda * E

No segmentation masks are used to generate A, B, E1, or E2.
Masks are loaded only for visual overlay and evaluation statistics.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from mp2rage_simulator import MP2RAGESimulator
from tissue_library import PROTOCOLS
from run_simulation import main as get_null_times


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def robust_norm(img, mask, lo=1, hi=99):
    """Normalise img to [0,1] using percentiles inside mask."""
    vals = img[mask]
    if vals.size == 0:
        return np.zeros_like(img)
    vmin, vmax = np.percentile(vals, lo), np.percentile(vals, hi)
    if vmax <= vmin:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)


def simulate_inv1_slice(T1_slice, PDn_slice, valid_mask,
                        base_protocol, TI1_target,
                        T2star_fixed=30.0):
    """
    Simulate INV1 voxel-wise on a 2D slice at a given TI1.
    TI2 is shifted by the same gap as base_protocol.
    Returns signed INV1 array (same shape as T1_slice).
    """
    gap = base_protocol["TI2"] - base_protocol["TI1"]
    protocol = base_protocol.copy()
    protocol["TI1"] = float(TI1_target)
    protocol["TI2"] = float(TI1_target + gap)

    sim = MP2RAGESimulator(protocol, verbose=False)
    if not sim.timing_is_valid():
        raise ValueError(f"Timing invalid at TI1={TI1_target:.1f} ms. "
                         f"TA={sim.TA:.1f}, TB={sim.TB:.1f}, TC={sim.TC:.1f}")

    INV1 = np.zeros_like(T1_slice, dtype=np.float32)
    ys, zs = np.where(valid_mask)
    for y, z in zip(ys, zs):
        inv1, _ = sim.calculate_signals(
            T1=float(T1_slice[y, z]),
            PD=float(PDn_slice[y, z]),
            T2star=T2star_fixed,
            B1minus=1.0,
        )
        INV1[y, z] = float(inv1)
    return INV1


def print_stats(label, E, boundary_valid, wm_interior, gm_interior):
    """Print 5th/50th/95th percentiles and boundary/WM contrast ratio."""
    def pct(arr):
        return np.percentile(arr, [5, 50, 95]) if arr.size > 0 else [np.nan] * 3

    med_b = np.median(E[boundary_valid]) if np.any(boundary_valid) else np.nan
    med_wm = np.median(E[wm_interior]) if np.any(wm_interior) else np.nan
    med_gm = np.median(E[gm_interior]) if np.any(gm_interior) else np.nan

    print(f"\n── {label} ──────────────────────────────────────")
    print(f"  E (boundary): {pct(E[boundary_valid])}")
    print(f"  E (WM):       {pct(E[wm_interior])}")
    print(f"  E (GM):       {pct(E[gm_interior])}")
    print(f"  Boundary median = {med_b:.4f}")
    print(f"  WM median       = {med_wm:.4f}")
    print(f"  GM median       = {med_gm:.4f}")
    print(f"  Boundary / WM contrast ratio: {med_b / (med_wm + 1e-9):.2f}x")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # ── Step 1: get null times from run_simulation ──────────────
    # run_simulation sweeps INV1 vs TI1 using paediatric T1 values
    # and returns wm_zero[0], gm_zero[0], plus optional INV2 nulls.
    # Close the run_simulation plots to continue.
    print("Running simulation sweep to find null times...")
    TI_WM, TI_GM, _, _ = get_null_times()
    print(f"\nNull times received: TI_WM={TI_WM:.1f} ms, TI_GM={TI_GM:.1f} ms")

    # ── Step 2: load data ───────────────────────────────────────
    mat = sio.loadmat(
        "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
        "Child01_lsq_fit_16022024_x0_20000_1500.mat"
    )

    T1_3d = mat["T1_soln"].astype(np.float64)
    PD_3d = mat["PD_soln"].astype(np.float64)

    # masks loaded for boundary contour overlay and evaluation statistics only
    wm_3d = mat["mask_wm"].astype(np.uint8)
    gm_3d = mat["mask_gm"].astype(np.uint8)

    # middle axial slice
    x = T1_3d.shape[2] // 2
    T1 = T1_3d[x, :, :]
    PD = PD_3d[x, :, :]
    wm = wm_3d[x, :, :] > 0
    gm = gm_3d[x, :, :] > 0

    # ── Step 3: valid mask — T1 range only, no segmentation ────
    T1 = np.clip(T1, 200.0, 30000.0)
    valid = (
        np.isfinite(T1) &
        np.isfinite(PD) &
        (T1 >= 400.0) &
        (T1 <= 4500.0)
    )

    # ── Step 4: normalise PD ────────────────────────────────────
    PD_scale = np.percentile(PD[valid], 99) if np.any(valid) else np.max(PD)
    PD_scale = max(PD_scale, 1e-6)
    PDn = np.clip(PD / PD_scale, 0.0, 2.0)

    # ── Step 5: simulate Image A and Image B ───────────────────
    base_protocol = PROTOCOLS["protocol_1"].copy()

    print("\nSimulating Image A (WM-nulled, TI1 = TI_WM)...")
    A = simulate_inv1_slice(T1, PDn, valid, base_protocol, TI_WM)

    print("Simulating Image B (GM-nulled, TI1 = TI_GM)...")
    B = simulate_inv1_slice(T1, PDn, valid, base_protocol, TI_GM)

    absA = np.abs(A)
    absB = np.abs(B)

    # ── Step 6a: Method 1 — raw minimum ────────────────────────
    # min(|A|, |B|) on unnormalised signals.
    # Suffers if |A| and |B| have different dynamic ranges.
    E1_raw = np.minimum(absA, absB)
    E1_raw[~valid] = 0.0
    E1 = robust_norm(E1_raw, valid, lo=1, hi=99)
    E1[~valid] = 0.0

    # ── Step 6b: Method 2 — independently normalised minimum ──
    # Normalise each image independently to [0,1] first,
    # then take the minimum. This gives both nullings a more
    # comparable contribution regardless of absolute signal magnitude.
    A_n = robust_norm(absA, valid, lo=1, hi=99)
    B_n = robust_norm(absB, valid, lo=1, hi=99)
    A_n[~valid] = 0.0
    B_n[~valid] = 0.0

    E2_raw = np.minimum(A_n, B_n)
    E2_raw[~valid] = 0.0
    E2 = robust_norm(E2_raw, valid, lo=1, hi=99)
    E2[~valid] = 0.0

    # ── Boundary contour for overlay/evaluation only ───────────
    wm_d = binary_dilation(wm, iterations=2)
    gm_d = binary_dilation(gm, iterations=2)
    boundary = (wm_d & gm) | (gm_d & wm)

    boundary_valid = boundary & valid
    wm_interior = wm & valid & ~boundary
    gm_interior = gm & valid & ~boundary

    # ── Statistics ──────────────────────────────────────────────
    print_stats("Method 1: raw min(|A|, |B|)",
                E1, boundary_valid, wm_interior, gm_interior)
    print_stats("Method 2: independently normalised min(A_n, B_n)",
                E2, boundary_valid, wm_interior, gm_interior)

    # ── Plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    def overlay(ax, img, title, cmap="hot"):
        ax.imshow(img, cmap=cmap)
        ax.contour(boundary.astype(float), levels=[0.5],
                   colors="cyan", linewidths=0.5)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1: inputs
    axes[0, 0].imshow(
        A, cmap="seismic",
        vmin=-np.percentile(np.abs(A[valid]), 99),
        vmax=np.percentile(np.abs(A[valid]), 99)
    )
    axes[0, 0].contour(boundary.astype(float), levels=[0.5],
                       colors="lime", linewidths=0.5)
    axes[0, 0].set_title(f"Image A — WM nulled\n(TI1={TI_WM:.0f} ms)", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        B, cmap="seismic",
        vmin=-np.percentile(np.abs(B[valid]), 99),
        vmax=np.percentile(np.abs(B[valid]), 99)
    )
    axes[0, 1].contour(boundary.astype(float), levels=[0.5],
                       colors="lime", linewidths=0.5)
    axes[0, 1].set_title(f"Image B — GM nulled\n(TI1={TI_GM:.0f} ms)", fontsize=9)
    axes[0, 1].axis("off")

    overlay(axes[0, 2], A_n, "|A| normalised (A_n)")
    overlay(axes[0, 3], B_n, "|B| normalised (B_n)")

    # Row 2: edge terms
    overlay(axes[1, 0], E1, "Method 1: raw min(|A|,|B|)\nnormalised")
    overlay(axes[1, 1], E2, "Method 2: independently normalised\nmin(A_n, B_n)")

    diff = np.abs(E1 - E2)
    diff[~valid] = 0.0
    overlay(axes[1, 2], diff, "|E1 - E2|\n(where methods differ)")

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.05, 0.95,
        f"Method 1 (raw min)\n"
        f"  Boundary median: {np.median(E1[boundary_valid]):.3f}\n"
        f"  WM median:       {np.median(E1[wm_interior]):.3f}\n"
        f"  Ratio:           {np.median(E1[boundary_valid])/(np.median(E1[wm_interior])+1e-9):.2f}x\n\n"
        f"Method 2 (norm min)\n"
        f"  Boundary median: {np.median(E2[boundary_valid]):.3f}\n"
        f"  WM median:       {np.median(E2[wm_interior]):.3f}\n"
        f"  Ratio:           {np.median(E2[boundary_valid])/(np.median(E2[wm_interior])+1e-9):.2f}x",
        transform=axes[1, 3].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    plt.suptitle(
        f"Dual-null edge term comparison\n"
        f"TI_WM={TI_WM:.0f} ms  |  TI_GM={TI_GM:.0f} ms",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    print("\nDone. Next: run t1_notch.py for the baseline comparison.")
    print("Then combine best E with I_base (synthetic UNI) for I_enh.")

    return E1, E2, A, B, TI_WM, TI_GM


if __name__ == "__main__":
    main()