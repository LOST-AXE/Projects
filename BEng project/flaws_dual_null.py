"""
flaws_dual_null.py
==================
Builds the dual-null edge term E from two MP2RAGE inversion images.

Pipeline
--------
1. Calls run_simulation.main() to get TI_WM and TI_GM from INV1 zero crossings
   (computed using paediatric T1 values from tissue_library.py)

2. Loads T1/PD maps and takes the middle slice

3. Simulates INV1 voxel-wise at two inversion times:
     Image A  (TI1 = TI_WM) : WM signal = 0, GM survives
     Image B  (TI1 = TI_GM) : GM signal = 0, WM survives

4. Computes edge term:
     E = min(A_norm, B_norm)
   where A_norm and B_norm are robust-normalized versions of |A| and |B|.

5. This E is the physics-grounded edge emphasis term for:
     I_enh = I_base + lambda * E

No segmentation masks are used to generate A, B, or E.
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
    Returns INV1 array (same shape as T1_slice).
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


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():

    # ── Step 1: get null times from run_simulation ──────────────
    # run_simulation sweeps INV1 vs TI1 using paediatric T1 values
    # and returns wm_zero[0], gm_zero[0] — the individual tissue null times.
    # Note: this will display the run_simulation plots. Close them to continue.
    print("Running simulation sweep to find null times...")
    TI_WM, TI_GM = get_null_times()
    print(f"\nNull times received: TI_WM={TI_WM:.1f} ms, TI_GM={TI_GM:.1f} ms")

    # ── Step 2: load data ───────────────────────────────────────
    mat = sio.loadmat(
        "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
        "Child01_lsq_fit_16022024_x0_20000_1500.mat"
    )

    T1_3d = mat["T1_soln"].astype(np.float64)
    PD_3d = mat["PD_soln"].astype(np.float64)

    # masks loaded for boundary contour overlay and evaluation stats only
    wm_3d = mat["mask_wm"].astype(np.uint8)
    gm_3d = mat["mask_gm"].astype(np.uint8)

    # middle axial slice
    x  = T1_3d.shape[2] // 2
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

    # ── Step 6: dual-null edge term E = min(A_norm, B_norm) ────
    A_n = robust_norm(np.abs(A), valid, lo=1, hi=99)
    B_n = robust_norm(np.abs(B), valid, lo=1, hi=99)

    E_raw = A_n * B_n
    E_raw[~valid] = 0.0

    E = robust_norm(E_raw, valid, lo=1, hi=99)
    E[~valid] = 0.0

    # ── Boundary contour for overlay and evaluation only ────────
    wm_d = binary_dilation(wm, iterations=2)
    gm_d = binary_dilation(gm, iterations=2)
    boundary = (wm_d & gm) | (gm_d & wm)

    # ── Statistics ──────────────────────────────────────────────
    def pct(arr):
        return np.percentile(arr, [5, 50, 95]) if arr.size > 0 else [np.nan] * 3

    boundary_valid = boundary & valid
    wm_interior = wm & valid & ~boundary
    gm_interior = gm & valid & ~boundary

    print("\n── E (edge term) statistics ──────────────────────────")
    print(f"  E (boundary band): {pct(E[boundary_valid])}")
    print(f"  E (WM interior):   {pct(E[wm_interior])}")
    print(f"  E (GM interior):   {pct(E[gm_interior])}")

    med_boundary = np.median(E[boundary_valid]) if np.any(boundary_valid) else np.nan
    med_wm = np.median(E[wm_interior]) if np.any(wm_interior) else np.nan
    print(f"  Boundary median / WM median contrast ratio: {med_boundary / (med_wm + 1e-9):.2f}x")

    # ── Plots ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    vA = np.percentile(np.abs(A[valid]), 99)
    axes[0].imshow(A, cmap="seismic", vmin=-vA, vmax=vA)
    axes[0].contour(boundary.astype(float), levels=[0.5],
                    colors="lime", linewidths=0.5)
    axes[0].set_title(f"Image A — WM nulled\n(TI1 = {TI_WM:.0f} ms)")
    axes[0].axis("off")

    vB = np.percentile(np.abs(B[valid]), 99)
    axes[1].imshow(B, cmap="seismic", vmin=-vB, vmax=vB)
    axes[1].contour(boundary.astype(float), levels=[0.5],
                    colors="lime", linewidths=0.5)
    axes[1].set_title(f"Image B — GM nulled\n(TI1 = {TI_GM:.0f} ms)")
    axes[1].axis("off")

    axes[2].imshow(E_raw, cmap="hot")
    axes[2].contour(boundary.astype(float), levels=[0.5],
                    colors="cyan", linewidths=0.5)
    axes[2].set_title("E_raw = min(A_norm, B_norm)\n(boundary survives)")
    axes[2].axis("off")

    axes[3].imshow(E, cmap="hot")
    axes[3].contour(boundary.astype(float), levels=[0.5],
                    colors="cyan", linewidths=0.5)
    axes[3].set_title("E normalised [0,1]\nedge term ready for I_enh")
    axes[3].axis("off")

    plt.suptitle(
        f"FLAWS dual-null edge term E\n"
        f"TI_WM={TI_WM:.0f} ms (WM=0)   |   TI_GM={TI_GM:.0f} ms (GM=0)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    print("\nDone. E is ready.")
    print("Next step: build I_base (synthetic UNI from INV1 x INV2) and compute:")
    print("  I_enh = normalise(I_base) + lambda * E")

    return E, A, B, TI_WM, TI_GM


if __name__ == "__main__":
    main()