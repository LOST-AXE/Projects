"""
Runs all three edge map methods on the full 3D volume of a subject
and saves the results as NIfTI .nii.gz files.
Notes:
- get_null_times() is called once at the start. The run_simulation
  plots are suppressed.
- All slice-level logic is copied directly from the 2D method files.
  Nothing has been changed in the core functions
"""

import os
import time
import numpy as np
import scipy.io as sio

# Suppress matplotlib plots from run_simulation before importing it
import matplotlib
matplotlib.use("Agg")

from mp2rage_simulator import MP2RAGESimulator
from tissue_library import PROTOCOLS, TISSUE_PARAMS
from run_simulation import main as get_null_times
from edge_filters import sobel_mag
from Utils import get_subject_id, get_valid_mask, robust_norm

import nibabel as nib

MAT_PATH = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
    "Adult05_lsq_fit_16022024_x0_20000_1500"
)

SIGMAS = [75, 100, 125, 150, 200]


# Slice-level functions (copied directly from 2D files, unchanged)

def _simulate_inv1_slice(T1_slice, PDn_slice, valid_mask,
                         base_protocol, TI1_target, T2star_fixed=30.0):
    """Copied verbatim from flaws_dual_null.py simulate_inv1_slice."""
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
            T1=float(T1_slice[y, z]),
            PD=float(PDn_slice[y, z]),
            T2star=T2star_fixed,
            B1minus=1.0)
        INV1[y, z] = float(inv1)
    return INV1


def _optionA_slice(T1, PDn, valid, base, TI1_star):
    """Copied verbatim from optionA_edge_map.py core logic."""
    INV1    = _simulate_inv1_slice(T1, PDn, valid, base, TI1_star)
    absINV1 = np.abs(INV1)
    EdgeA   = 1.0 - robust_norm(absINV1, valid)
    EdgeA[~valid] = 0.0
    grad    = sobel_mag(absINV1, mask=valid)
    grad_s  = np.clip(grad / (np.percentile(grad[valid], 99) + 1e-12), 0, 1) \
              if np.any(valid) else grad
    score   = EdgeA * grad_s
    score[~valid] = 0.0
    return score


def _flaws_E2_slice(T1, PDn, valid, base, TI_WM, TI_GM):
    """Copied verbatim from flaws_dual_null.py Method 2 logic."""
    A    = _simulate_inv1_slice(T1, PDn, valid, base, TI_WM)
    B    = _simulate_inv1_slice(T1, PDn, valid, base, TI_GM)
    A_n  = robust_norm(np.abs(A), valid);  A_n[~valid] = 0.0
    B_n  = robust_norm(np.abs(B), valid);  B_n[~valid] = 0.0
    E2   = robust_norm(np.minimum(A_n, B_n), valid)
    E2[~valid] = 0.0
    return E2


def _notch_slice(T1, valid, T1_WM, T1_GM, sigma):
    """Copied verbatim from t1_notch.py t1_notch function."""
    g_wm = np.exp(-0.5 * ((T1 - T1_WM) / sigma) ** 2)
    g_gm = np.exp(-0.5 * ((T1 - T1_GM) / sigma) ** 2)
    W    = np.clip(1.0 - g_wm - g_gm, 0.0, 1.0)
    mid  = 0.5 * (T1_WM + T1_GM)
    half = 0.5 * (T1_GM - T1_WM)
    envelope = np.clip(1.0 - ((T1 - mid) / (1.5 * half)) ** 2, 0.0, 1.0)
    W    = W * envelope
    W[~valid] = 0.0
    vals = W[valid]
    if vals.size > 0:
        vmax = np.percentile(vals, 99)
        if vmax > 0:
            W = np.clip(W / vmax, 0.0, 1.0)
    return W


# Save helper

def _save_nifti(volume, output_path):
    """Save a 3D numpy array as NIfTI .nii.gz."""
    img = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(img, output_path)
    print(f"  Saved → {output_path}")


# Main

def main(mat_path=MAT_PATH):

    subject_id = get_subject_id(mat_path)
    print(f"\n{'-----'}")
    print(f"  3D processing  |  subject: {subject_id}")
    print(f"{'-----'}")

    # Output directory
    out_dir = os.path.join(os.path.dirname(mat_path), "outputs_3d")
    os.makedirs(out_dir, exist_ok=True)

    # Get null times once
    # run_simulation plots are suppressed.
    print("\nGetting null times from run_simulation (plots suppressed)")
    TI_WM, TI_GM, _, _, TI1_star = get_null_times()
    print(f"  TI1*  = {TI1_star:.1f} ms  (Option A)")
    print(f"  TI_WM = {TI_WM:.1f} ms  (FLAWS Image A)")
    print(f"  TI_GM = {TI_GM:.1f} ms  (FLAWS Image B)")

    # Load full 3D data
    print("\nLoading data")
    mat    = sio.loadmat(mat_path)
    T1_3d  = mat["T1_soln"].astype(np.float64)
    PD_3d  = mat["PD_soln"].astype(np.float64)
    n_slices = T1_3d.shape[2]
    print(f"  Volume shape: {T1_3d.shape}  ({n_slices} slices)")

    # Notch tissue T1 targets
    T1_WM_notch = TISSUE_PARAMS["white_matter"]["T1"]
    T1_GM_notch = TISSUE_PARAMS["grey_matter"]["T1"]

    base = PROTOCOLS["protocol_1"].copy()

    # Preallocate output volumes
    vol_A  = np.zeros(T1_3d.shape, dtype=np.float32)
    vol_F  = np.zeros(T1_3d.shape, dtype=np.float32)
    vol_N  = {sigma: np.zeros(T1_3d.shape, dtype=np.float32)
              for sigma in SIGMAS}

    # Slice loop
    print(f"\nProcessing {n_slices} slices...")
    t_start = time.time()

    for x in range(n_slices):

        T1  = np.clip(T1_3d[:, :, x], 200.0, 30000.0)
        PD  = PD_3d[:, :, x]

        valid    = get_valid_mask(T1, PD)

        # Skip empty slices (all background)
        if not np.any(valid):
            continue

        PD_scale = np.percentile(PD[valid], 99)
        PD_scale = max(PD_scale, 1e-6)
        PDn      = np.clip(PD / PD_scale, 0.0, 2.0)

        # Option A
        vol_A[:, :, x] = _optionA_slice(T1, PDn, valid, base, TI1_star)

        # FLAWS E2
        vol_F[:, :, x] = _flaws_E2_slice(T1, PDn, valid, base, TI_WM, TI_GM)

        # T1 notch :all sigmas
        for sigma in SIGMAS:
            vol_N[sigma][:, :, x] = _notch_slice(T1, valid,
                                                   T1_WM_notch, T1_GM_notch,
                                                   sigma)

        # Progress + estimated time remaining
        elapsed   = time.time() - t_start
        per_slice = elapsed / (x + 1)
        remaining = per_slice * (n_slices - x - 1)
        print(f"  Slice {x+1:>3}/{n_slices}  |  "
              f"elapsed {elapsed:>6.1f}s  |  "
              f"remaining ~{remaining:>6.1f}s",
              end="\r")

    print(f"\n  Done in {time.time()-t_start:.1f}s")

    # Save
    print("\nSaving NIfTI volumes")
    _save_nifti(vol_A,
                os.path.join(out_dir, f"{subject_id}_3d_optionA.nii.gz"))
    _save_nifti(vol_F,
                os.path.join(out_dir, f"{subject_id}_3d_flaws_E2.nii.gz"))
    for sigma in SIGMAS:
        _save_nifti(vol_N[sigma],
                    os.path.join(out_dir,
                                 f"{subject_id}_3d_notch_s{sigma}.nii.gz"))

    print(f"\nAll outputs saved to: {out_dir}")
    return vol_A, vol_F, vol_N


if __name__ == "__main__":
    main()