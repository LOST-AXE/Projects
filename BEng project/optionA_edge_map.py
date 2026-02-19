import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation  # quick way to get a "boundary band" from WM/GM masks
from edge_filters import sobel_mag, clean_binary_edge


# import simulator + protocol
from mp2rage_simulator import MP2RAGESimulator
from tissue_library import PROTOCOLS


def robust_norm(img, mask, lo=1, hi=99):
    """Robust normalize to [0,1] based on percentiles inside mask."""
    vals = img[mask]
    if vals.size == 0:
        return np.zeros_like(img)
    vmin = np.percentile(vals, lo)
    vmax = np.percentile(vals, hi)
    if vmax <= vmin:
        return np.zeros_like(img)
    out = (img - vmin) / (vmax - vmin)
    return np.clip(out, 0, 1)


def main():
    # Load maps
    mat = sio.loadmat(
        "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/Child01_lsq_fit_16022024_x0_20000_1500.mat"
    )

    T1_3d = mat["T1_soln"].astype(np.float64)  # ms
    PD_3d = mat["PD_soln"].astype(np.float64)  # arbitrary scale (often huge)

    # Use tissue masks so we focus on GM/WM junction not CSF/background
    wm_3d = mat["mask_wm"].astype(np.uint8)
    gm_3d = mat["mask_gm"].astype(np.uint8)

    # pick slice
    x = T1_3d.shape[2] // 2
    T1 = T1_3d[x, :, :]
    PD = PD_3d[x, :, :]
    wm = wm_3d[x, :, :] > 0
    gm = gm_3d[x, :, :] > 0
    brain = wm | gm  # only WM+GM for this test

    # Avoid T1=0 / causing math issues
    T1 = np.clip(T1, 200.0, 30000.0)
    valid = brain & np.isfinite(T1) & np.isfinite(PD)

    # Normalize PD so it behaves like effective PD ~= [0, 1]
    PD_scale = np.percentile(PD[valid], 99) if np.any(valid) else np.max(PD)
    PD_scale = max(PD_scale, 1e-6)
    PDn = PD / PD_scale
    PDn = np.clip(PDn, 0.0, 2.0)

    # Set protocol at TI1* found in run_simulation.py
    base = PROTOCOLS["protocol_1"].copy()

    TI1_star = 983.8  # from sweep

    # Keep the same gap (Option A: shift TI2 with TI1)
    gap = base["TI2"] - base["TI1"]
    base["TI1"] = float(TI1_star)
    base["TI2"] = float(TI1_star + gap)

    sim = MP2RAGESimulator(base, verbose=True)
    if not sim.timing_is_valid():
        raise ValueError("Timing invalid with chosen TI1/TI2. Adjust TR/gap/n.")

    # Compute INV1 voxel-wise (single slice)
    INV1 = np.zeros_like(T1, dtype=np.float32)
    T2star_fixed = 30.0  # fine for now

    ys, zs = np.where(valid)
    for y, z in zip(ys, zs):
        inv1, _inv2 = sim.calculate_signals(
            T1=float(T1[y, z]),
            PD=float(PDn[y, z]),
            T2star=T2star_fixed,
            B1minus=1.0,
        )
        INV1[y, z] = float(inv1)

    # Option A edge map: boundary voxels should have small abs INV1
    absINV1 = np.abs(INV1)

    # Build a 1-voxel "boundary band" between WM and GM
    wm_d = binary_dilation(wm, iterations=2)
    gm_d = binary_dilation(gm, iterations=2)
    boundary = (wm_d & gm) | (gm_d & wm)

    # Normalize only within GM+WM so the dynamic range isn't dominated by other tissues
    absINV1_n = robust_norm(absINV1, valid, lo=1, hi=99)
    EdgeA = 1.0 - absINV1_n  # bright where abs INV1 is small
    EdgeA[~valid] = 0.0  # hide outside GM/WM

    # Spatial edge map from INV1 (optional post-processing for Option A)
    grad = sobel_mag(np.abs(INV1), mask=valid)
    p = np.percentile(grad[valid], 99)
    grad_s = np.clip(grad / (p + 1e-12), 0, 1)  # scaled gradient
    score = EdgeA * grad_s
    score_thresh = 0.35  # start higher since grad_s is more stable

    edgeA_spatial = clean_binary_edge(score >= score_thresh, k=3)
    edgeA_spatial[~valid] = 0

    # Plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    v = np.percentile(np.abs(INV1[valid]), 99) if np.any(valid) else 1.0
    plt.imshow(INV1, cmap="seismic", vmin=-v, vmax=v)
    plt.title("INV1 (signed) — GM/WM only")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 2)
    plt.imshow(absINV1_n, cmap="gray")
    plt.title("|INV1| (robust normalized) — GM/WM only")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(EdgeA, cmap="gray")
    plt.contour(boundary.astype(float), levels=[0.5], linewidths=0.6)  # boundary overlay
    plt.title("Edge_A = 1 - |INV1| (bright = boundary)")
    plt.axis("off")


    plt.tight_layout()
    plt.figure(figsize=(6, 6))
    plt.imshow(edgeA_spatial.astype(float), cmap="gray")
    plt.contour(boundary.astype(float), levels=[0.5], linewidths=0.6)
    plt.title(f"Option A + spatial filter (thresh={score_thresh:.2f})")
    plt.axis("off")
    plt.show()

    # Quick checks: boundary should have lower INV1 than pure WM/GM interiors
    def pct(z):
        return np.percentile(z, [5, 50, 95]) if z.size else np.array([np.nan, np.nan, np.nan])

    print("\nSanity checks (slice, GM/WM only):")
    print(f"  INV1 signed percentiles (valid): {np.percentile(INV1[valid], [1, 50, 99])}")
    print(f"  |INV1| percentiles (valid):      {np.percentile(absINV1[valid], [1, 50, 99])}")
    print(f"  |INV1| (WM):                     {pct(absINV1[wm & valid])}")
    print(f"  |INV1| (GM):                     {pct(absINV1[gm & valid])}")
    print(f"  |INV1| (boundary band):          {pct(absINV1[boundary & valid])}")
    print(f"  PD scale used (99th percentile): {PD_scale:.3f}")



if __name__ == "__main__":
    main()
