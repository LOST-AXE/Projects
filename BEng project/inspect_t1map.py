"""
inspect_t1map.py
----------------
Inspect the T1/PD maps and estimate representative WM and GM tissue T1 values
from the T1 histogram

The T1 histogram of brain tissue shows two dominant peaks:
  - WM peak  (shorter T1, lower ms)
  - GM peak  (longer T1, higher ms)

We find these peaks and use their T1 values to calibrate tissue_library.py
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Histogram-based tissue T1 estimator

def estimate_tissue_T1s(T1_map: np.ndarray,
                        t1_min: float = 100.0,
                        t1_max: float = 3000.0,
                        n_bins: int = 500,
                        smooth_sigma: float = 3.0) -> dict:
    """
    Estimate representative WM and GM T1 values from the T1 histogram.

    Strategy:
    1. Restrict to the physiological T1 range for brain tissue [t1_min, t1_max].
       (Excludes background, CSF, and fitting artefacts.)
    2. Build a histogram and smooth it.
    3. Find the two dominant peaks; the lower-T1 peak is WM, the higher is GM.
    4. Return their T1 values as the representative tissue parameters.

    Parameters:
    T1_map: 2D or 3D T1 map in ms
    t1_min: lower bound for brain tissue T1 search (excludes background)
    t1_max: upper bound (excludes CSF ~4000ms and artefacts)
    n_bins: histogram resolution
    smooth_sigma: Gaussian smoothing width (bins) to reduce noise peaks

    Returns:
    dict with keys:
        T1_WM: float - WM representative T1 (ms)
        T1_GM: float - GM representative T1 (ms)
        bin_centres: array - T1 axis of histogram
        hist_smooth: array - smoothed histogram
        wm_peak_idx: int - bin index of WM peak
        gm_peak_idx: int - bin index of GM peak
    """
    # Flatten and restrict to physiological range
    vals = T1_map.ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= t1_min) & (vals <= t1_max)]

    if vals.size == 0:
        raise ValueError(f"No valid T1 values in range [{t1_min}, {t1_max}] ms. "
                         "Check t1_min/t1_max or your T1 map.")

    # Build histogram
    counts, edges = np.histogram(vals, bins=n_bins, range=(t1_min, t1_max))
    bin_centres = (edges[:-1] + edges[1:]) / 2.0

    # Smooth to suppress noise peaks
    hist_smooth = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)

    # Find peaks - require minimum prominence and separation
    bin_width = bin_centres[1] - bin_centres[0]
    min_separation_bins = max(1, int(100.0 / bin_width))  # peaks must be >100ms apart

    peaks, props = find_peaks(
        hist_smooth,
        prominence=hist_smooth.max() * 0.05,
        distance=min_separation_bins
    )

    def pick_two_peaks_fallback(h, min_sep_bins):
        """
        Robust fallback if find_peaks returns <2 peaks:
        - pick global maximum
        - suppress a neighborhood around it
        - pick next maximum
        """
        h = h.copy()
        i1 = int(np.argmax(h))
        lo = max(0, i1 - min_sep_bins)
        hi = min(h.size, i1 + min_sep_bins + 1)
        h[lo:hi] = -np.inf  # suppress neighborhood
        i2 = int(np.argmax(h))
        if not np.isfinite(h[i2]):
            raise RuntimeError("Fallback peak picking failed: could not find second peak.")
        return np.sort([i1, i2])

    if len(peaks) >= 2:
        # choose two most prominent peaks (by height here; you can use props["prominences"] too)
        top2 = np.argsort(hist_smooth[peaks])[::-1][:2]
        peaks = np.sort(peaks[top2])
    else:
        peaks = pick_two_peaks_fallback(hist_smooth, min_separation_bins)

    wm_peak_idx = int(peaks[0])
    gm_peak_idx = int(peaks[1])

    T1_WM = float(bin_centres[wm_peak_idx])
    T1_GM = float(bin_centres[gm_peak_idx])

    print("\n-- Histogram-based tissue T1 estimation (mask-free) --")
    print(f"  WM peak T1 = {T1_WM:.1f} ms")
    print(f"  GM peak T1 = {T1_GM:.1f} ms")
    print(f"  (These will be used to compute TI_WM and TI_GM for dual-null simulation)")

    return dict(T1_WM=T1_WM, T1_GM=T1_GM,
                bin_centres=bin_centres, hist_smooth=hist_smooth,
                wm_peak_idx=wm_peak_idx, gm_peak_idx=gm_peak_idx)


#  main
def main():
    mat = sio.loadmat(
        "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
        "Child01_lsq_fit_16022024_x0_20000_1500.mat"
    )

    # Print all variables
    print("Variables in file:")
    for k in mat.keys():
        if k.startswith("__"):
            continue
        arr = mat[k]
        if hasattr(arr, "shape"):
            try:
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}")
            except Exception:
                print(f"  {k}: shape={arr.shape}")

    # Load T1 map
    if "T1_soln" not in mat:
        print("T1_soln not found in file.")
        return

    T1_full = mat["T1_soln"].astype(np.float64)
    print(f"\nLoaded T1_soln: shape={T1_full.shape}")

    # Middle slice for visualisation
    if T1_full.ndim == 3:
        mid = T1_full.shape[2] // 2
        T1_slice = T1_full[mid, :, :]
    else:
        T1_slice = T1_full

    # Estimate WM/GM T1s from histogram
    tissue_T1s = estimate_tissue_T1s(
        T1_full,          # use full 3D volume for robust statistics
        t1_min=100.0,
        t1_max=3000.0,
        n_bins=500,
        smooth_sigma=3.0
    )

    # Plot 1: T1 map slice
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(T1_slice, cmap="gray",
                        vmin=400, vmax=2500)
    axes[0].set_title("T1 map — middle slice")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], label="T1 (ms)", fraction=0.046, pad=0.04)

    # Plot 2: T1 histogram with detected peaks
    bc  = tissue_T1s["bin_centres"]
    hs  = tissue_T1s["hist_smooth"]
    wmi = tissue_T1s["wm_peak_idx"]
    gmi = tissue_T1s["gm_peak_idx"]

    axes[1].plot(bc, hs, "k", lw=1.5, label="T1 histogram (smoothed)")
    axes[1].axvline(tissue_T1s["T1_WM"], color="blue",  ls="--", lw=1.5,
                    label=f"WM peak = {tissue_T1s['T1_WM']:.0f} ms")
    axes[1].axvline(tissue_T1s["T1_GM"], color="darkgreen", ls="--", lw=1.5,
                    label=f"GM peak = {tissue_T1s['T1_GM']:.0f} ms")
    axes[1].scatter([bc[wmi], bc[gmi]], [hs[wmi], hs[gmi]],
                    color=["blue", "darkgreen"], zorder=5, s=60)
    axes[1].set_xlabel("T1 (ms)")
    axes[1].set_ylabel("Voxel count")
    axes[1].set_title("T1 histogram — WM and GM peaks\n(used for mask-free tissue calibration)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(100, 3000)

    plt.tight_layout()
    plt.show()

    print("Code ran succesfully")

    return tissue_T1s


if __name__ == "__main__":
    main()