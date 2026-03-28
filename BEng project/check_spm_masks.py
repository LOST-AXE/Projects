"""
test GM and WM probability maps from matlab spm and check if they look sensible
before using them as ground truth in run_comparison.py.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

# Paths
DATA_DIR = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
)

gm_path  = os.path.join(DATA_DIR, "c1Child01_T1map.nii")
wm_path  = os.path.join(DATA_DIR, "c2Child01_T1map.nii")


def main():
    gm_prob = nib.load(gm_path).get_fdata().astype(np.float32)
    wm_prob = nib.load(wm_path).get_fdata().astype(np.float32)

    print(f"GM prob shape: {gm_prob.shape}")
    print(f"WM prob shape: {wm_prob.shape}")
    print(f"GM prob range: {gm_prob.min():.3f} to {gm_prob.max():.3f}")
    print(f"WM prob range: {wm_prob.min():.3f} to {wm_prob.max():.3f}")
    print(f"Voxels with GM > 0.5: {(gm_prob > 0.5).sum()}")
    print(f"Voxels with WM > 0.5: {(wm_prob > 0.5).sum()}")

    # Middle slice
    x = gm_prob.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SPM25 segmentation output — middle slice", fontweight="bold")

    axes[0].imshow(gm_prob[x, :, :], cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("GM probability (c1)")
    axes[0].axis("off")

    axes[1].imshow(wm_prob[x, :, :], cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("WM probability (c2)")
    axes[1].axis("off")

    # Boundary = voxels with both GM and WM probability > threshold
    boundary_prob = np.minimum(gm_prob, wm_prob)
    axes[2].imshow(boundary_prob[x, :, :], cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title("min(GM, WM) - boundary probability")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return gm_prob, wm_prob


if __name__ == "__main__":
    main()