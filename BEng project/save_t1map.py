"""
Saves the T1 map from a .mat file as a NIfTI .nii.gz file
so it can be used as input to SPM segmentation.
Usage:
    python save_t1map.py
Output:
    <subject_id>_T1map.nii.gz
"""

import numpy as np
import scipy.io as sio
import nibabel as nib
import os

from nibabel.affines import voxel_sizes

from Utils import get_subject_id

MAT_PATH = (
    "C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/"
    "Child01_lsq_fit_16022024_x0_20000_1500.mat"
)


def main(mat_path=MAT_PATH):
    subject_id = get_subject_id(mat_path)
    print(f"Saving T1 map for {subject_id}...")

    mat = sio.loadmat(mat_path)
    # Use t1_stack instead of T1_soln
    t1_raw = mat["t1_stack"].astype(np.float32)
    voxel_size = 0.7
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    img = nib.Nifti1Image(t1_raw, affine=affine)
    out_path = os.path.join(os.path.dirname(mat_path),
                            f"{subject_id}_T1map.nii")
    nib.save(img, out_path)
    print(f"Saved {out_path}")
    return out_path

if __name__ == "__main__":
    main()