import nibabel as nib
import numpy as np

img = nib.load("C:/Users/jiges/Downloads/outputs_3d/RICE092_3d_optionA.nii.gz")
print("Shape:", img.shape)
print("Affine:", img.affine)
data = img.get_fdata()
print("Nonzero voxels:", np.count_nonzero(data))
print("Data range:", data.min(), data.max())