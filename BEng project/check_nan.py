import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

c1 = nib.load("C:/Users/jiges/Downloads/c1RICE092_T1map.nii").get_fdata()
c2 = nib.load("C:/Users/jiges/Downloads/c2RICE092_T1map.nii").get_fdata()

x = 320 // 2 - 40  # = 120

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(c1[:, x, :], cmap="gray"); axes[0].set_title("c1 GM [:, x, :]")
axes[1].imshow(c2[:, x, :], cmap="gray"); axes[1].set_title("c2 WM [:, x, :]")
axes[2].imshow(c1[x, :, :], cmap="gray"); axes[2].set_title("c1 GM [x, :, :]")
axes[3].imshow(c2[x, :, :], cmap="gray"); axes[3].set_title("c2 WM [x, :, :]")
plt.show()

print("c1 max:", c1.max())
print("c2 max:", c2.max())
print("c1 > 0.9 count:", np.sum(c1 > 0.9))
print("c2 > 0.9 count:", np.sum(c2 > 0.9))