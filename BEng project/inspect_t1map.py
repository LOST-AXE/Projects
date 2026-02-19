import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


mat = sio.loadmat("C:/Users/jiges/Downloads/Example_T1_data/Example_T1_data/Child01_lsq_fit_16022024_x0_20000_1500.mat")

# List keys
keys = []
for k in mat.keys():
    if not k.startswith("__"):  # Skip metadata keys that start with __
        keys.append(k)

print("Variables in file:")
for k in keys:
    arr = mat[k]
    if hasattr(arr, "shape"):
        try:
            print(f"{k}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}")
        except:
            print(f"{k}: shape={arr.shape}")

# Try to visualise T1_soln
if "T1_soln" in mat:
    T1 = mat["T1_soln"]
    print("\nLoaded T1_soln")

    if T1.ndim == 3:
        mid = T1.shape[2] // 2
        plt.imshow(T1[:, :, mid], cmap="gray")
        plt.title("Middle slice of T1_soln")
        plt.colorbar(label="T1 (ms)")
        plt.show()

    elif T1.ndim == 2:
        plt.imshow(T1, cmap="gray")
        plt.title("T1_soln (2D)")
        plt.colorbar(label="T1 (ms)")
        plt.show()

else:
    print("T1_soln not found in file.")