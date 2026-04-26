import scipy.io as sio

# Load the RICE mat file
mat = sio.loadmat(
    "C:/Users/jiges/Downloads/T1_Fit_Results_RICE092_s1_256_BETmask_B1.mat"
)

# Print keys to see what's in there
print("Keys:", [k for k in mat.keys() if not k.startswith('__')])

# Fix any hyphens in key names
fixed = {}
for k, v in mat.items():
    new_key = k.replace('-', '_')
    fixed[new_key] = v

# Save fixed version
sio.savemat(
    "C:/Users/jiges/Downloads/RICE092_fixed.mat",
    fixed
)
print("Saved fixed file.")