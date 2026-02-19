import numpy as np
import matplotlib.pyplot as plt
from tissue_library import TISSUE_PARAMS, PROTOCOLS

# Select tissues
tissues = {
    "White Matter": TISSUE_PARAMS["white_matter_adult"],
    "Grey Matter": TISSUE_PARAMS["grey_matter_adult"],
    "CSF": TISSUE_PARAMS["csf"],
}
angle_degree = PROTOCOLS["protocol_1"]["alpha1"]
angle_radian = np.radians(angle_degree)
# Define simplified inversion recovery formula

def inversion_recovery_signal(T1, PD, TI, TR):
    """
    Calculate longitudinal magnetization using inversion recovery formula
    in slides
    Parameters:
    T1 (float): Longitudinal relaxation time in ms
    PD (float): Proton density (relative to CSF)
    TI (float): Inversion time in ms
    TR (float): Repetition time in ms

    Return:
    float: Magnetization Mz(TI)
    """
    M0 = PD  # Equilibrium magnetization proportional to PD

    # Apply the formula: Mz(t) = M0 * {1 - 2*exp(-TI/T1) + exp(-TR/T1)} * sin()
    Mz = M0*(1 - 2 * np.exp(-TI / T1) + np.exp(-TR / T1))*np.sin(angle_radian)

    return Mz

# MP2RAGE simulation parameters

# Protocol parameters
TR_MP2RAGE = 10000  # Total repetition time (ms)
TI1_values = [650 + i * 100 for i in range(94)]  # TI1 from 650 to 9950 ms
TI2 = 9000  # Second inversion time (ms) - typical value

# Calculate signals for different TI1 values

print("MP2RAGE Simulation Results")
print("=" * 60)

# Store results for plotting
results = {tissue: {'INV1': [], 'INV2': [], 'UNI': []} for tissue in
           tissues}

for TI1 in TI1_values:
    print(f"\nTI1 = {TI1} ms:")
    print("-" * 40)

    for tissue_name, params in tissues.items():
        T1 = params['T1']
        PD = params['PD']

        # Calculate INV1 signal (after first inversion)
        INV1 = inversion_recovery_signal(T1, PD, TI1, TR_MP2RAGE)

        # Store results
        results[tissue_name]['INV1'].append(INV1)

        print(
            f"{tissue_name:12} | INV1: {INV1:7.4f}")

# Plot the results

def find_best_poly_degree(x, y, max_degree=5):
    """Find best polynomial degree using simple RSS criterion"""
    best_degree = 1
    best_rss = np.inf

    for degree in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        rss = np.sum((y - y_pred) ** 2)

        if rss < best_rss:
            best_rss = rss
            best_degree = degree

    return best_degree

# Create the plot
plt.figure(figsize=(12, 8))

# Colors for different tissues
colors = {'White Matter': 'blue', 'Grey Matter': 'green', 'CSF': 'red'}

# Convert TI1_values to numpy array for fitting
TI1_array = np.array(TI1_values)

for tissue_name in tissues.keys():
    # Get the INV1 values for this tissue
    inv1_values = results[tissue_name]['INV1']

    # Find best polynomial degree
    best_degree = find_best_poly_degree(TI1_array, inv1_values)
    print(f"{tissue_name}: Best polynomial degree = {best_degree}")

    # Fit polynomial with best degree
    coefficients = np.polyfit(TI1_array, inv1_values, best_degree)
    polynomial = np.poly1d(coefficients)

    # Create smooth curve for plotting
    TI1_smooth = np.linspace(min(TI1_values), max(TI1_values), 300)
    inv1_smooth = polynomial(TI1_smooth)

    # Plot original data points
    plt.scatter(TI1_values, inv1_values, color=colors[tissue_name],
                s=80, label=f'{tissue_name} Data', alpha=0.7)

    # Plot smooth fitted curve
    plt.plot(TI1_smooth, inv1_smooth, color=colors[tissue_name],
             linewidth=2, label=f'{tissue_name} Fit (deg {best_degree})')

    # Print polynomial equation
    print(f"{tissue_name} polynomial: {polynomial}")

# Customize the plot
plt.xlabel('TI1 (ms)', fontsize=12)
plt.ylabel('INV1 Signal', fontsize=12)
plt.title('TI1 with Polynomial Fits', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

