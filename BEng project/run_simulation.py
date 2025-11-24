import numpy as np
import matplotlib.pyplot as plt
from tissue_library import TISSUE_PARAMS, PROTOCOLS
from mp2rage_simulator import MP2RAGESimulator

# Create a figure for all iterations, 10 inches wide by 5 inches tall
plt.figure(figsize=(10, 5))

# Create a dictionary that maps tissue names to color codes
colors = {"White Matter": "k", "Grey Matter": "r", "CSF": "b"}
# Create a list of different marker shapes for each iteration
markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'l']

# Run simulation 10 times with increasing TI1
for iteration in range(10):  # run ten times from 0 to 9
    # Create a copy of the protocol and modify TI1
    protocol = PROTOCOLS['protocol_1'].copy()
    protocol['TI1'] = 650 + iteration * 100  # Set to base + increment

    # Initialize simulator with modified protocol
    sim = MP2RAGESimulator(protocol)

    # Select tissues
    tissues = {
        "White Matter": TISSUE_PARAMS["white_matter_adult"],
        "Grey Matter": TISSUE_PARAMS["grey_matter_adult"],
        "CSF": TISSUE_PARAMS["csf"],
    }

    # Calculate signals for each tissue
    results = {}  # Create an empty dictionary to store results
    for name, params in tissues.items():
        INV1, INV2 = sim.calculate_signals(
            T1=params["T1"],
            PD=params["PD"],
            T2star=params["T2star"]
        )
        # FIX: Convert to floats
        INV1_float = float(INV1)
        INV2_float = float(INV2)
        results[name] = {"INV1": INV1_float, "INV2": INV2_float}

    # Get UNI image for each tissue
    for name, data in results.items():
        INV1 = data["INV1"]
        INV2 = data["INV2"]
        UNI = INV1 / (INV2 + 1e-12)  # ratio form, avoid divide-by-zero
        results[name]["UNI"] = UNI

    # Plot simulated INV1 and INV2 signals for this iteration
    for name, data in results.items():
        INV1_plot = data["INV1"]
        INV2_plot = data["INV2"]

        # Use different marker for each iteration
        plt.plot(
            [sim.TI1, sim.TI2],
            [INV1_plot, INV2_plot],
            marker=markers[iteration],
            color=colors[name],
            label=f"{name} (TI1={sim.TI1}ms)" if iteration == 0 else "",
            alpha=0.7
        )

    # Print values for this iteration
    print(f"\nIteration {iteration + 1} (TI1 = {sim.TI1}ms):")
    print("Tissue           INV1         INV2         UNI")
    print("-----------------------------------------------")
    for name, data in results.items():
        print(
            f"{name:12} {data['INV1']:10.6f} {data['INV2']:10.6f} {data['UNI']:10.6f}")

plt.title(
    "MP2RAGE Longitudinal Recovery Simulation (7T)\n10 Iterations with TI1 increasing by 100ms")
plt.xlabel("Inversion Time (ms)")
plt.ylabel("Longitudinal Magnetization (Mz)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Alternative visualization: Separate plots for each tissue
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tissue_names = ["White Matter", "Grey Matter", "CSF"]

for idx, tissue in enumerate(tissue_names):
    # Reset for clean plotting
    for iteration in range(10):
        protocol = PROTOCOLS['protocol_1'].copy()
        protocol['TI1'] = 1000 + iteration * 100
        sim = MP2RAGESimulator(protocol)

        params = TISSUE_PARAMS[
            "white_matter_adult" if tissue == "White Matter" else
            "grey_matter_adult" if tissue == "Grey Matter" else "csf"]

        INV1, INV2 = sim.calculate_signals(
            T1=params["T1"],
            PD=params["PD"],
            T2star=params["T2star"]
        )

        # FIX: Convert to floats here too
        INV1_float = float(INV1)
        INV2_float = float(INV2)

        axes[idx].plot(
            [sim.TI1, sim.TI2],
            [INV1_float, INV2_float],  # Use the float values
            marker=markers[iteration],
            color=colors[tissue],
            label=f"TI1={sim.TI1}ms",
            alpha=0.7
        )

    axes[idx].set_title(f"{tissue}")
    axes[idx].set_xlabel("Inversion Time (ms)")
    axes[idx].set_ylabel("Longitudinal Magnetization (Mz)")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()