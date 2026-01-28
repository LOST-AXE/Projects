import numpy as np
import matplotlib.pyplot as plt

from tissue_library import TISSUE_PARAMS, PROTOCOLS
from mp2rage_simulator import MP2RAGESimulator


def main():
    colors = {"White Matter": "k", "Grey Matter": "r", "CSF": "b"}

    tissues = {
        "White Matter": TISSUE_PARAMS["white_matter_adult"],
        "Grey Matter": TISSUE_PARAMS["grey_matter_adult"],
        "CSF": TISSUE_PARAMS["csf"],
    }

    base_protocol = PROTOCOLS["protocol_1"].copy()

    # Keep TI2 - TI1 constant (Option A)
    gap = base_protocol["TI2"] - base_protocol["TI1"]  # e.g. 1570ms

    # Sweep setup
    n_iters = 29
    base_TI1 = 650
    step_TI1 = 100

    # Store values per tissue (only valid points)
    TI1_values = []
    INV1_series = {name: [] for name in tissues.keys()}

    skipped = 0

    for i in range(n_iters):
        protocol = base_protocol.copy()
        protocol["TI1"] = base_TI1 + i * step_TI1
        protocol["TI2"] = protocol["TI1"] + gap  # <-- THE KEY OPTION A CHANGE

        sim = MP2RAGESimulator(protocol, verbose=False)

        # Only keep physically valid timing
        if not sim.timing_is_valid():
            skipped += 1
            continue

        TI1_values.append(sim.TI1)

        for name, params in tissues.items():
            inv1, inv2 = sim.calculate_signals(
                T1=params["T1"],
                PD=params["PD"],
                T2star=params["T2star"],
            )
            INV1_series[name].append(float(inv1))

    if len(TI1_values) == 0:
        print("No valid points were generated. Increase TR_MP2RAGE or reduce TI1 sweep range.")
        return

    TI1_values = np.array(TI1_values)

    # Plot joined points only
    plt.figure(figsize=(10, 5))

    for name in tissues.keys():
        y = np.array(INV1_series[name])
        plt.plot(
            TI1_values, y,
            marker="o",
            linewidth=2,
            color=colors[name],
            label=f"{name} INV1"
        )

    title = "MP2RAGE INV1 vs TI1 (7T) — Joined Points Only (Option A: TI2 shifts with TI1)"
    if skipped > 0:
        title += f"\n(skipped {skipped} invalid TI1 values due to TR/Timing limits)"

    plt.title(title)
    plt.xlabel("TI1 (ms)")
    plt.ylabel("INV1 signal (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
