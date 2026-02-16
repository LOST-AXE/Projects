import numpy as np
import matplotlib.pyplot as plt

from tissue_library import TISSUE_PARAMS, PROTOCOLS
from mp2rage_simulator import MP2RAGESimulator


def main():
    colors = {"White Matter": "b", "Grey Matter": "k", "CSF": "r"}

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
        protocol["TI2"] = 2220  # <-- THE KEY OPTION A CHANGE

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
    # ------------------------------
    # ADD-ON FIGURE: Mz(t) recovery curve over full TR_MP2RAGE
    # ------------------------------
    sim_for_timecourse = MP2RAGESimulator(base_protocol, verbose=False)

    plt.figure(figsize=(10, 5))

    gre1 = gre2 = None
    for name, params in tissues.items():
        t_ms, mz, gre1, gre2 = sim_for_timecourse.mz_timecourse(
            T1=params["T1"],
            PD=params["PD"],
            dt_ms=2.0
        )
        plt.plot(t_ms, mz, linewidth=2, color=colors[name], label=f"{name} Mz(t)")

    # Shade GRE blocks
    plt.axvspan(gre1[0], gre1[1], alpha=0.15)
    plt.axvspan(gre2[0], gre2[1], alpha=0.15)

    plt.title("MP2RAGE Longitudinal Recovery Mz(t) over full TR (GRE blocks shaded)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mz (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
