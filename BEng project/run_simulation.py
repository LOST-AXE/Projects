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
    gap = base_protocol["TI2"] - base_protocol["TI1"]

    # Sweep setup


    base_TI1 = 900
    step_TI1 = 5
    requested_n_iters = int((1100 - base_TI1) / step_TI1) + 1
    # -----------------------------
    # NEW: compute valid TI1 range
    # Constraints:
    # TA >= 0  -> TI1 >= (n*TR_GRE)/2
    # TC >= 0  -> TI1 <= TR - gap - (n*TR_GRE)/2
    # -----------------------------
    half_block = (base_protocol["n"] * base_protocol["TR_GRE"]) / 2.0
    max_TI1_allowed = base_protocol["TR_MP2RAGE"] - gap - half_block

    print(f"Min TI1 allowed (TA>=0) = {half_block:.1f} ms")
    print(f"Max TI1 allowed (TC>=0) = {max_TI1_allowed:.1f} ms")

    # Ensure starting TI1 valid
    base_TI1 = max(base_TI1, half_block + 1.0)

    max_iters_allowed = int(np.floor((max_TI1_allowed - base_TI1) / step_TI1)) + 1
    if max_iters_allowed < 1:
        print("No valid TI1 values possible. Increase TR_MP2RAGE or reduce n/gap/TR_GRE.")
        return

    n_iters = min(requested_n_iters, max_iters_allowed)

    # -----------------------------
    # Sweep
    # -----------------------------
    TI1_values = []
    INV1_series = {name: [] for name in tissues.keys()}
    skipped = 0

    for i in range(n_iters):
        protocol = base_protocol.copy()
        protocol["TI1"] = base_TI1 + i * step_TI1
        protocol["TI2"] = protocol["TI1"] + gap  # Option A

        # Guard TA >= 0
        min_TI1 = (protocol["n"] * protocol["TR_GRE"]) / 2.0
        if protocol["TI1"] < min_TI1:
            protocol["TI1"] = min_TI1 + 1.0
            protocol["TI2"] = protocol["TI1"] + gap

        sim = MP2RAGESimulator(protocol, verbose=False)

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
        print("No valid points were generated.")
        return

    TI1_values = np.array(TI1_values)
    wm = np.array(INV1_series["White Matter"])
    gm = np.array(INV1_series["Grey Matter"])

    # 1) Opposite sign (polarity)
    opposite = np.sign(wm) != np.sign(gm)

    # 2) Edge-nulling score (equal & opposite)
    eps = 1e-12
    E = np.abs(wm + gm)
    E_norm = E / (np.abs(wm) + np.abs(gm) + eps)
    A = np.abs(wm) + np.abs(gm)  # amplitude proxy (bigger is better SNR)
    best = np.argmin(E_norm)
    print(f"\nBest (min E_norm) at TI1={TI1_values[best]:.1f} ms | "
          f"WM={wm[best]:+.6f} GM={gm[best]:+.6f} | "
          f"E_norm={E_norm[best]:.4f} | A={A[best]:.6f} | opposite={opposite[best]}")

    # Print best TI1 candidates (lowest E_norm)
    best_idx = np.argsort(E_norm)[:5]
    print("\nBest TI1 candidates for WM/GM cancellation (lowest E_norm):")
    for i in best_idx:
        print(f"TI1={TI1_values[i]:.1f} ms | WM={wm[i]:+.6f} GM={gm[i]:+.6f} "
              f"| WM+GM={wm[i] + gm[i]:+.6f} | E_norm={E_norm[i]:.4f} | opposite={opposite[i]}")

    # -----------------------------
    # Plot INV1 vs TI1
    # -----------------------------
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

    title = "MP2RAGE INV1 vs TI1 (7T) — Option A (TI2 shifts with TI1)"
    if skipped > 0:
        title += f"\n(skipped {skipped} invalid points due to timing limits)"

    plt.title(title)
    plt.xlabel("TI1 (ms)")
    plt.ylabel("INV1 signal (a.u.)")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.axhline(0, linewidth=1)

    # mark best TI1 (linear interpolation using WM+GM)
    f = wm + gm
    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if len(idx) > 0:
        j = idx[0]
        t0, t1 = TI1_values[j], TI1_values[j + 1]
        f0, f1 = f[j], f[j + 1]
        ti1_star = t0 + (0 - f0) * (t1 - t0) / (f1 - f0)
        plt.axvline(ti1_star, linestyle="--")
        print(
            f"\nEstimated TI1* where WM+GM=0 (linear interp): {ti1_star:.1f} ms")

    plt.tight_layout()

    # -----------------------------
    # Mz(t) recovery plot (unchanged)
    # -----------------------------
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

    plt.axvspan(gre1[0], gre1[1], alpha=0.15)
    plt.axvspan(gre2[0], gre2[1], alpha=0.15)

    plt.title("MP2RAGE Longitudinal Recovery Mz(t) over full TR (GRE blocks shaded)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mz (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.xticks(np.linspace(0, 5000, num=21))
    plt.show()


if __name__ == "__main__":
    main()
