"""
Runs all three edge map methods on a dataset and evaluates them using
Precision-Recall curves and Average Precision (AP) scores.

The boundary detection is imbalanced as boundary voxels
are far fewer than WM/GM interior voxels.
PR curves are unaffected by this and directly measure:
   1. Precision: of voxels flagged as boundary, how many actually are?
   2. Recall: of actual boundary voxels, how many were found?
Average Precision (AP = area under PR curve) is the single summary metric.

Ground truth definition:
  y_true = 1  for boundary_valid voxels  (positives)
  y_true = 0  for wm_interior + gm_interior voxels  (negatives)
  Invalid voxels, CSF, background: excluded entirely.

Note on sigma selection:
  All sigmas are evaluated here. Best sigma is chosen by AP on this
  dataset. Report this; validate on independent data in future work.

Methods compared:
  Option A     : EdgeA * Sobel gradient
  FLAWS E1     : raw min(|A|, |B|)
  FLAWS E2     : weighted min(A_n, B_n)
  T1 notch     : all sigmas evaluated independently
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

from Utils import get_subject_id, get_boundary_masks, get_valid_mask, save_figure

MAT_PATH = (
    "C:/Users/jiges/Downloads/RICE092_fixed.mat"
)

SIGMAS = [75, 100, 125, 150, 200]


def build_y_true(boundary_valid, wm_interior, gm_interior):
    eval_mask = boundary_valid | wm_interior | gm_interior
    y_true    = boundary_valid[eval_mask].astype(int)
    return eval_mask, y_true


def compute_pr(y_true, score_map, eval_mask, label):
    y_score = score_map[eval_mask].astype(float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap      = average_precision_score(y_true, y_score)
    f1      = 2 * precision * recall / (precision + recall + 1e-9)
    best    = int(np.argmax(f1[:-1]))

    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  AP={ap:.4f}  Best F1={f1[best]:.4f}  "
          f"Thresh={thresholds[best]:.4f}  "
          f"Prec={precision[best]:.4f}  Rec={recall[best]:.4f}")
    print(f"{'═'*55}")

    return dict(precision=precision, recall=recall, thresholds=thresholds,
                ap=ap, best_thresh=float(thresholds[best]),
                best_f1=float(f1[best]),
                best_prec=float(precision[best]),
                best_rec=float(recall[best]))


def main(mat_path: str = MAT_PATH):

    subject_id = get_subject_id(mat_path)
    print(f"\n{'-'*10}")
    print(f"  PR COMPARISON  |  subject: {subject_id}")
    print(f"{'-'*10}")

    # Ground truth
    mat  = sio.loadmat(mat_path)
    x    = mat["T1_soln"].shape[2] // 2
    T1   = np.clip(mat["T1_soln"].astype(np.float64)[x, :, :], 200.0, 30000.0)
    if "mask_wm" in mat and "mask_gm" in mat:
        wm = mat["mask_wm"].astype(np.uint8)[x, :, :] > 0
        gm = mat["mask_gm"].astype(np.uint8)[x, :, :] > 0
    else:
        wm = np.zeros(T1.shape, dtype=bool)
        gm = np.zeros(T1.shape, dtype=bool)
        print("  No WM/GM masks in file — will use SPM probability maps.")

    valid = get_valid_mask(T1)
    boundary_valid, wm_interior, gm_interior = get_boundary_masks(wm, gm,
                                                                  valid,
                                                                  mat_path)
    eval_mask, y_true = build_y_true(boundary_valid, wm_interior, gm_interior)

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    baseline_ap = n_pos / (n_pos + n_neg)
    print(f"\n  {n_pos} boundary voxels  |  {n_neg} interior voxels")
    print(f"  Baseline AP (random) = {baseline_ap:.4f}")

    # Run methods
    print("\n[1/3] Running Option A...")
    from optionA_edge_map import main as run_optionA
    score_A, _ = run_optionA(mat_path)

    print("\n[2/3] Running FLAWS dual-null...")
    from flaws_dual_null import main as run_flaws
    E1, _, E2, _ = run_flaws(mat_path)

    print("\n[3/3] Running T1 notch (all sigmas)...")
    from t1_notch import main as run_notch
    notch_maps, _, _ = run_notch(mat_path)
    # notch_maps is dict {sigma: score_map}

    # PR curves
    print("\n── Precision-Recall evaluation ───────────────────────")
    pr_A  = compute_pr(y_true, score_A, eval_mask, "Option A")
    pr_E1 = compute_pr(y_true, E1,      eval_mask, "FLAWS E1 — raw min")
    pr_E2 = compute_pr(y_true, E2,      eval_mask, "FLAWS E2 — weighted min")
    pr_N  = {sigma: compute_pr(y_true, notch_maps[sigma], eval_mask,
                               f"T1 notch sigma={sigma} ms")
             for sigma in SIGMAS}

    # Summary table
    all_results = [
        ("Option A",           pr_A),
        ("FLAWS E1",           pr_E1),
        ("FLAWS E2",           pr_E2),
    ] + [(f"Notch σ={s}ms", pr_N[s]) for s in SIGMAS]

    best_ap = max(r["ap"] for _, r in all_results)
    print(f"\n{'═'*62}")
    print(f"  {'Method':<30} {'AP':>7} {'Best F1':>8} {'Thresh':>8}")
    print(f"  {'─'*60}")
    for label, r in all_results:
        star = " ★" if abs(r["ap"] - best_ap) < 1e-6 else "  "
        print(f"  {label:<30} {r['ap']:>7.4f} {r['best_f1']:>8.4f} "
              f"{r['best_thresh']:>8.4f}{star}")
    print(f"  {'─'*60}")
    print(f"  best = highest AP  |  baseline = {baseline_ap:.4f}")
    print(f"{'═'*62}")

    # PR figure
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(baseline_ap, color="gray", ls=":", lw=1.2,
               label=f"Baseline AP={baseline_ap:.3f}")

    for pr, label, colour, ls in [
        (pr_A,  "Option A", "steelblue",  "-"),
        (pr_E1, "FLAWS E1", "darkorange", "--"),
        (pr_E2, "FLAWS E2", "green",      "-"),
    ]:
        ax.plot(pr["recall"], pr["precision"], color=colour, ls=ls, lw=2,
                label=f"{label}  AP={pr['ap']:.4f}")
        ax.scatter(pr["best_rec"], pr["best_prec"], color=colour, s=80, zorder=5)

    notch_colours = plt.cm.Reds(np.linspace(0.35, 0.9, len(SIGMAS)))
    for (sigma, pr), colour in zip(pr_N.items(), notch_colours):
        ax.plot(pr["recall"], pr["precision"], color=colour, ls="--", lw=1.5,
                label=f"Notch σ={sigma}ms  AP={pr['ap']:.4f}")
        ax.scatter(pr["best_rec"], pr["best_prec"], color=colour, s=60, zorder=5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1])
    ax.set_title(f"Precision-Recall — {subject_id}\n"
                 f"Dots = best-F1 threshold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, mat_path, "PR_comparison")
    plt.show()

    return dict(optionA=pr_A, flaws_E1=pr_E1, flaws_E2=pr_E2, notch=pr_N)


if __name__ == "__main__":
    main()