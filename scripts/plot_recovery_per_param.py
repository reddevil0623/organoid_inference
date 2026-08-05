#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-parameter recovery error versus true simulation time.

Three panels: matched versus true t_D, and the absolute grid error for tau_V
and xi, each summarised by a sliding-window mean over the k queries nearest in
true time.

    python scripts/plot_recovery_per_param.py --out outputs/time_recovery_wass
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TAUV_STEP, XI_STEP, STRIDE = 10, 0.02, 50
C_TAU, C_XI, C_T = "#4477aa", "#ee6677", "#228833"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("outputs/time_recovery_wass"),
                   help="Directory holding *_time_recovery.csv (and output).")
    p.add_argument("--out-name", default="recovery_per_param.png",
                   help="Output PNG filename within --out.")
    p.add_argument("--k", type=int, default=15,
                   help="Sliding-window width (nearest queries in t).")
    p.add_argument("--eval-n", type=int, default=160,
                   help="Number of evaluation points along the t-axis.")
    return p.parse_args()


def sliding_window(t, err, k=15, eval_n=160):
    """Mean ± SEM of ``err`` over the k queries nearest each eval point in t."""
    order = np.argsort(t)
    t_sorted = t[order]
    e_sorted = err[order]
    eval_t = np.linspace(t_sorted[0], t_sorted[-1], eval_n)
    means, sems = [], []
    for tv in eval_t:
        idx = np.argsort(np.abs(t_sorted - tv))[:k]
        w = e_sorted[idx]
        means.append(float(np.mean(w)))
        sems.append(float(np.std(w, ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0.0)
    return eval_t, np.array(means), np.array(sems)


def main() -> int:
    args = parse_args()

    csvs = sorted(glob.glob(str(args.out / "fakexp_*__time_recovery.csv")))
    if not csvs:
        raise SystemExit(f"No fakexp_*__time_recovery.csv under {args.out}")
    rows = []
    for c in csvs:
        with open(c) as f:
            rows.extend(list(csv.DictReader(f)))
    n_fakexps = len({r["fakeexp"] for r in rows})
    print(f"[load] {len(rows)} queries from {n_fakexps} fakexps")

    true_t   = np.array([float(r["true_t"])        for r in rows])
    pred_t   = np.array([float(r["pred_t_l2"])     for r in rows])
    true_tau = np.array([int(r["true_tauV"])       for r in rows])
    pred_tau = np.array([int(r["pred_tauV_l2"])    for r in rows])
    true_xi  = np.array([int(r["true_xi_pct"])     for r in rows])
    pred_xi  = np.array([int(r["pred_xi_pct_l2"])  for r in rows])

    # Raw absolute grid errors -- no capping, no floor subtraction.
    tau_g = np.abs(pred_tau - true_tau) / TAUV_STEP
    xi_g  = np.abs(pred_xi  - true_xi) * 0.01 / XI_STEP
    t_g   = np.abs(pred_t   - true_t)  / STRIDE

    sp_tau = spearmanr(true_t, tau_g)
    sp_xi  = spearmanr(true_t, xi_g)
    sp_t   = spearmanr(true_t, t_g)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth_panel(ax, err_a, color, label, sp_res):
        et, mu, se = sliding_window(true_t, err_a, k=args.k, eval_n=args.eval_n)
        ax.fill_between(et, mu - se, mu + se, color=color, alpha=0.20,
                        linewidth=0, label=r"sliding mean $\pm$ SEM ($k{=}15$)")
        ax.plot(et, mu, color=color, linewidth=2.2, zorder=3)
        ax.set_xlim(et[0], et[-1])
        ax.set_ylim(bottom=-0.1)
        ax.set_xlabel("true $t_D$ (model time units)")
        ax.set_ylabel(rf"$|\Delta|$ in {label} grid units")
        ax.set_title(f"{label}: error vs true $t_D$  "
                     f"($\\rho={sp_res.statistic:+.2f}$, $p={sp_res.pvalue:.2g}$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7))

    smooth_panel(axes[1], tau_g, C_TAU, r"$\tau_V$", sp_tau)
    smooth_panel(axes[2], xi_g, C_XI, r"$\xi$", sp_xi)

    ax_t = axes[0]
    ax_t.scatter(true_t, pred_t, s=34, alpha=0.85, color=C_T,
                 edgecolor="white", linewidth=0.5)
    lo = float(min(true_t.min(), pred_t.min()))
    hi = float(max(true_t.max(), pred_t.max()))
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    ax_t.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--",
              linewidth=1.0, alpha=0.5, label="$y = x$")
    ax_t.set_xlim(lo - pad, hi + pad)
    ax_t.set_ylim(lo - pad, hi + pad)
    ax_t.set_xlabel("true $t_D$ (model time units)")
    ax_t.set_ylabel("matched $t_D$ (model time units)")
    ax_t.set_title(f"$t_D$: true vs matched  "
                   f"($\\rho={sp_t.statistic:+.2f}$, $p={sp_t.pvalue:.2g}$)")
    ax_t.set_aspect("equal", adjustable="box")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8, loc="upper left")

    fig.suptitle("Per-parameter recovery error versus true simulation time",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_png = args.out / args.out_name
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
