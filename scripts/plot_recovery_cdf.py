#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cumulative error distribution for the frame-level recovery experiment.

Reads the ``*_time_recovery.csv`` files written by
``recovery_wasserstein_cuda.py`` and plots, per parameter, the empirical CDF of
the per-query grid error: both the raw error, and the error above the floor set
by the truth lying off the sweep grid.

    python scripts/plot_recovery_cdf.py --out outputs/time_recovery_wass
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("outputs/time_recovery_wass"),
                   help="Directory holding *_time_recovery.csv (and output).")
    p.add_argument("--cache", type=Path, default=None,
                   help="sweep_features.npz, for the exact sweep grids; "
                        "default <out>/sweep_features.npz. If absent, grids "
                        "are derived from the matched (grid-valued) preds.")
    p.add_argument("--tauV-step", type=int, default=10)
    p.add_argument("--xi-step", type=float, default=0.02)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--x-cap", type=float, default=None,
                   help="Truncate the x-axis at this many grid units, piling the "
                        "tail into the final step (zoomed view). Default: show "
                        "the full error range out to --x-max.")
    p.add_argument("--x-max", type=float, default=12.5,
                   help="Right x-axis limit when showing the full range.")
    p.add_argument("--out-name", default="recovery_cdf.png",
                   help="Output PNG filename within --out.")
    p.add_argument("--matcher-label",
                   default="SampEuler Wasserstein distance nearest-neighbour matching",
                   help="Descriptive matcher name shown in the figure suptitle.")
    return p.parse_args()


def load_grids(cache: Path, rows):
    """Return (tauV_grid, xi_pct_grid, t_grid) as float arrays in native units."""
    if cache is not None and cache.exists():
        d = np.load(str(cache), allow_pickle=False, mmap_mode="r")
        return (np.unique(np.asarray(d["tauV"])).astype(float),
                np.unique(np.asarray(d["xi_pct"])).astype(float),
                np.unique(np.asarray(d["t"])).astype(float))
    # Fallback: matched (predicted) values are always sweep-grid points.
    print(f"[warn] cache {cache} not found; deriving grids from matched preds")
    return (np.unique([int(r["pred_tauV_l2"])   for r in rows]).astype(float),
            np.unique([int(r["pred_xi_pct_l2"]) for r in rows]).astype(float),
            np.unique([int(r["pred_t_l2"])      for r in rows]).astype(float))


def ecdf(arr: np.ndarray):
    a = np.sort(np.asarray(arr, dtype=float))
    y = np.arange(1, len(a) + 1) / len(a)
    return a, y


def main() -> int:
    args = parse_args()
    cache = args.cache or (args.out / "sweep_features.npz")

    csvs = sorted(glob.glob(str(args.out / "*_time_recovery.csv")))
    if not csvs:
        raise SystemExit(f"No *_time_recovery.csv under {args.out}")
    rows = []
    for c in csvs:
        with open(c) as f:
            rows.extend(list(csv.DictReader(f)))
    n_fakexp = len({r["fakeexp"] for r in rows})
    print(f"[load] {len(rows)} queries from {n_fakexp} fakexps")

    tauV_grid, xi_grid, t_grid = load_grids(cache, rows)

    true_tau = np.array([float(r["true_tauV"])   for r in rows])
    true_xi  = np.array([float(r["true_xi_pct"]) for r in rows])
    true_t   = np.array([float(r["true_t"])      for r in rows])
    abs_tau  = np.abs(np.array([float(r["tauV_grid_err_l2"]) for r in rows]))
    abs_xi   = np.abs(np.array([float(r["xi_grid_err_l2"])   for r in rows]))
    abs_t    = np.abs(np.array([float(r["t_grid_err_l2"])    for r in rows]))

    # Per-query floor = distance from truth to nearest grid point (grid units).
    # τV/ξ floors are constant within a sim (fixed true param); t floors vary
    # per query because the fakexp frame times sit off the sweep's t-grid.
    floor_tau = np.array([np.min(np.abs(tauV_grid - v)) for v in true_tau]) / args.tauV_step
    floor_xi  = np.array([np.min(np.abs(xi_grid - v)) for v in true_xi]) * 0.01 / args.xi_step
    floor_t   = np.array([np.min(np.abs(t_grid - v)) for v in true_t]) / args.stride

    # Excess: error above the best a grid-restricted matcher can achieve.
    # For τV/ξ the clip never bites (|Δ| ≥ floor by construction); it can for t.
    exc_tau = np.maximum(0.0, abs_tau - floor_tau)
    exc_xi  = np.maximum(0.0, abs_xi  - floor_xi)
    exc_t   = np.maximum(0.0, abs_t   - floor_t)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        ("C0", r"$\tau_V$", abs_tau, exc_tau),
        ("C1", r"$\xi$",    abs_xi,  exc_xi),
        ("C2", r"$t_D$",    abs_t,   exc_t),
    ]
    # Default: show the full error range (nothing piled). Pass --x-cap N to
    # truncate at N grid units and pile the tail into the final step.
    if args.x_cap is not None:
        cap, x_max, piled = float(args.x_cap), float(args.x_cap) + 0.5, True
    else:
        cap = max(float(a.max()) for a in (abs_tau, abs_xi, abs_t,
                                           exc_tau, exc_xi, exc_t))
        x_max, piled = args.x_max, False

    def draw(ax, mode):
        for color, name, raw, exc in series:
            arr = exc if mode == "excess" else raw
            xs, ys = ecdf(arr)
            # Pile the tail beyond the cap into the final step, then carry the
            # line flat out to x_max so the plateau at 1 is visible instead of
            # the final rise sitting exactly on the right spine.
            xs_p = np.concatenate(([0.0], np.minimum(xs, cap), [x_max]))
            ys_p = np.concatenate(([0.0], ys, [ys[-1]]))
            if mode == "excess":
                lab = (f"{name}  at floor {np.mean(arr <= 1e-9):.0%}, "
                       f"≤1 above {np.mean(arr <= 1.0):.0%}")
            else:
                lab = f"{name}  ≤1 cell {np.mean(arr <= 1.0):.0%}"
            ax.step(xs_p, ys_p, where="post", color=color, linewidth=1.8, label=lab)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        if piled:
            ax.text(0.98, 0.34,
                    f"tail beyond {cap:.0f} grid units\npiled into the final step",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="0.45")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    draw(axL, "raw")
    axL.set_title("Raw error (includes off-grid discretisation cost)")
    axL.set_xlabel(r"$|\Delta|$ in grid units (1 = one grid neighbour)")
    axL.set_ylabel("fraction of queries")

    draw(axR, "excess")
    axR.set_title(r"Excess error above grid floor:  $\max(0,\;|\Delta|-\mathrm{floor})$")
    axR.set_xlabel(r"excess $|\Delta|$ in grid units (0 = grid-optimal)")
    axR.tick_params(labelleft=True)  # show y-axis numbers on the shared right panel

    fig.suptitle(f"{args.matcher_label}: cumulative error distribution",
                 fontsize=11, y=0.99)

    out_png = args.out / args.out_name
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
