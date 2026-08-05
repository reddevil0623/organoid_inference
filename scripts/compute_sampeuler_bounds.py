#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determine the (x, chi) discretisation used for SampEuler vectorisation.

Computes the Euler characteristic curve of every (frame, direction) pair,
records the empirical chi range, and saves the chosen grid together with the
directions, so downstream featurisation is reproducible.

    python scripts/compute_sampeuler_bounds.py \
        --root data/DATA_stride5/sweep --pattern 'dt002_tauV*_xi*' \
        --n-dirs 100 --xpoints 384 --y-min -10 --y-max 70 --n-chi 80 \
        --out data/sampeuler_bounds.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np

import paths as _p
_p.ensure_eucalc()

from core import ECTComputer, iter_run_frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", nargs="+", type=Path, required=True,
                   help="Directories under which to search for run subdirs.")
    p.add_argument("--pattern", default="dt002_tauV*_xi*",
                   help="Glob pattern for run subdir names.")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Cap the number of run subdirs scanned (debug).")
    p.add_argument("--n-dirs", type=int, default=100,
                   help="Number of random directions per frame.")
    p.add_argument("--xpoints", type=int, default=600,
                   help="Number of samples along the x-axis (filtration).")
    p.add_argument("--x-min", type=float, default=-1.5,
                   help="Lower bound of the x-grid (filtration heights).")
    p.add_argument("--x-max", type=float, default=1.5,
                   help="Upper bound of the x-grid.")
    p.add_argument("--y-min", type=float, default=-10.0,
                   help="Lower bound of the chi-grid for SampEuler image.")
    p.add_argument("--y-max", type=float, default=70.0,
                   help="Upper bound of the chi-grid.")
    p.add_argument("--n-chi", type=int, default=80,
                   help="Number of chi-bins for the SampEuler image. "
                        "Choose n_chi == y_max - y_min for unit-width bins "
                        "(matches the integer chi domain).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the sampled directions.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output npz path (will overwrite).")
    return p.parse_args()


def discover_runs(roots: Iterable[Path], pattern: str) -> List[Path]:
    runs: List[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"  skip (not a dir): {root}", file=sys.stderr)
            continue
        if any(root.glob("u_*.dat")):
            runs.append(root)
            continue
        runs.extend(sorted(p for p in root.glob(pattern) if p.is_dir()))
    return runs


def main() -> int:
    args = parse_args()
    roots = [r.resolve() for r in args.root]
    runs = discover_runs(roots, args.pattern)
    if args.max_runs:
        runs = runs[:args.max_runs]
    if not runs:
        print("No run directories found.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    thetas = rng.uniform(0, 2 * np.pi, args.n_dirs)

    ect = ECTComputer(
        n_dirs=args.n_dirs, xpoints=args.xpoints,
        x_min=args.x_min, x_max=args.x_max,
    )

    print(f"Scanning {len(runs)} run dir(s)")
    print(f"  x-grid: [{args.x_min}, {args.x_max}] @ {args.xpoints} pts")
    print(f"  y-grid: [{args.y_min}, {args.y_max}] @ {args.n_chi} bins "
          f"(width = {(args.y_max - args.y_min) / args.n_chi:g})")
    print(f"  n_dirs={args.n_dirs}, seed={args.seed}")
    for r in runs[:5]:
        print(f"  {r}")
    if len(runs) > 5:
        print(f"  ... and {len(runs) - 5} more")

    y_min_obs = +np.inf
    y_max_obs = -np.inf
    n_frames = 0
    n_nonempty = 0
    t0 = time.time()
    for r in runs:
        for _, lbl in iter_run_frames(r):
            n_frames += 1
            if lbl.sum() == 0:
                continue
            curves = ect.compute(lbl, thetas=thetas)   # (n_dirs, xpoints)
            y_min_obs = min(y_min_obs, float(curves.min()))
            y_max_obs = max(y_max_obs, float(curves.max()))
            n_nonempty += 1
            if n_frames % 50 == 0:
                dt = time.time() - t0
                print(f"  scanned {n_frames} frames in {dt:.1f}s "
                      f"({n_frames / max(dt, 1e-6):.1f} f/s)  "
                      f"observed chi in [{y_min_obs:+.0f}, {y_max_obs:+.0f}]")

    if n_frames == 0:
        print("No frames found across roots.", file=sys.stderr)
        return 1
    if n_nonempty == 0:
        print("All frames empty; chi range undefined.", file=sys.stderr)
        y_min_obs, y_max_obs = 0.0, 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(args.out),
        x_min=np.float64(args.x_min),
        x_max=np.float64(args.x_max),
        y_min=np.float64(args.y_min),
        y_max=np.float64(args.y_max),
        n_chi=np.int64(args.n_chi),
        y_min_obs=np.float64(y_min_obs),
        y_max_obs=np.float64(y_max_obs),
        n_dirs=np.int64(args.n_dirs),
        xpoints=np.int64(args.xpoints),
        seed=np.int64(args.seed),
        thetas=thetas,
    )

    covers = (args.y_min <= y_min_obs) and (args.y_max >= y_max_obs)
    print(f"\nScanned {n_frames} frames over {len(runs)} runs "
          f"({n_nonempty} non-empty).")
    print(f"  x-grid (chosen) = [{args.x_min:+.4f}, {args.x_max:+.4f}] "
          f"@ {args.xpoints} pts")
    print(f"  y-grid (chosen) = [{args.y_min:+.1f}, {args.y_max:+.1f}] "
          f"@ {args.n_chi} bins")
    print(f"  chi observed    = [{y_min_obs:+.1f}, {y_max_obs:+.1f}]  "
          f"{'(grid covers)' if covers else '*** GRID UNDERCOVERS ***'}")
    print(f"\nSaved: {args.out}")
    print(f"Feature dim per frame (n_chi * xpoints): "
          f"{args.n_chi * args.xpoints}")
    if not covers:
        print("\nWARNING: chosen y-grid does not cover the observed chi "
              "range. Curves outside [y_min, y_max] are dropped from the "
              "SampEuler image -- widen --y-min / --y-max.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
