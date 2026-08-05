#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU Hungarian-Wasserstein distances for ABC rejection. Requires a CUDA GPU.

Accelerator for ``scripts.abc_rej.compute_distances``, writing the same
``<label>__distances.csv`` schema.

``--mode single``   distance from one observation frame to every cached frame.
``--mode temporal`` distance between n-frame trajectories, summed over aligned
                    frames.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

_LIB = Path(__file__).resolve().parent.parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
import eucalc as ec  # noqa: E402
from numba import cuda  # noqa: E402

_FAKEXP_RE = re.compile(r"fakexp_(?P<idx>\d+)_tauV(?P<tauV>\d+)_xi(?P<xi>\d+)_[0-9a-f]+")


# Frame helpers (pinned; identical preprocessing to the abc_rej cache).
def load_field(path: Path) -> np.ndarray:
    return np.loadtxt(str(path))


def u_s_to_labels(U, S, u_thresh=1e-4, s_thresh=0.5):
    L = np.zeros_like(U, dtype=np.uint8)
    L[U > u_thresh] = 1
    L[S > s_thresh] = 2
    return L


def list_field_times(run_dir: Path) -> List[int]:
    out = []
    for p in run_dir.glob("u_*.dat"):
        try:
            out.append(int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    return sorted(out)


def even_sample(ts: List[int], n: int) -> List[int]:
    if not ts or n >= len(ts):
        return list(ts)
    idx = np.linspace(0, len(ts) - 1, num=n)
    return sorted({ts[int(round(x))] for x in idx})


def find_fakexp_dir(idx: int, roots: List[Path]) -> Optional[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for d in sorted(root.glob(f"fakexp_{idx}_*")):
            if d.is_dir() and any(d.glob("u_*.dat")):
                m = _FAKEXP_RE.match(d.name)
                if m and int(m.group("idx")) == idx:
                    return d
    return None


# SampEuler (ECT) on {0,1,2} labels with random directions — matches the cache.
def compute_ect(label_img: np.ndarray, n_dirs: int, xpoints: int,
                x_min: float, x_max: float, rng) -> np.ndarray:
    if int(label_img.sum()) == 0:
        return np.zeros((n_dirs, xpoints), dtype=np.float32)
    cplx = ec.EmbeddedComplex(label_img)
    cplx.preproc_ect()
    thetas = rng.uniform(0.0, 2.0 * np.pi, n_dirs)
    T = np.linspace(x_min, x_max, xpoints)
    out = np.empty((n_dirs, xpoints), dtype=np.float32)
    for i, th in enumerate(thetas):
        d = np.array((np.sin(th), np.cos(th)))
        euler = cplx.compute_euler_characteristic_transform(d)
        out[i] = [euler.evaluate(t) for t in T]
    return out


def obs_ects(fakedir: Path, times: List[int], cfg, seed: int) -> np.ndarray:
    """ECTs for the chosen observation frame times -> (len(times), n_dirs, xpoints)."""
    arrs = []
    for k, t in enumerate(times):
        L = u_s_to_labels(load_field(fakedir / f"u_{t}.dat"),
                          load_field(fakedir / f"s_{t}.dat"),
                          cfg.u_thresh, cfg.s_thresh)
        arrs.append(compute_ect(L, cfg.n_dirs, cfg.xpoints, cfg.x_min, cfg.x_max,
                                np.random.default_rng(seed + k)))
    return np.stack(arrs).astype(np.float32)


# GPU Hungarian--Wasserstein: cost matrices on device, assignment on host.
@cuda.jit
def _cost_kernel(Q, R, b_off, B, C, delta_x):
    """C[b,i,j] = delta_x * sum_t |Q[i,t] - R[b_off+b, j, t]|  (Q vs a batch of R)."""
    b, i, j = cuda.grid(3)
    if b < B and i < Q.shape[0] and j < R.shape[1]:
        s = 0.0
        for t in range(Q.shape[1]):
            s += abs(Q[i, t] - R[b_off + b, j, t])
        C[b, i, j] = s * delta_x


def hungarian_one_to_many(q: np.ndarray, R: np.ndarray, delta_x: float,
                          C_dev, batch: int) -> np.ndarray:
    """Hungarian-L1 distance from one ECT curve set ``q`` (n_dirs, xpoints) to
    every ECT in ``R`` (N, n_dirs, xpoints). Returns (N,) float64.
    ``C_dev`` is a reusable (batch, n_dirs, n_dirs) device array."""
    N, nd = R.shape[0], R.shape[1]
    R_dev = cuda.to_device(np.ascontiguousarray(R, dtype=np.float32))
    q_dev = cuda.to_device(np.ascontiguousarray(q, dtype=np.float32))
    tpb = (4, 8, 8)
    out = np.empty(N, dtype=np.float64)
    for b0 in range(0, N, batch):
        B = min(batch, N - b0)
        bpg = (math.ceil(B / tpb[0]), math.ceil(nd / tpb[1]), math.ceil(nd / tpb[2]))
        _cost_kernel[bpg, tpb](q_dev, R_dev, b0, B, C_dev, delta_x)
        Ch = C_dev.copy_to_host()[:B]
        for b in range(B):
            r, c = linear_sum_assignment(Ch[b])
            out[b0 + b] = Ch[b][r, c].mean()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("single", "temporal"), default="single")
    p.add_argument("--bounds", type=Path,
                   default=Path("data/sampeuler_bounds.npz"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("outputs/ABC_REJ/cache"),
                   help="dir of abcrej_*.npz (single) or the temporal cache npzs.")
    p.add_argument("--cache-glob", default="abcrej_[0-9]*.npz")
    p.add_argument("--fakexp-idx", type=int, required=True)
    p.add_argument("--fakexp-root", type=Path,
                   default=Path("data/DATA_stride5/fakexp"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/ABC_REJ/distances"))
    p.add_argument("--obs-t", type=int, default=None,
                   help="single mode: observation frame time (default: last).")
    p.add_argument("--n-frames", type=int, default=10,
                   help="temporal mode: frames per trajectory.")
    p.add_argument("--time-sampling", choices=("absolute", "relative"),
                   default="absolute",
                   help="temporal mode: how cache & obs frame times are matched.")
    p.add_argument("--temporal-agg", choices=("sum", "max", "quantile", "pooled"),
                   default="sum",
                   help="temporal mode: aligned per-frame Hungarian aggregated over "
                        "time as 'sum' (mean / L1), 'max' (L-inf), 'quantile' "
                        "(soft-max, set --temporal-quantile); or 'pooled' = single "
                        "Hungarian over all frames' curves (alignment-free).")
    p.add_argument("--temporal-quantile", type=float, default=0.9,
                   help="quantile for --temporal-agg quantile (default 0.9 = q90).")
    p.add_argument("--u-thresh", type=float, default=1e-4)
    p.add_argument("--s-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-batch", type=int, default=256)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bd = np.load(str(args.bounds))
    args.n_dirs = int(bd["n_dirs"]); args.xpoints = int(bd["xpoints"])
    args.x_min = float(bd["x_min"]); args.x_max = float(bd["x_max"])
    delta_x = (args.x_max - args.x_min) / (args.xpoints - 1)
    print(f"[cfg] mode={args.mode} n_dirs={args.n_dirs} xpoints={args.xpoints} "
          f"delta_x={delta_x:.6g}; {len(cuda.gpus)} GPU(s)")

    fakedir = find_fakexp_dir(args.fakexp_idx, [args.fakexp_root.resolve()])
    if fakedir is None:
        sys.exit(f"fakexp idx {args.fakexp_idx} not found under {args.fakexp_root}")
    all_t = list_field_times(fakedir)

    C_dev = cuda.device_array((args.gpu_batch, args.n_dirs, args.n_dirs),
                              dtype=np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.mode == "single":
        obs_t = args.obs_t if args.obs_t is not None else all_t[-1]
        obs_t = min(all_t, key=lambda x: abs(x - obs_t))
        obs = obs_ects(fakedir, [obs_t], args, args.seed)[0]   # (n_dirs, xpoints)
        label = f"fakexp_idx{args.fakexp_idx}_t{obs_t}"
        print(f"[obs] {label}: 1 frame")

        rows = []
        npzs = sorted(args.cache_dir.glob(args.cache_glob))
        print(f"[cache] {len(npzs)} sim NPZs")
        for k, p in enumerate(npzs):
            d = np.load(p, allow_pickle=False)
            if "ect_curves" not in d.files:
                continue
            R = d["ect_curves"]                      # (n_sampled, n_dirs, xpoints)
            dist = hungarian_one_to_many(obs, R, delta_x, C_dev, args.gpu_batch)
            tau = float(d["tauV"]); xi = float(d["xi"]); th = int(d["theta_idx"])
            ts = np.asarray(d["t"]).astype(int)
            for fi in range(R.shape[0]):
                rows.append((th, fi, tau, xi, int(ts[fi]), float(dist[fi])))
            if (k + 1) % 200 == 0:
                print(f"  scored {k+1}/{len(npzs)} NPZs in {time.time()-t0:.1f}s",
                      flush=True)
        _write_csv(args.out_dir / f"{label}__distances.csv", rows)

    else:  # temporal
        if args.time_sampling == "relative":
            obs_times = even_sample(all_t, args.n_frames)
        else:  # absolute: evenly spaced over the observed time range
            grid = np.linspace(all_t[0], all_t[-1], args.n_frames)
            obs_times = sorted({min(all_t, key=lambda x: abs(x - g)) for g in grid})
        obs = obs_ects(fakedir, obs_times, args, args.seed)     # (n, n_dirs, xpoints)
        n = obs.shape[0]
        agg_tag = (f"q{int(round(args.temporal_quantile * 100))}"
                   if args.temporal_agg == "quantile" else args.temporal_agg)
        label = f"fakexp_idx{args.fakexp_idx}_seq{n}_{args.time_sampling}_{agg_tag}"
        print(f"[obs] {label}: {n} frames at t={obs_times}")

        rows = []
        npzs = sorted(args.cache_dir.glob(args.cache_glob))
        print(f"[cache] {len(npzs)} sim trajectories")
        for k, p in enumerate(npzs):
            d = np.load(p, allow_pickle=False)
            R = d["ect_curves"]                      # (m, n_dirs, xpoints), m aligned frames
            tau = float(d["tauV"]); xi = float(d["xi"]); th = int(d["theta_idx"])
            m = min(n, R.shape[0])
            if args.temporal_agg == "pooled":
                # one Hungarian over the stacked curve sets (alignment-free)
                qpool = obs[:m].reshape(-1, args.xpoints)       # (m*n_dirs, xpoints)
                rpool = R[:m].reshape(1, -1, args.xpoints)      # (1, m*n_dirs, xpoints)
                dval = hungarian_one_to_many(qpool, rpool, delta_x, C_dev,
                                             args.gpu_batch)[0]
            else:
                # aligned per-frame Hungarian, then a norm over the n time
                # positions: sum -> mean (L1, dilutes), max -> worst frame
                # (Linf, most-discriminative), quantile -> soft-max.
                dper = np.array([
                    hungarian_one_to_many(
                        obs[i], R[i:i + 1], delta_x, C_dev, args.gpu_batch)[0]
                    for i in range(m)])
                if args.temporal_agg == "max":
                    dval = float(dper.max())
                elif args.temporal_agg == "quantile":
                    dval = float(np.quantile(dper, args.temporal_quantile))
                else:  # sum -> mean over the n positions (matches prior runs)
                    dval = float(dper.mean())
            rows.append((th, -1, tau, xi, -1, float(dval)))
            if (k + 1) % 200 == 0:
                print(f"  scored {k+1}/{len(npzs)} trajectories in "
                      f"{time.time()-t0:.1f}s", flush=True)
        _write_csv(args.out_dir / f"{label}__distances.csv", rows)

    print(f"[done] {len(rows)} rows in {time.time()-t0:.1f}s")
    return 0


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_idx", "frame_idx", "tauV", "xi", "t", "distance"])
        w.writerows(rows)
    print(f"[out] wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
