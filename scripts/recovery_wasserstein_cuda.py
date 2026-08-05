#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frame-level (tauV, xi, t) recovery under the Hungarian-Wasserstein distance
on SampEuler ECT curves. Requires a CUDA GPU.

Matches each query frame to its nearest frame in the parameter sweep and writes
``<name>__time_recovery.csv`` plus ``sweep_features.npz``, which
``plot_recovery_cdf.py`` and ``plot_recovery_per_param.py`` read.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# eucalc lives as a compiled .so under <project>/lib (same as core.paths).
_LIB = Path(__file__).resolve().parent.parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
import eucalc as ec  # noqa: E402

from numba import cuda  # noqa: E402

_PARAM_RE = re.compile(r"tauV(?P<tauV>\d+)_xi(?P<xi>\d+)")


# Frame helpers — pinned copies of the core.py functions, so this script does
# not depend on the core.py installed on the remote host.
def load_field(path: Path) -> np.ndarray:
    return np.loadtxt(str(path))


def list_field_indices(run_dir: Path) -> List[int]:
    out = []
    for p in run_dir.glob("u_*.dat"):
        try:
            out.append(int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    out.sort()
    return out


def even_sample(ts: List[int], n: int) -> List[int]:
    if not ts:
        return []
    if n >= len(ts):
        return list(ts)
    idxs = np.linspace(0, len(ts) - 1, num=n)
    return sorted({ts[int(round(x))] for x in idxs})


def u_s_to_labels(U, S, u_thresh=1e-4, s_thresh=0.5):
    # Binarise the two fields into a {0,1,2} label image:
    #   0 = background, 1 = cell (U > u_thresh), 2 = lumen (S > s_thresh).
    L = np.zeros_like(U, dtype=np.uint8)
    L[U > u_thresh] = 1
    L[S > s_thresh] = 2
    return L


def parse_label_params(name: str) -> Optional[Tuple[int, int]]:
    m = _PARAM_RE.search(name)
    if not m:
        return None
    return int(m.group("tauV")), int(m.group("xi"))


# SampEuler (ECT) with numpy-uniform random directions, fed the {0,1,2} label
# directly, with no vectorisation step.
def compute_ect(label_img: np.ndarray, n_dirs: int, xpoints: int,
                x_min: float, x_max: float, rng) -> np.ndarray:
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


def frame_to_ect(u_path: Path, s_path: Path, cfg, seed: int) -> Optional[np.ndarray]:
    """load -> u_s_to_labels -> ECT(random dirs). Returns (n_dirs, xpoints)
    float32, or None for an empty frame."""
    L = u_s_to_labels(load_field(u_path), load_field(s_path),
                      cfg.u_thresh, cfg.s_thresh)
    if int(L.sum()) == 0:
        return None
    return compute_ect(L, cfg.n_dirs, cfg.xpoints, cfg.x_min, cfg.x_max,
                       np.random.default_rng(seed))


# Reference cache (sweep frames).
def build_reference_ects(runs: List[Path], cfg, cache_path: Path):
    """Compute (or load) ECTs for every sweep frame. Returns
    (R, tauV, xi_pct, t, label) where R is (N, n_dirs, xpoints) float32."""
    if cache_path.exists():
        d = np.load(str(cache_path), allow_pickle=False)
        ok = (int(d["n_dirs"]) == cfg.n_dirs and int(d["xpoints"]) == cfg.xpoints
              and float(d["x_min"]) == cfg.x_min and float(d["x_max"]) == cfg.x_max
              and float(d["u_thresh"]) == cfg.u_thresh
              and float(d["s_thresh"]) == cfg.s_thresh
              and int(d["seed"]) == cfg.seed)
        if ok:
            print(f"[cache] reusing reference ECTs from {cache_path}")
            return (d["R"], d["tauV"], d["xi_pct"], d["t"], d["label"])
        print(f"[cache] {cache_path} config mismatch — recomputing")

    jobs = []
    for r in runs:
        params = parse_label_params(r.name)
        if params is None:
            continue
        tauV, xi_pct = params
        for t in list_field_indices(r):
            u = r / f"u_{t}.dat"
            s = r / f"s_{t}.dat"
            if u.exists() and s.exists():
                jobs.append((u, s, tauV, xi_pct, t, r.name))
    print(f"[refs] {len(jobs)} sweep frames to featurise "
          f"(n_dirs={cfg.n_dirs}, xpoints={cfg.xpoints})")

    from joblib import Parallel, delayed

    def _one(idx, u, s):
        # Deterministic per-frame direction seed for reproducibility.
        return frame_to_ect(u, s, cfg, seed=cfg.seed + idx)

    t0 = time.time()
    ects = Parallel(n_jobs=cfg.jobs, verbose=5)(
        delayed(_one)(i, j[0], j[1]) for i, j in enumerate(jobs))
    R, tauV, xi_pct, t, label = [], [], [], [], []
    for (u, s, tv, xp, tt, nm), e in zip(jobs, ects):
        if e is None:
            continue
        R.append(e); tauV.append(tv); xi_pct.append(xp); t.append(tt); label.append(nm)
    R = np.stack(R).astype(np.float32)
    tauV = np.asarray(tauV, np.int32); xi_pct = np.asarray(xi_pct, np.int32)
    t = np.asarray(t, np.int32); label = np.asarray(label)
    print(f"[refs] featurised {R.shape[0]} frames in {time.time()-t0:.1f}s "
          f"-> R={R.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), R=R, tauV=tauV, xi_pct=xi_pct, t=t, label=label,
             n_dirs=np.int64(cfg.n_dirs), xpoints=np.int64(cfg.xpoints),
             x_min=np.float64(cfg.x_min), x_max=np.float64(cfg.x_max),
             u_thresh=np.float64(cfg.u_thresh), s_thresh=np.float64(cfg.s_thresh),
             seed=np.int64(cfg.seed))
    print(f"[cache] saved reference ECTs to {cache_path}")
    return R, tauV, xi_pct, t, label


# GPU Hungarian--Wasserstein distance matrix (queries x references).
@cuda.jit
def _cost_kernel(Q, R, b_off, B, C, delta_x):
    """C[b,i,j] = delta_x * sum_t |Q[i,t] - R[b_off+b, j, t]|."""
    b, i, j = cuda.grid(3)
    if b < B and i < Q.shape[0] and j < R.shape[1]:
        s = 0.0
        for t in range(Q.shape[1]):
            s += abs(Q[i, t] - R[b_off + b, j, t])
        C[b, i, j] = s * delta_x


def wasserstein_rows(R: np.ndarray, queries: List[np.ndarray],
                     delta_x: float, batch: int) -> np.ndarray:
    """Return D (n_queries, n_refs): Hungarian-L1 distance from each query to
    each reference. Cost matrices on GPU, assignment on CPU."""
    N = R.shape[0]
    nd = R.shape[1]
    R_dev = cuda.to_device(np.ascontiguousarray(R, dtype=np.float32))
    C_dev = cuda.device_array((batch, nd, nd), dtype=np.float32)
    tpb = (4, 8, 8)
    D = np.empty((len(queries), N), dtype=np.float64)
    for qi, q in enumerate(queries):
        q_dev = cuda.to_device(np.ascontiguousarray(q, dtype=np.float32))
        for b0 in range(0, N, batch):
            B = min(batch, N - b0)
            bpg = (math.ceil(B / tpb[0]), math.ceil(nd / tpb[1]),
                   math.ceil(nd / tpb[2]))
            _cost_kernel[bpg, tpb](q_dev, R_dev, b0, B, C_dev, delta_x)
            Ch = C_dev.copy_to_host()[:B]
            for b in range(B):
                r, c = linear_sum_assignment(Ch[b])
                D[qi, b0 + b] = Ch[b][r, c].mean()
        print(f"  [match] query {qi+1}/{len(queries)} done", flush=True)
    return D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bounds", type=Path, default=Path("data/sampeuler_bounds.npz"),
                   help="SampEuler bounds npz (n_dirs, xpoints, x_min, x_max).")
    p.add_argument("--sweep-root", type=Path,
                   default=Path("data/DATA_stride5/sweep"))
    p.add_argument("--sweep-pattern", default="dt002_tauV*_xi*")
    p.add_argument("--fakeexp-root", type=Path,
                   default=Path("data/DATA_stride5/fakexp"))
    p.add_argument("--fakeexp-pattern", default="fakexp_*")
    p.add_argument("--out", type=Path, default=Path("outputs/time_recovery_wass"))
    p.add_argument("--queries-per-fake", type=int, default=20)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--tauV-step", type=int, default=10)
    p.add_argument("--xi-step", type=float, default=0.02)
    p.add_argument("--u-thresh", type=float, default=1e-4)
    p.add_argument("--s-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=-1, help="joblib workers for ECTs")
    p.add_argument("--gpu-batch", type=int, default=256,
                   help="references per GPU cost-kernel launch")
    p.add_argument("--ect-cache", type=Path, default=None,
                   help="reference-ECT cache npz; default <out>/ref_ects.npz")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ECT geometry from the bounds npz.
    if not args.bounds.exists():
        sys.exit(f"Bounds file not found: {args.bounds}")
    bd = np.load(str(args.bounds))
    args.n_dirs = int(bd["n_dirs"])
    args.xpoints = int(bd["xpoints"])
    args.x_min = float(bd["x_min"])
    args.x_max = float(bd["x_max"])
    delta_x = (args.x_max - args.x_min) / (args.xpoints - 1)
    print(f"[cfg] n_dirs={args.n_dirs} xpoints={args.xpoints} "
          f"x=[{args.x_min}, {args.x_max}] delta_x={delta_x:.6g} "
          f"(random directions, {{0,1,2}} labels, no crop/rescale/pad)")
    print(f"[cuda] {len(cuda.gpus)} GPU(s) visible; using current device")

    args.out.mkdir(parents=True, exist_ok=True)
    ect_cache = args.ect_cache or (args.out / "ref_ects.npz")

    runs = sorted(p for p in args.sweep_root.resolve().glob(args.sweep_pattern)
                  if p.is_dir())
    if not runs:
        sys.exit(f"No sweep runs under {args.sweep_root}/{args.sweep_pattern}")
    R, ref_tauV, ref_xi, ref_t, ref_label = build_reference_ects(
        runs, args, ect_cache)

    # Reference grids (for floors in plot_recovery_cdf) — written as
    # sweep_features.npz so the plot scripts find it unchanged.
    np.savez(str(args.out / "sweep_features.npz"),
             tauV=ref_tauV, xi_pct=ref_xi, t=ref_t, label=ref_label)

    fakedirs = sorted(p for p in args.fakeexp_root.resolve().glob(
        args.fakeexp_pattern) if p.is_dir())
    if not fakedirs:
        sys.exit(f"No fakexp dirs under {args.fakeexp_root}/{args.fakeexp_pattern}")

    # Build the 60 query ECTs (even-in-time sampling within each fakexp).
    from joblib import Parallel, delayed
    qjobs = []
    for fd in fakedirs:
        params = parse_label_params(fd.name)
        if params is None:
            print(f"  skip (no tauV/xi): {fd.name}"); continue
        true_tauV, true_xi = params
        for tq in even_sample(list_field_indices(fd), args.queries_per_fake):
            u, s = fd / f"u_{tq}.dat", fd / f"s_{tq}.dat"
            if u.exists() and s.exists():
                qjobs.append((fd.name, true_tauV, true_xi, tq, u, s))
    print(f"[queries] {len(qjobs)} query frames from {len(fakedirs)} fakexps")
    q_ects = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(frame_to_ect)(u, s, args, args.seed + 10_000_000 + k)
        for k, (_, _, _, _, u, s) in enumerate(qjobs))
    q_meta = [m[:4] for m, e in zip(qjobs, q_ects) if e is not None]
    q_arr = [e for e in q_ects if e is not None]
    print(f"[queries] featurised {len(q_arr)} frames")

    # GPU distance matrix and nearest-neighbour recovery.
    t0 = time.time()
    D = wasserstein_rows(R, q_arr, delta_x, args.gpu_batch)
    print(f"[match] {D.shape} distances in {time.time()-t0:.1f}s")

    rows_by_fake: Dict[str, List[Dict]] = {}
    for qi, (fake, true_tauV, true_xi, true_t) in enumerate(q_meta):
        j = int(np.argmin(D[qi]))
        pred_tau, pred_xi, pred_t = int(ref_tauV[j]), int(ref_xi[j]), int(ref_t[j])
        row = {
            "fakeexp": fake,
            "true_tauV": true_tauV, "true_xi_pct": true_xi, "true_t": true_t,
            "pred_tauV_l2": pred_tau, "pred_xi_pct_l2": pred_xi,
            "pred_t_l2": pred_t,
            "tauV_grid_err_l2": (pred_tau - true_tauV) / args.tauV_step,
            "xi_grid_err_l2": (pred_xi - true_xi) * 0.01 / args.xi_step,
            "t_grid_err_l2": (pred_t - true_t) / args.stride,
            "l2_dist": float(D[qi, j]),       # Wasserstein distance to the match
            # cos_* columns kept for schema parity; duplicate the W-NN result.
            "pred_tauV_cos": pred_tau, "pred_xi_pct_cos": pred_xi,
            "pred_t_cos": pred_t, "cos_sim": float(-D[qi, j]),
        }
        rows_by_fake.setdefault(fake, []).append(row)

    for fake, rows in rows_by_fake.items():
        out_csv = args.out / f"{fake}__time_recovery.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"[out] {out_csv}  ({len(rows)} rows)")

    print(f"\nDone. Figures:\n"
          f"  python scripts/plot_recovery_cdf.py --out {args.out}\n"
          f"  python scripts/plot_recovery_per_param.py --out {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
