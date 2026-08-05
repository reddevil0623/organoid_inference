"""Trajectory featurisation: n even-in-time frames per simulation.

Samples ``--n-frames`` model times evenly across each simulation's trajectory
and writes ``traj_<theta_idx>.npz`` holding the aligned ECT curves. Read by
``abc_rej_distance_cuda.py --mode temporal``.

    python -m scripts.abc_rej.featurise_temporal --n-frames 10 \
        --out-dir outputs/ABC_REJ/cache_temporal --jobs 25
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import META_DIR, DEFAULT_BOUNDS_FILE, load_sampeuler_bounds


def even_sample(ts: List[int], n: int) -> List[int]:
    ts = sorted(int(x) for x in ts)
    if not ts or n >= len(ts):
        return ts
    idx = np.linspace(0, len(ts) - 1, num=n)
    return sorted({ts[int(round(x))] for x in idx})


def _featurise_one(theta_idx, outdir, t_list, tauV, xi, bounds, out_dir, seed):
    """Featurise one sim's n-frame trajectory -> traj_<idx>.npz. Module-level
    for ProcessPoolExecutor."""
    _ensure_scripts_on_path()
    from core import ECTComputer, load_field, u_s_to_labels
    from .sim_worker import compute_ect_curves  # {0,1,2} ECT, random dirs

    npz = Path(out_dir) / f"traj_{theta_idx:05d}.npz"
    sorted_t = sorted(int(t) for t in t_list)
    # Idempotent: reuse if the cached trajectory has exactly these times.
    if npz.exists():
        try:
            c = np.load(npz, allow_pickle=False)
            if sorted(np.asarray(c["t"]).astype(int).tolist()) == sorted_t:
                return (int(theta_idx), int(c["n_frames"]), "")
        except Exception:
            pass

    ect = ECTComputer(n_dirs=int(bounds["n_dirs"]), xpoints=int(bounds["xpoints"]),
                      x_min=float(bounds["x_min"]), x_max=float(bounds["x_max"]))
    rng = np.random.default_rng(int(seed) + int(theta_idx))
    outdir = Path(outdir)
    curves, ts = [], []
    for t in sorted_t:
        u, s = outdir / f"u_{t}.dat", outdir / f"s_{t}.dat"
        if not (u.exists() and s.exists()):
            continue
        L = u_s_to_labels(load_field(u), load_field(s))
        curves.append(compute_ect_curves(L, ect=ect, bounds=bounds, rng=rng))
        ts.append(int(t))
    if not curves:
        return (int(theta_idx), 0, "no frames featurised")

    arr = np.stack(curves).astype(np.float32)   # (n, n_dirs, xpoints)
    tmp = npz.with_suffix(f".tmp.{socket.gethostname().split('.')[0]}."
                          f"{os.getpid()}.npz")
    try:
        np.savez_compressed(
            tmp, ect_curves=arr, t=np.array(ts, np.int64),
            tauV=np.float64(tauV), xi=np.float64(xi),
            theta_idx=np.int64(theta_idx), n_frames=np.int64(len(ts)),
            n_dirs=np.int64(bounds["n_dirs"]), xpoints=np.int64(bounds["xpoints"]),
            x_min=np.float64(bounds["x_min"]), x_max=np.float64(bounds["x_max"]),
            delta_x=np.float64((float(bounds["x_max"]) - float(bounds["x_min"]))
                               / max(1, int(bounds["xpoints"]) - 1)))
        os.replace(tmp, npz)
    except Exception as e:
        try:
            tmp.unlink()
        except Exception:
            pass
        return (int(theta_idx), 0, f"save failed: {e}")
    return (int(theta_idx), len(ts), "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-frames", type=int, default=10)
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/ABC_REJ/cache_temporal"))
    p.add_argument("--meta-dir", type=Path, default=META_DIR)
    p.add_argument("--bounds", type=Path, default=DEFAULT_BOUNDS_FILE)
    p.add_argument("--jobs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--my-host", type=str, default=None,
                   help="Override hostname filter (default: this host).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    host = args.my_host or socket.gethostname().split(".")[0]
    bounds = load_sampeuler_bounds(args.bounds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metas = sorted(glob.glob(str(args.meta_dir / "sim_meta_[0-9]*.npz")))
    jobs = []
    for mp in metas:
        try:
            m = np.load(mp, allow_pickle=False)
        except Exception:
            continue
        if str(m["hostname"]) != host:
            continue
        t_list = even_sample(np.asarray(m["t_indices"]).astype(int).tolist(),
                             args.n_frames)
        if not t_list:
            continue
        jobs.append((int(m["theta_idx"]), str(m["outdir"]), t_list,
                     float(m["tauV"]), float(m["xi"])))
    print(f"[temporal] host={host}: {len(jobs)} sims, n_frames={args.n_frames}, "
          f"jobs={args.jobs}", flush=True)
    if not jobs:
        print("[temporal] no sims for this host; nothing to do.")
        return 0

    t0 = time.time()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_featurise_one, ti, od, tl, tv, x, bounds,
                          str(args.out_dir), args.seed): ti
                for (ti, od, tl, tv, x) in jobs}
        done = 0
        for f in as_completed(futs):
            ti = futs[f]
            try:
                idx, n, err = f.result()
            except Exception as e:
                fail += 1
                print(f"[temporal] idx={ti} crash: {e}", file=sys.stderr)
                continue
            if err:
                fail += 1
                print(f"[temporal] idx={idx} FAILED: {err}", file=sys.stderr)
            else:
                ok += 1
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} in {time.time()-t0:.1f}s "
                      f"({ok} ok, {fail} failed)", flush=True)
    print(f"done: {ok} ok, {fail} failed in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
