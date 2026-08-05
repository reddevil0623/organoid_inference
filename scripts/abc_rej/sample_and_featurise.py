"""Stage A2: pool-sample the cached frames, then featurise the sampled ones.

``--plan`` reads the Stage A1 metadata, draws ``--n-pool-samples`` frames
uniformly without replacement, and writes a per-host work list. ``--featurise``
runs on each host and computes ECT + SampEuler for the frames in its own list.
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import (
    CACHE_DIR,
    CONFIG,
    DEFAULT_BOUNDS_FILE,
    META_DIR,
    WORK_LIST_DIR,
)
from .sim_worker import featurise_sampled


logger = logging.getLogger("abc_rej.sample")


# Pool building

def build_global_pool(meta_dir: Path) -> Dict[str, np.ndarray]:
    """Scan every ``sim_meta_*.npz``; build parallel flat arrays."""
    paths = sorted(meta_dir.glob("sim_meta_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No metadata NPZs in {meta_dir}")

    theta_chunks: List[np.ndarray] = []
    frame_chunks: List[np.ndarray] = []
    t_chunks: List[np.ndarray] = []
    tauV_chunks: List[np.ndarray] = []
    xi_chunks: List[np.ndarray] = []
    host_chunks: List[List[str]] = []
    outdir_chunks: List[List[str]] = []
    for p in paths:
        try:
            blob = np.load(p, allow_pickle=False)
            theta_idx = int(blob["theta_idx"])
            t_indices = np.asarray(blob["t_indices"], dtype=np.int64)
            tauV = float(blob["tauV"])
            xi = float(blob["xi"])
            hostname = str(blob["hostname"])
            outdir = str(blob["outdir"])
        except Exception as e:
            logger.warning("skip %s: %s", p.name, e)
            continue
        n = int(len(t_indices))
        if n == 0:
            continue
        theta_chunks.append(np.full(n, theta_idx, dtype=np.int64))
        frame_chunks.append(np.arange(n, dtype=np.int64))
        t_chunks.append(t_indices)
        tauV_chunks.append(np.full(n, tauV, dtype=np.float64))
        xi_chunks.append(np.full(n, xi, dtype=np.float64))
        host_chunks.append([hostname] * n)
        outdir_chunks.append([outdir] * n)

    if not theta_chunks:
        raise RuntimeError("Empty pool — no usable metadata NPZs.")
    return {
        "theta_idx": np.concatenate(theta_chunks),
        "frame_idx": np.concatenate(frame_chunks),
        "t": np.concatenate(t_chunks),
        "tauV": np.concatenate(tauV_chunks),
        "xi": np.concatenate(xi_chunks),
        "hostname": np.array([h for chunk in host_chunks for h in chunk]),
        "outdir": np.array([o for chunk in outdir_chunks for o in chunk]),
    }


def sample_pool(pool_size: int, n_samples: int, seed: int) -> np.ndarray:
    """Return a sorted index array selecting at most ``n_samples`` from the pool."""
    if n_samples >= pool_size:
        return np.arange(pool_size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(pool_size, size=int(n_samples), replace=False)
    sel.sort()
    return sel


def write_work_lists(pool: Dict[str, np.ndarray],
                     sel: np.ndarray,
                     work_lists_dir: Path) -> Dict[str, Tuple[Path, int]]:
    """Group sampled pool entries by hostname; write per-host NPZs."""
    work_lists_dir.mkdir(parents=True, exist_ok=True)

    sel_theta_idx = pool["theta_idx"][sel]
    sel_frame_idx = pool["frame_idx"][sel]
    sel_t = pool["t"][sel]
    sel_tauV = pool["tauV"][sel]
    sel_xi = pool["xi"][sel]
    sel_hostname = pool["hostname"][sel]
    sel_outdir = pool["outdir"][sel]

    by_host: Dict[str, Tuple[Path, int]] = {}
    for host in np.unique(sel_hostname):
        mask = sel_hostname == host
        out_path = work_lists_dir / f"work_{host}.npz"
        # Atomic write via tmp-then-replace to avoid partial-file races if
        # --plan is re-launched while a worker is mid-read of an old list.
        # Tmp filename MUST end in '.npz' (np.savez_compressed auto-appends).
        tmp_path = out_path.with_suffix(
            f".tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}.npz"
        )
        try:
            np.savez_compressed(
                tmp_path,
                theta_idx=sel_theta_idx[mask],
                frame_idx=sel_frame_idx[mask],
                t=sel_t[mask],
                tauV=sel_tauV[mask],
                xi=sel_xi[mask],
                outdir=sel_outdir[mask],
            )
            os.replace(tmp_path, out_path)
        except Exception:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        by_host[str(host)] = (out_path, int(mask.sum()))
    return by_host


# Controller mode: --plan

def run_planner(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    meta_dir = cache_dir / "meta"
    work_lists_dir = cache_dir / "work_lists"

    pool = build_global_pool(meta_dir)
    pool_size = int(len(pool["theta_idx"]))
    n_sims = int(len(np.unique(pool["theta_idx"])))
    logger.info("pool: %d frames across %d sims", pool_size, n_sims)

    sel = sample_pool(pool_size, args.n_pool_samples, args.pool_seed)
    logger.info("sampled %d / %d frames (seed=%d, fraction=%.3f)",
                len(sel), pool_size, args.pool_seed,
                len(sel) / max(1, pool_size))

    by_host = write_work_lists(pool, sel, work_lists_dir)
    for host, (path, n) in sorted(by_host.items()):
        logger.info("  host %-14s  n=%5d  -> %s", host, n, path)
    logger.info("wrote work lists for %d hosts", len(by_host))


# Worker mode: --featurise

def run_featurise_worker(args: argparse.Namespace) -> None:
    hostname = args.my_host or socket.gethostname().split(".")[0]
    cache_dir = Path(args.cache_dir)
    work_lists_dir = cache_dir / "work_lists"
    work_path = work_lists_dir / f"work_{hostname}.npz"
    if not work_path.exists():
        sys.exit(f"No work list at {work_path}; run --plan first.")

    blob = np.load(work_path, allow_pickle=False)
    theta_idx = np.asarray(blob["theta_idx"], dtype=np.int64)
    t = np.asarray(blob["t"], dtype=np.int64)
    outdir = np.asarray(blob["outdir"])

    unique_thetas = np.unique(theta_idx)
    logger.info("host %s: %d unique sims, %d frames total",
                hostname, len(unique_thetas), len(theta_idx))

    # Group (t-list, outdir) per theta_idx so we featurise each sim once.
    work_items: List[Tuple[int, List[int], str]] = []
    for theta in unique_thetas:
        mask = theta_idx == theta
        ts = sorted(int(x) for x in t[mask].tolist())
        od = str(outdir[mask][0])
        work_items.append((int(theta), ts, od))

    n_workers = max(1, int(args.jobs))
    n_ok = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for theta, ts, od in work_items:
            fut = ex.submit(
                featurise_sampled,
                theta, ts, od, str(cache_dir),
                u_thresh=CONFIG.u_thresh,
                s_thresh=CONFIG.s_thresh,
                canvas_side=args.canvas_side,
                bounds_path=args.bounds,
            )
            futures[fut] = theta

        for fut in as_completed(futures):
            theta = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # pragma: no cover
                logger.exception("featurise idx=%d crashed: %s", theta, e)
                row = {"error": f"crash: {e}", "n_frames": 0,
                       "npz_path": "", "theta_idx": theta}
            if row.get("error"):
                n_err += 1
                logger.warning("idx=%d FAILED: %s", theta, row["error"])
            else:
                n_ok += 1
                logger.info("idx=%4d n_frames=%d -> %s",
                            theta, row["n_frames"], row.get("npz_path", ""))

    logger.info("done: %d ok, %d failed", n_ok, n_err)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=str, default=str(CACHE_DIR),
                   help="NFS-shared cache directory (contains meta/ and "
                        "work_lists/ sub-dirs).")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true",
                      help="Controller mode: build pool, sample, write "
                           "per-host work lists.")
    mode.add_argument("--featurise", action="store_true",
                      help="Worker mode: read this host's work list and "
                           "featurise the assigned sampled frames.")

    p.add_argument("--my-host", type=str, default=None,
                   help="Override the hostname autodetection in "
                        "--featurise mode. Useful for testing.")
    p.add_argument("--n-pool-samples", type=int, default=CONFIG.n_pool_samples,
                   help="Target number of (theta_idx, frame_idx) samples "
                        "to draw from the pool (default 20 000). Used in "
                        "--plan only.")
    p.add_argument("--pool-seed", type=int, default=CONFIG.pool_seed,
                   help="RNG seed for the pool draw. Used in --plan only.")
    p.add_argument("--bounds", type=str, default=str(DEFAULT_BOUNDS_FILE),
                   help="Path to data/sampeuler_bounds.npz (used in "
                        "--featurise mode).")
    p.add_argument("--canvas-side", type=int, default=1024,
                   help="Padded canvas size for SampEuler featurisation "
                        "(used in --featurise mode).")
    p.add_argument("--jobs", type=int, default=CONFIG.n_jobs,
                   help="Per-host worker pool size for --featurise mode.")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Pin BLAS to a single thread per worker, matching generate_sims.py.
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    args = parse_args()
    if args.plan:
        run_planner(args)
    else:
        run_featurise_worker(args)


if __name__ == "__main__":
    main()
