"""Stage A1: draw the parameter grid and run the simulator.

Each host processes a contiguous slice of the deterministically generated
parameter array, writing per-simulation metadata and the simulator's
``u_*.dat`` / ``s_*.dat`` output. Featurisation happens in Stage A2.

    python -m scripts.abc_rej.generate_sims --start 0 --end 100 --jobs 25 \
        --scratch <scratch-dir>
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import (
    CACHE_DIR,
    CONFIG,
    Config,
    META_DIR,
    PER_HOST_MANIFEST_DIR,
    SIM_MANIFEST_COLS,
    make_thetas,
    DEFAULT_BOUNDS_FILE,
)
from .sim_worker import simulate_only


logger = logging.getLogger("abc_rej.generate")


# Manifest CSV I/O

def init_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SIM_MANIFEST_COLS).writeheader()


def append_manifest(path: Path, row: dict) -> None:
    """Append a row to a per-host manifest CSV."""
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SIM_MANIFEST_COLS)
        w.writerow({col: row.get(col, "") for col in SIM_MANIFEST_COLS})


def load_all_manifest_indices(per_host_dir: Path) -> dict:
    """Return {theta_idx: row_dict} from ALL per-host manifest files."""
    if not per_host_dir.is_dir():
        return {}
    out: dict = {}
    for path in sorted(per_host_dir.glob("manifest_*.csv")):
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        idx = int(row["theta_idx"])
                    except (KeyError, ValueError):
                        continue
                    out[idx] = row
        except Exception:  # pragma: no cover
            continue
    return out


def host_manifest_path(per_host_dir: Path) -> Path:
    """Per-host manifest file: cache/manifests/manifest_<hostname>.csv."""
    hostname = socket.gethostname().split(".")[0]
    return per_host_dir / f"manifest_{hostname}.csv"


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-sims", type=int, default=CONFIG.n_sims,
                   help="Total ABC rejection grid size (must match across hosts).")
    p.add_argument("--start", type=int, default=0,
                   help="First theta index this host should process (inclusive).")
    p.add_argument("--end", type=int, default=None,
                   help="Last theta index +1 this host should process. "
                        "Default: n_sims (process the whole range).")
    p.add_argument("--scratch", type=str, required=True,
                   help="Per-host scratch root (e.g. /scratch/abc_rej).")
    p.add_argument("--cache-dir", type=str, default=str(CACHE_DIR),
                   help="NFS-shared cache directory for per-sim NPZs.")
    p.add_argument("--manifest-dir", type=str, default=str(PER_HOST_MANIFEST_DIR),
                   help="Directory holding per-host manifest CSVs "
                        "(NFS-safe; one file per hostname).")
    p.add_argument("--bounds", type=str, default=str(DEFAULT_BOUNDS_FILE),
                   help="Path to data/sampeuler_bounds.npz (unused in "
                        "Stage A1 — Stage A2 reads this for featurisation).")
    p.add_argument("--jobs", type=int, default=CONFIG.n_jobs,
                   help="Per-host worker pool size.")
    p.add_argument("--n-cells", type=int, default=CONFIG.n_cells)
    p.add_argument("--rs", type=float, default=CONFIG.rs)
    p.add_argument("--no-retry-errors", action="store_true",
                   help="Don't re-attempt indices whose manifest row has a non-empty error.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned (theta_idx, tauV, xi) triples and exit.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Pin BLAS to a single thread per worker; the parallelism here is at
    # the worker level, not inside numpy. Without this, 25 workers x 8
    # BLAS threads = 200 threads on 25 cores -> heavy oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    args = parse_args()

    cfg = CONFIG
    end = args.end if args.end is not None else cfg.n_sims
    if not (0 <= args.start < end <= args.n_sims):
        sys.exit(f"Invalid index range: start={args.start} end={end} "
                 f"n_sims={args.n_sims}")
    if args.n_sims != cfg.n_sims:
        # Override n_sims via the CLI; rebuild thetas with the override.
        import dataclasses
        cfg = dataclasses.replace(cfg, n_sims=args.n_sims)

    scratch_root = Path(args.scratch)
    scratch_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta").mkdir(parents=True, exist_ok=True)
    per_host_manifest_dir = Path(args.manifest_dir)
    per_host_manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = host_manifest_path(per_host_manifest_dir)
    init_manifest(manifest)

    # Generate the full theta array deterministically; every host shares this seed.
    thetas = make_thetas(args.n_sims, cfg)
    if thetas.shape != (args.n_sims, 2):
        sys.exit(f"make_thetas produced unexpected shape {thetas.shape}")

    # Build the work list for this host's index slice.
    already = load_all_manifest_indices(per_host_manifest_dir)
    work: List[Tuple[int, float, float]] = []
    meta_dir = cache_dir / "meta"
    for i in range(args.start, end):
        meta_path = meta_dir / f"sim_meta_{i:05d}.npz"
        if meta_path.exists():
            continue  # already produced by us or another host
        if i in already and already[i].get("error", ""):
            if args.no_retry_errors:
                continue
        work.append((i, float(thetas[i, 0]), float(thetas[i, 1])))

    logger.info("host slice [%d, %d): %d total, %d to process (rest already cached)",
                args.start, end, end - args.start, len(work))

    if args.dry_run:
        for idx, tv, xi in work[:10]:
            print(f"  idx={idx}  tauV={tv:.4f}  xi={xi:.4f}")
        if len(work) > 10:
            print(f"  ... and {len(work) - 10} more")
        return
    if not work:
        logger.info("Nothing to do.")
        return

    # ── Dispatch via ProcessPoolExecutor ──
    n_workers = max(1, args.jobs)
    n_ok = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for idx, tv, xi in work:
            fut = ex.submit(
                simulate_only,
                idx, tv, xi,
                str(scratch_root),
                str(cache_dir),
                idx % n_workers,         # stable worker_id from theta_idx
                n_cells=args.n_cells,
                rs=args.rs,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # pragma: no cover
                logger.exception("worker for idx=%d crashed: %s", idx, e)
                row = {"theta_idx": idx, "error": f"future crash: {e}"}
            append_manifest(manifest, row)
            if row.get("error"):
                n_err += 1
                logger.warning("idx=%d FAILED: %s", idx, row["error"])
            else:
                n_ok += 1
                logger.info(
                    "idx=%4d tauV=%.2f xi=%.3f T_end=%d n_frames=%d wall=%.1fmin",
                    idx, row["tauV"], row["xi"], row["T_end"], row["n_frames"],
                    row.get("wall_seconds", 0.0) / 60.0,
                )

    logger.info("done: %d ok, %d failed", n_ok, n_err)


if __name__ == "__main__":
    main()
