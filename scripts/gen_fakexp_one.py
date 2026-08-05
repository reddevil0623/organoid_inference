#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one synthetic test observation at a chosen (tau_V, xi).

Runs the phase-field simulator and copies the resulting frame set to the target
directory. One simulation per invocation. Idempotent: does nothing if the
destination already holds a completed run for this index.

    python scripts/gen_fakexp_one.py --idx 3 --tauV 25 --xi 0.15 \
        --scratch <scratch-dir> --dest <data-dir>/fakexp
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as _p          # noqa: E402
_p.ensure_eucalc()          # core imports eucalc at module load
from core import Simulator, xi_tag  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--tauV", type=int, required=True)
    ap.add_argument("--xi", type=float, required=True)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--rs", type=float, default=0.70)
    ap.add_argument("--scratch", default="/scratch/abc_fakexp",
                    help="Scratch root for the simulator run.")
    ap.add_argument("--dest",
                    default=str(_p.PROJECT / "data" / "DATA_stride5" / "fakexp"))
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    # Idempotent resume: a completed dest dir for this idx -> nothing to do.
    for d in dest_root.glob(f"fakexp_{args.idx}_*"):
        if (d / ".done").exists() and any(d.glob("u_*.dat")):
            print(f"[gen] idx={args.idx} already complete at {d.name}; skip",
                  flush=True)
            return 0

    label = (f"fakexp_{args.idx}_tauV{args.tauV}_xi{xi_tag(args.xi)}_"
             f"{uuid.uuid4().hex[:6]}")
    host = os.uname().nodename.split(".")[0]
    print(f"[gen] idx={args.idx} {label}: tauV={args.tauV} xi={args.xi:.2f} "
          f"rs={args.rs} n_cells={args.n_cells} host={host}", flush=True)

    sim = Simulator(model_repo=_p.MODEL_REPO, scratch=Path(args.scratch))
    t0 = time.time()
    outdir = sim.run_checked(label, args.n_cells, args.tauV, args.xi, args.rs)
    dt_h = (time.time() - t0) / 3600.0
    print(f"[gen] idx={args.idx} sim done in {dt_h:.2f} h -> {outdir}",
          flush=True)

    # Copy the full frame set + metadata to NFS; .done last so a reader never
    # sees the marker before the frames.
    dst = dest_root / label
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in outdir.iterdir():
        if not f.is_file() or f.name == ".done":
            continue
        if f.suffix == ".dat" or f.name in ("param.txt", "numofcell.dat", "out"):
            shutil.copy2(f, dst / f.name)
            n += 1
    if (outdir / ".done").exists():
        shutil.copy2(outdir / ".done", dst / ".done")
    print(f"[gen] idx={args.idx} copied {n} files -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
