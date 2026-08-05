"""Size-normalised featurisation, so experimental images are comparable to simulations.

Builds a second feature cache in which every frame is rescaled to a common
organoid diameter, then cropped and centre-padded to a common canvas.

    python -m scripts.abc_rej.featurise_diamnorm --diameter 400 --canvas 512 --jobs 25
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import CACHE_DIR, CONFIG, DEFAULT_BOUNDS_FILE, load_sampeuler_bounds
from .sim_worker import compute_ect_curves

logger = logging.getLogger("abc_rej.featurise_diamnorm")


def normalize_to_diameter(label_img: np.ndarray, target_diameter: float,
                          canvas_side: int) -> np.ndarray:
    """Rescale so organoid MEC diameter == target_diameter; bbox-crop; centre-pad."""
    from core import rescale_to_diameter, nonzero_bbox, pad_to, resize_nearest
    r = rescale_to_diameter(label_img, float(target_diameter))
    bb = nonzero_bbox(r)
    crop = r[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1] if bb is not None else r
    h, w = crop.shape
    if h > canvas_side or w > canvas_side:
        sc = canvas_side / max(h, w)
        crop = resize_nearest(crop, max(1, int(w * sc)), max(1, int(h * sc)))
    return pad_to(crop, canvas_side, canvas_side, center=True)


def normalize_by_scale(label_img: np.ndarray, scale: float, canvas_side: int) -> np.ndarray:
    """Rescale by a FIXED factor (set once per sim from its final frame),
    bbox-crop, centre-pad. Preserves the organoid's size *relative* to the
    final frame, so within-trajectory growth (and thus age) is retained."""
    from core import nonzero_bbox, resize_nearest, pad_to
    bb = nonzero_bbox(label_img)
    crop = label_img[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1] if bb is not None else label_img
    h, w = crop.shape
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if max(nh, nw) > canvas_side:                       # never exceed the canvas
        f = canvas_side / max(nh, nw)
        nh, nw = max(1, int(nh * f)), max(1, int(nw * f))
    crop = resize_nearest(crop, nw, nh)
    return pad_to(crop, canvas_side, canvas_side, center=True)


def featurise_one(theta_idx: int, ts: "List[int]", outdir: str,
                  meta_dir: str, out_root: str, diameter: float, canvas: int,
                  u_thresh: float, s_thresh: float, bounds_path: str,
                  mode: str = "perframe") -> dict:
    """Diameter-normalise + ECT the requested frames of one sim; write NPZ."""
    _ensure_scripts_on_path()
    from core import ECTComputer, load_field, u_s_to_labels

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    npz_path = out_root / f"abcrej_{theta_idx:05d}.npz"
    sorted_t = sorted(int(t) for t in ts)

    # Idempotency: skip if already done with the same frames.
    if npz_path.exists():
        try:
            cached = np.load(npz_path, allow_pickle=False)
            if sorted(np.asarray(cached["t"], dtype=np.int64).tolist()) == sorted_t:
                return {"theta_idx": theta_idx, "n_frames": int(cached["n_frames"]),
                        "error": ""}
        except Exception:
            pass

    meta = np.load(Path(meta_dir) / f"sim_meta_{theta_idx:05d}.npz", allow_pickle=False)
    tauV, tauV_int, xi = float(meta["tauV"]), int(meta["tauV_int"]), float(meta["xi"])
    T_end = int(meta["T_end"])

    bounds = load_sampeuler_bounds(Path(bounds_path))
    ect = ECTComputer(n_dirs=int(bounds["n_dirs"]), xpoints=int(bounds["xpoints"]),
                      x_min=float(bounds["x_min"]), x_max=float(bounds["x_max"]))

    od = Path(outdir)
    # finalframe mode: one scale per sim, fixed by the final (most mature) frame.
    sim_scale = None
    if mode == "finalframe":
        from core import organoid_diameter
        Uf = load_field(od / f"u_{T_end}.dat")
        Sf = load_field(od / f"s_{T_end}.dat")
        d_final = organoid_diameter(u_s_to_labels(Uf, Sf, u_thresh, s_thresh))
        sim_scale = (diameter / d_final) if d_final > 0 else 1.0

    ect_list: List[np.ndarray] = []
    kept: List[int] = []
    for t in sorted_t:
        try:
            U = load_field(od / f"u_{t}.dat")
            S = load_field(od / f"s_{t}.dat")
            lab = u_s_to_labels(U, S, u_thresh, s_thresh)
            if mode == "finalframe":
                lab = normalize_by_scale(lab, sim_scale, canvas)
            else:
                lab = normalize_to_diameter(lab, diameter, canvas)
            curves = compute_ect_curves(lab, ect=ect, bounds=bounds)
        except Exception as e:
            print(f"[diamnorm idx={theta_idx} t={t}] failed: {e}", file=sys.stderr)
            continue
        ect_list.append(curves)
        kept.append(int(t))

    if not ect_list:
        return {"theta_idx": theta_idx, "n_frames": 0, "error": "no frames featurised"}

    ect_arr = np.stack(ect_list).astype(np.float32)
    tmp = npz_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        tmp, ect_curves=ect_arr, t=np.array(kept, dtype=np.int64),
        tauV=np.float64(tauV), tauV_int=np.int64(tauV_int), xi=np.float64(xi),
        T_end=np.int64(T_end), n_frames=np.int64(len(kept)),
        theta_idx=np.int64(theta_idx), n_dirs=np.int64(bounds["n_dirs"]),
        xpoints=np.int64(bounds["xpoints"]), x_min=np.float64(bounds["x_min"]),
        x_max=np.float64(bounds["x_max"]), delta_x=np.float64(ect.delta_x),
        diameter=np.float64(diameter), canvas=np.int64(canvas))
    os.replace(tmp, npz_path)
    return {"theta_idx": theta_idx, "n_frames": len(kept), "error": ""}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=str(CACHE_DIR),
                   help="Original cache (work_lists/ and meta/ live here).")
    p.add_argument("--out-dir", default=None,
                   help="Where the diameter-normalised NPZs go (default: <cache-dir>/../cache_diamnorm).")
    p.add_argument("--diameter", type=float, default=400.0)
    p.add_argument("--canvas", type=int, default=512)
    p.add_argument("--mode", choices=("perframe", "finalframe"), default="perframe",
                   help="perframe: every frame -> diameter D (size removed). "
                        "finalframe: one scale per sim from its final frame "
                        "(within-trajectory size/age preserved).")
    p.add_argument("--my-host", default=None)
    p.add_argument("--jobs", type=int, default=CONFIG.n_jobs)
    p.add_argument("--bounds", default=str(DEFAULT_BOUNDS_FILE))
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    default_sub = "cache_finalnorm" if args.mode == "finalframe" else "cache_diamnorm"
    out_root = Path(args.out_dir) if args.out_dir else cache_dir.parent / default_sub
    meta_dir = cache_dir / "meta"
    host = args.my_host or socket.gethostname().split(".")[0]
    work_path = cache_dir / "work_lists" / f"work_{host}.npz"
    if not work_path.exists():
        sys.exit(f"No work list at {work_path}")

    b = np.load(work_path, allow_pickle=False)
    theta = np.asarray(b["theta_idx"], dtype=np.int64)
    t = np.asarray(b["t"], dtype=np.int64)
    outdir = np.asarray(b["outdir"])
    work = []
    for th in np.unique(theta):
        m = theta == th
        work.append((int(th), sorted(int(x) for x in t[m].tolist()), str(outdir[m][0])))
    logger.info("host %s: %d sims -> %s (mode=%s, D=%.0f, canvas=%d)",
                host, len(work), out_root, args.mode, args.diameter, args.canvas)

    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(featurise_one, th, ts, od, str(meta_dir), str(out_root),
                          args.diameter, args.canvas, CONFIG.u_thresh,
                          CONFIG.s_thresh, args.bounds, args.mode): th
                for th, ts, od in work}
        for fut in as_completed(futs):
            row = fut.result()
            if row.get("error"):
                n_err += 1
                logger.warning("idx=%d FAILED: %s", futs[fut], row["error"])
            else:
                n_ok += 1
    logger.info("done: %d ok, %d failed", n_ok, n_err)


if __name__ == "__main__":
    main()
