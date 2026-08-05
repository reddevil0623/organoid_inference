"""ABC rejection distances for one experimental image.

Applies the same size normalisation used to build the diameter-normalised
cache, computes the image's SampEuler curves, then the Wasserstein distance to
every cached frame. Writes ``<label>__distances.csv``.

    python -m scripts.abc_rej.exp_distances \
        --exp-image data/experimental_images/<name>.tif \
        --cache-dir outputs/ABC_REJ/cache_diamnorm --jobs 25
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import CONFIG, DEFAULT_BOUNDS_FILE, load_sampeuler_bounds, OUTPUTS_DIR
from .sim_worker import compute_ect_curves
from .featurise_diamnorm import normalize_to_diameter

logger = logging.getLogger("abc_rej.exp_distances")


def exp_label(path: str) -> str:
    """Short, filesystem-safe label from a long experimental filename."""
    stem = Path(path).stem
    stem = re.sub(r"_rescaled_predicted_merged$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return "exp_" + stem


def featurise_exp(path: str, diameter: float, canvas: int, bounds: dict) -> np.ndarray:
    from core import ECTComputer
    from PIL import Image
    lab = np.array(Image.open(path)).astype(np.uint8)
    if set(np.unique(lab).tolist()) - {0, 1, 2}:
        raise ValueError(f"{path}: expected {{0,1,2}} labels, got {np.unique(lab)}")
    lab = normalize_to_diameter(lab, diameter, canvas)
    ect = ECTComputer(n_dirs=int(bounds["n_dirs"]), xpoints=int(bounds["xpoints"]),
                      x_min=float(bounds["x_min"]), x_max=float(bounds["x_max"]))
    return compute_ect_curves(lab, ect=ect, bounds=bounds).astype(np.float32)


def _score_npz(npz_path: str, obs_ect: np.ndarray, delta_x: float):
    _ensure_scripts_on_path()
    from core import WassersteinDistance
    wd = WassersteinDistance(delta_x, method="hungarian")
    b = np.load(npz_path, allow_pickle=False)
    if "ect_curves" not in b.files:
        return []
    th = int(b["theta_idx"]); tauV = float(b["tauV"]); xi = float(b["xi"])
    ts = np.asarray(b["t"], dtype=np.int64)
    curves = b["ect_curves"]                       # (n_frames, n_dirs, xpoints)
    out = []
    for i in range(len(ts)):
        d = float(wd([obs_ect], [curves[i]]))
        out.append((th, i, tauV, xi, int(ts[i]), d))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-image", default=None,
                   help="Path to one experimental .tif (or use --exp-index).")
    p.add_argument("--exp-index", type=int, default=None,
                   help="Pick the i-th sorted image from --exp-dir (avoids paths with spaces).")
    p.add_argument("--exp-dir", default="data/experimental_images")
    p.add_argument("--cache-dir", default=str(OUTPUTS_DIR / "cache_diamnorm"))
    p.add_argument("--out-dir", default=str(OUTPUTS_DIR / "distances_exp"))
    p.add_argument("--diameter", type=float, default=400.0)
    p.add_argument("--canvas", type=int, default=512)
    p.add_argument("--jobs", type=int, default=CONFIG.n_jobs)
    p.add_argument("--bounds", default=str(DEFAULT_BOUNDS_FILE))
    args = p.parse_args()

    exp_image = args.exp_image
    if exp_image is None:
        if args.exp_index is None:
            sys.exit("Need --exp-image or --exp-index")
        imgs = sorted(glob.glob(str(Path(args.exp_dir) / "*.tif")))
        if not (0 <= args.exp_index < len(imgs)):
            sys.exit(f"--exp-index {args.exp_index} out of range (0..{len(imgs)-1})")
        exp_image = imgs[args.exp_index]

    bounds = load_sampeuler_bounds(Path(args.bounds))
    delta_x = (float(bounds["x_max"]) - float(bounds["x_min"])) / max(1, int(bounds["xpoints"]) - 1)
    label = exp_label(exp_image)
    logger.info("featurising observation %s (D=%.0f, canvas=%d) -> %s",
                exp_image, args.diameter, args.canvas, label)
    obs_ect = featurise_exp(exp_image, args.diameter, args.canvas, bounds)

    npzs = sorted(glob.glob(str(Path(args.cache_dir) / "abcrej_*.npz")))
    if not npzs:
        sys.exit(f"No cache NPZs in {args.cache_dir}")
    logger.info("scoring %s against %d cache sims", label, len(npzs))

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_score_npz, p, obs_ect, delta_x): p for p in npzs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 500 == 0:
                logger.info("  scored %d / %d sims", done, len(npzs))

    rows.sort(key=lambda r: r[5])  # by distance ascending
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}__distances.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta_idx", "frame_idx", "tauV", "xi", "t", "distance"])
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
