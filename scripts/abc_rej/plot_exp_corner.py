"""3-parameter corner plots for the experimental organoids.

Marks the posterior mean rather than the single best-matching candidate, which
is high variance. No ground-truth cross: experimental images have no truth.
"""
from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()
from .config import CONFIG, prior_box
from .plot_posterior_3p import plot_corner


def block_key(s: str) -> str:
    m = re.search(r"block[_\s]*(\d+)", s, re.I)
    return m.group(1) if m else s


def bbox_crop(lbl: np.ndarray, pad: int = 6) -> np.ndarray:
    ys, xs = np.where(lbl > 0)
    if ys.size == 0:
        return lbl
    y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    y1, x1 = min(lbl.shape[0] - 1, ys.max() + pad), min(lbl.shape[1] - 1, xs.max() + pad)
    return lbl[y0:y1 + 1, x0:x1 + 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dist-dir", default="outputs/ABC_REJ/distances_exp")
    p.add_argument("--img-dir", default="data/experimental_images")
    p.add_argument("--quantile", type=float, default=0.01)
    p.add_argument("--out-dir", default="draft/figures_updated/exp_corners")
    args = p.parse_args()

    tifs = {block_key(Path(t).stem): t for t in glob.glob(str(Path(args.img_dir) / "*.tif"))}
    # First pass: load all, find a COMMON t-axis range so the 10 corners are comparable.
    rows = []
    for csv in sorted(glob.glob(str(Path(args.dist_dir) / "*__distances.csv"))):
        lbl = Path(csv).name[: -len("__distances.csv")]
        df = pd.read_csv(csv)
        acc = df[df["distance"] <= df["distance"].quantile(args.quantile)]
        # Posterior mean (marginal means) -- standard Bayesian point estimate
        # (posterior expectation); stable, unlike the single best match.
        mean_pt = (float(np.mean(acc["tauV"])), float(np.mean(acc["xi"])),
                   float(np.mean(acc["t"])))
        img = None
        tif = tifs.get(block_key(lbl))
        if tif:
            img = bbox_crop(np.array(Image.open(tif)).astype(np.uint8))
        rows.append((lbl, acc, mean_pt, img))

    t_top = max((float(r[1]["t"].max()) for r in rows if len(r[1])), default=1.0)
    t_range = (0.0, max(1.0, t_top * 1.05))

    for lbl, acc, mean_pt, img in rows:
        plot_corner(acc["tauV"].values, acc["xi"].values, acc["t"].values,
                    prior=prior_box(CONFIG), t_range=t_range, truth=None, label=lbl,
                    out_path=Path(args.out_dir) / f"{lbl}__corner3p.png",
                    image=img, ref=mean_pt)
        print("  corner:", lbl, "block", block_key(lbl))


if __name__ == "__main__":
    main()
