"""3-parameter (tau_V, xi, t) corner plot from a distance file.

Applies the ABC rejection acceptance quantile and plots the joint posterior as
filled KDE contours, with 1-D marginals on the diagonal.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import CONFIG, DIST_DIR, POST_DIR, prior_box
from .observations import FAKEXP_TRUTH
from .plot_posterior import load_distances, find_distances_file

logger = logging.getLogger("abc_rej.plot3p")

TRUTH_KW = dict(marker="x", color="red", s=200, linewidths=3, zorder=10,
                clip_on=False)
GOLD = "#e8a800"   # reference-point (best-match) marker colour

# Font-size multiplier: these corner plots are placed at ~0.32\textwidth in the
# PLOS template (a large downscale), so bump all font sizes to stay legible.
FS = 1.5


def plot_corner(tauV, xi, t, *, prior, t_range, truth, label, out_path,
                show_title=False, image=None, ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    # McDonald's truncated Blues (drop the near-white / near-black extremes).
    full = sns.color_palette("Blues", as_cmap=True)
    cmap = LinearSegmentedColormap.from_list(
        "trunc_blues", full(np.linspace(0.0, 0.85, 256)))

    # Order [tau_V, t, xi] so the bottom-left corner panel is (tau_V on x,
    # xi on y) -- the key physical pair, matching the 2-parameter KDE.
    names = [r"$\tau_V$", r"$t_D$", r"$\xi$"]
    data = [np.asarray(tauV), np.asarray(t), np.asarray(xi)]
    ranges = [tuple(prior[0]), tuple(t_range), tuple(prior[1])]
    truths = ([truth[0], truth[2], truth[1]] if truth is not None
              else [None, None, None])
    refs = ([ref[0], ref[2], ref[1]] if ref is not None
            else [None, None, None])
    n = 3
    fig, axes = plt.subplots(n, n, figsize=(10, 9))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            ax.tick_params(labelsize=10 * FS)
            if i == j:
                # 1-D marginal posterior, PEAK-NORMALISED to a shared [0, 1]
                # scale so the three diagonals are comparable; the y-axis is
                # labelled "density" on the top-left panel so it is explicit.
                d = np.asarray(data[i])
                lo, hi = ranges[i]
                if len(d) >= 5 and np.std(d) > 0:
                    kde = gaussian_kde(d)
                    xs = np.linspace(lo, hi, 256)
                    ys = kde(xs)
                    ys = ys / ys.max() if ys.max() > 0 else ys
                    ax.fill_between(xs, ys, color="#3b6fb6", alpha=0.55, lw=0)
                    ax.plot(xs, ys, color="#3b6fb6", lw=1.2)
                if truths[i] is not None:
                    ax.axvline(truths[i], color="red", lw=2)
                if refs[i] is not None:
                    ax.axvline(refs[i], color=GOLD, lw=2, zorder=11)
                ax.set_xlim(lo, hi)
                ax.set_ylim(0.0, 1.08)
                ax.set_ylabel("density", size=11 * FS)
                ax.set_yticks([0.0, 1.0])
                ax.set_yticklabels(["0", "1"], fontsize=9 * FS)
            else:
                # 2-D: filled seaborn KDE + truth cross.
                try:
                    sns.kdeplot(x=data[j], y=data[i], ax=ax, cmap=cmap,
                                fill=True, thresh=0.02, levels=10)
                except Exception as e:  # pragma: no cover
                    logger.warning("kdeplot failed (%s); scatter", e)
                    ax.scatter(data[j], data[i], s=4, alpha=0.3,
                               color="steelblue")
                if truths[j] is not None and truths[i] is not None:
                    ax.scatter([truths[j]], [truths[i]], **TRUTH_KW)
                if refs[j] is not None and refs[i] is not None:
                    ax.plot([refs[j]], [refs[i]], marker="*", color=GOLD,
                            ms=20, mec="k", mew=1.2, zorder=12, clip_on=False)
                ax.set_xlim(*ranges[j])
                ax.set_ylim(*ranges[i])
            # Tick / label hygiene: labels only on the outer edges.
            if i == n - 1:
                ax.set_xlabel(names[j], size=13 * FS)
            else:
                ax.set_xticks([])
                ax.set_xlabel("")
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], size=13 * FS)
            elif i != j:
                ax.set_yticks([])
                ax.set_ylabel("")

    if show_title:
        ttxt = (f"   truth = ($\\tau_V$={truths[0]:.0f}, $\\xi$={truths[2]:.3f}, "
                f"$t_D$={truths[1]:.0f})" if truth is not None else "")
        fig.suptitle(f"{label}: 3-parameter ABC rejection posterior  "
                     f"($n_{{\\rm accept}}={len(data[0])}$){ttxt}", fontsize=12 * FS)
    fig.tight_layout()
    if image is not None:
        from matplotlib.colors import ListedColormap
        lcm = ListedColormap(["white", "#fb9a99", "#3690c0"])  # bg / cell / lumen
        # Place the image only in the top-right cell so it never overlaps the
        # lower-triangle posterior panels (the t-marginal sits at [1,1]).
        p02 = axes[0, n - 1].get_position()
        for a in (axes[0, 1], axes[0, 2], axes[1, 2]):
            a.remove()
        iax = fig.add_axes([p02.x0, p02.y0, p02.width, p02.height])
        iax.imshow(image, cmap=lcm, vmin=0, vmax=2, interpolation="nearest", aspect="equal")
        iax.set_xticks([]); iax.set_yticks([])
        iax.set_title("organoid", fontsize=11 * FS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches=("tight" if image is not None else None))
    plt.close(fig)
    logger.info("saved 3-param corner to %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="e.g. fakexp_idx0_t2260")
    p.add_argument("--dist-dir", default=str(DIST_DIR))
    p.add_argument("--out-dir", default=str(POST_DIR))
    p.add_argument("--accept-quantile", type=float, default=CONFIG.accept_quantile)
    p.add_argument("--distance", default=None)
    p.add_argument("--t-max", type=float, default=None,
                   help="t-axis max (default: 1.05 * max(accepted t, truth t)).")
    p.add_argument("--title", action="store_true",
                   help="Draw the figure title (default off; caption carries it).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    dist_path = find_distances_file(Path(args.dist_dir), args.label,
                                    distance=args.distance)
    cols = load_distances(dist_path)
    for c in ("tauV", "xi", "t", "distance"):
        if c not in cols:
            sys.exit(f"distances file missing column '{c}'")
    tauV = cols["tauV"].astype(float)
    xi = cols["xi"].astype(float)
    t = cols["t"].astype(float)
    dist = cols["distance"].astype(float)
    m = np.isfinite(dist)
    tauV, xi, t, dist = tauV[m], xi[m], t[m], dist[m]

    eps = float(np.quantile(dist, args.accept_quantile))
    acc = dist <= eps
    logger.info("%s: %d/%d accepted (eps=%.4g)", args.label, int(acc.sum()),
                len(dist), eps)
    atau, axi, at = tauV[acc], xi[acc], t[acc]

    truth: Optional[tuple] = None
    mlab = re.match(r"fakexp_idx(\d+)_t(\d+)", args.label)
    if mlab:
        txy = FAKEXP_TRUTH.get(int(mlab.group(1)))
        if txy is not None:
            truth = (float(txy[0]), float(txy[1]), float(mlab.group(2)))

    t_top = max(float(at.max()) if len(at) else 0.0, truth[2] if truth else 0.0)
    t_max = args.t_max if args.t_max is not None else max(1.0, t_top * 1.05)
    plot_corner(atau, axi, at, prior=prior_box(CONFIG), t_range=(0.0, t_max),
                truth=truth, label=args.label, show_title=args.title,
                out_path=Path(args.out_dir) / f"{args.label}__corner3p.png")


if __name__ == "__main__":
    main()
