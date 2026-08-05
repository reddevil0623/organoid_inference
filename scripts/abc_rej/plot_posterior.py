"""Threshold a distance file and plot the 2-D (tau_V, xi) posterior.

Applies the ABC rejection acceptance quantile, KDEs the accepted values, and
saves the contour and marginal plots.
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

from .config import (
    CONFIG,
    DIST_DIR,
    OBS_DIR,
    POST_DIR,
    prior_box,
)
from .observations import FAKEXP_TRUTH


logger = logging.getLogger("abc_rej.plot")


# Data loading

def load_distances(path: Path) -> dict:
    """Load a distance table, accepting either parquet or CSV."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet {path}: {e}") from e
        return {col: df[col].to_numpy() for col in df.columns}

    if suffix == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(path)
            return {col: df[col].to_numpy() for col in df.columns}
        except ImportError:
            pass
        # Stdlib CSV fallback
        import csv as _csv
        rows = list(_csv.DictReader(open(path)))
        out: dict = {k: [] for k in rows[0].keys()} if rows else {}
        for r in rows:
            for k, v in r.items():
                out[k].append(v)
        for k in out:
            try:
                out[k] = np.array([float(x) for x in out[k]])
            except ValueError:
                out[k] = np.array(out[k])
        return out

    raise ValueError(f"Unrecognised distances file format: {path}")


def find_distances_file(out_dir: Path, label: str,
                        distance: "Optional[str]" = None) -> Path:
    """Look for the parquet first, then CSV."""
    candidates: "list[Path]" = []
    if distance:
        candidates += [
            out_dir / f"{label}__distances_{distance}.parquet",
            out_dir / f"{label}__distances_{distance}.csv",
        ]
    candidates += [
        out_dir / f"{label}__distances.parquet",
        out_dir / f"{label}__distances.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No distances file for label '{label}' (distance='{distance}') "
        f"under {out_dir}; tried: {[c.name for c in candidates]}"
    )


# Plotting

def plot_2d_kde(
    accepted_tauV: np.ndarray,
    accepted_xi: np.ndarray,
    *,
    prior: "list[tuple[float, float]]",
    truth: Optional["tuple[float, float]"] = None,
    label: str = "",
    eps: float = float("nan"),
    n_total: int = 0,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # McDonald-style filled seaborn KDE (truncated Blues, no near-white/black).
    full = sns.color_palette("Blues", as_cmap=True)
    cmap = LinearSegmentedColormap.from_list(
        "trunc_blues", full(np.linspace(0.0, 0.85, 256)))

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    pbox = prior

    if len(accepted_tauV) >= 5:
        try:
            sns.kdeplot(x=accepted_tauV, y=accepted_xi, ax=ax, cmap=cmap,
                        fill=True, thresh=0.02, levels=10)
        except Exception as e:
            logger.warning("kdeplot failed (%s); falling back to scatter", e)
            ax.scatter(accepted_tauV, accepted_xi, s=4, alpha=0.3,
                       color="steelblue")
    else:
        ax.scatter(accepted_tauV, accepted_xi, s=8, alpha=0.5,
                   color="steelblue")

    if truth is not None and not (np.isnan(truth[0]) or np.isnan(truth[1])):
        ax.scatter([truth[0]], [truth[1]], marker="x", color="red", s=220,
                   linewidths=3, zorder=10, clip_on=False,
                   label=f"truth = ({truth[0]:.2f}, {truth[1]:.3f})")
        ax.legend(loc="best", fontsize=9)

    ax.set_xlim(*pbox[0])
    ax.set_ylim(*pbox[1])
    ax.set_xlabel(r"$\tau_V$")
    ax.set_ylabel(r"$\xi$")
    title = (f"{label}: ABC rejection posterior on $(\\tau_V, \\xi)$  "
             f"($n_{{\\rm accept}}={len(accepted_tauV)}$ / "
             f"$n_{{\\rm total}}={n_total}$, $\\epsilon={eps:.4g}$)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("saved 2D KDE to %s", out_path)


def plot_marginals(
    accepted_tauV: np.ndarray,
    accepted_xi: np.ndarray,
    *,
    prior: "list[tuple[float, float]]",
    truth: Optional["tuple[float, float]"] = None,
    label: str = "",
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = (r"$\tau_V$", r"$\xi$")
    data = (accepted_tauV, accepted_xi)
    for d, (ax, name, vals, (lo, hi)) in enumerate(zip(axes, names, data, prior)):
        sns.kdeplot(x=vals, ax=ax, color="#3b6fb6", fill=True, alpha=0.55,
                    cut=0, linewidth=1.2)
        if truth is not None and not np.isnan(truth[d]):
            ax.axvline(truth[d], color="red", lw=2, label=f"truth = {truth[d]:.4g}")
            ax.legend()
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
    fig.suptitle(f"{label}: ABC rejection marginals")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("saved marginals to %s", out_path)


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--label", type=str,
                     help="Observation label (e.g. fakexp_idx0).")
    src.add_argument("--distances", type=str,
                     help="Direct path to a __distances.{parquet,csv} file.")

    p.add_argument("--out-dir", type=str, default=str(POST_DIR))
    p.add_argument("--accept-quantile", type=float, default=CONFIG.accept_quantile,
                   help="Acceptance fraction; epsilon = quantile of distances.")
    p.add_argument("--obs-cache-dir", type=str, default=str(OBS_DIR),
                   help="Where the observation features NPZ lives "
                        "(used to recover the ground truth).")
    p.add_argument("--dist-dir", type=str, default=str(DIST_DIR))
    p.add_argument("--distance", default=None,
                   help="If set (e.g. 'hungarian'), look for a "
                        "metric-suffixed distances file first.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    cfg = CONFIG

    if args.distances:
        dist_path = Path(args.distances)
        # Strip any metric suffix or "__distances" from the stem to recover the label.
        stem = dist_path.stem
        for suffix_token in ("__distances_l2", "__distances_l2sq",
                             "__distances_hungarian", "__distances"):
            if stem.endswith(suffix_token):
                stem = stem[: -len(suffix_token)]
                break
        label = stem
    else:
        label = args.label
        dist_path = find_distances_file(Path(args.dist_dir), label,
                                        distance=args.distance)
    logger.info("loading distances %s", dist_path)
    cols = load_distances(dist_path)

    needed = {"tauV", "xi", "distance"}
    if not needed.issubset(cols.keys()):
        sys.exit(f"distance file missing required columns: {needed - cols.keys()}")

    tauV = cols["tauV"].astype(float)
    xi = cols["xi"].astype(float)
    distance = cols["distance"].astype(float)

    # Drop non-finite distances (failed sims, log issues).
    mask = np.isfinite(distance)
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.warning("dropped %d non-finite distances (of %d total)",
                       n_dropped, len(distance))
    tauV, xi, distance = tauV[mask], xi[mask], distance[mask]
    n_total = len(distance)
    if n_total == 0:
        sys.exit("No finite distances to threshold.")

    # Threshold.
    eps = float(np.quantile(distance, args.accept_quantile))
    accept_mask = distance <= eps
    accepted_tauV = tauV[accept_mask]
    accepted_xi = xi[accept_mask]
    logger.info(
        "%s: %d/%d accepted at q=%.3f (eps=%.4g)",
        label, accept_mask.sum(), n_total, args.accept_quantile, eps,
    )
    if accept_mask.sum() == 0:
        sys.exit("No particles accepted at the requested quantile.")

    # Truth lookup. Prefer FAKEXP table; fall back to observation NPZ if present.
    truth = None
    m = re.match(r"fakexp_idx(\d+)", label)
    if m:
        truth = FAKEXP_TRUTH.get(int(m.group(1)))
    if truth is None:
        obs_npz = Path(args.obs_cache_dir) / f"{label}__features.npz"
        if obs_npz.exists():
            try:
                blob = np.load(obs_npz, allow_pickle=False)
                t = blob.get("truth")
                if t is not None:
                    t = np.asarray(t).astype(float)
                    if t.shape == (2,) and not np.isnan(t).any():
                        truth = (float(t[0]), float(t[1]))
            except Exception:
                pass

    out_dir = Path(args.out_dir)
    suffix = f"_{args.distance}" if args.distance else ""
    plot_2d_kde(
        accepted_tauV, accepted_xi,
        prior=prior_box(cfg),
        truth=truth,
        label=label,
        eps=eps,
        n_total=n_total,
        out_path=out_dir / f"{label}__kde{suffix}.png",
    )
    plot_marginals(
        accepted_tauV, accepted_xi,
        prior=prior_box(cfg),
        truth=truth,
        label=label,
        out_path=out_dir / f"{label}__marginals{suffix}.png",
    )


if __name__ == "__main__":
    main()
