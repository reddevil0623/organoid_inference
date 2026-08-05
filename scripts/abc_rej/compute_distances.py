"""Compute observation-to-cache distances for ABC rejection.

Featurises one observation, samples candidates from the cached simulation
frames, computes the SampEuler-Wasserstein distance to each, and writes
``<label>__distances.csv`` with columns
``theta_idx, frame_idx, tau_V, xi, t, distance``.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import (
    CACHE_DIR,
    CONFIG,
    Config,
    DIST_DIR,
    OBS_DIR,
    SIM_MANIFEST,
    DEFAULT_BOUNDS_FILE,
    load_sampeuler_bounds,
)
from .observations import load_fakexp_observation
from .sim_worker import (
    compute_ect_curves,
    featurise_frame,
    vectorise_sampeuler,
)


logger = logging.getLogger("abc_rej.distances")


# Observation feature computation / cache

def _bounds_fingerprint(bounds: dict, canvas_side: int) -> dict:
    """Pack the bounds + canvas spec into a small dict of scalars for cache invariance checks."""
    return {
        "x_min": float(bounds["x_min"]),
        "x_max": float(bounds["x_max"]),
        "y_min": float(bounds["y_min"]),
        "y_max": float(bounds["y_max"]),
        "xpoints": int(bounds["xpoints"]),
        "n_chi": int(bounds["n_chi"]),
        "n_dirs": int(bounds["n_dirs"]),
        "seed": int(bounds["seed"]),
        "canvas_side": int(canvas_side),
    }


def _fingerprints_equal(a: dict, b: dict, tol: float = 1e-9) -> bool:
    """Compare two bounds fingerprints; floats with tolerance, ints exact."""
    if set(a) != set(b):
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float):
            if abs(av - float(bv)) > tol:
                return False
        else:
            if int(av) != int(bv):
                return False
    return True


def compute_observation_features(
    obs,
    *,
    bounds: dict,
    canvas_side: int,
    cache_path: Path,
    return_ect: bool = False,
) -> "tuple[np.ndarray, Optional[np.ndarray]]":
    """Compute SampEuler (and optional ECT) features for an observation, caching to NPZ.
    """
    fp_now = _bounds_fingerprint(bounds, canvas_side)
    if cache_path.exists():
        try:
            blob = np.load(cache_path, allow_pickle=False)
            X = blob["features"]
            cached_fp = {k: blob[k].item() if hasattr(blob[k], "item") else blob[k]
                         for k in fp_now if k in blob.files}
            if (X.ndim == 2 and X.dtype == np.float32
                    and len(cached_fp) == len(fp_now)
                    and _fingerprints_equal(cached_fp, fp_now)
                    and int(blob["n_obs_frames"]) == len(obs.frames)):
                if return_ect and "ect_curves" not in blob.files:
                    logger.warning("[%s] cache lacks ect_curves; recomputing",
                                   obs.label)
                else:
                    logger.info("[%s] reusing cached features %s",
                                obs.label, cache_path)
                    ect_arr = (blob["ect_curves"]
                               if return_ect and "ect_curves" in blob.files
                               else None)
                    return X, ect_arr
            else:
                logger.warning("[%s] cache invariant mismatch; recomputing",
                               obs.label)
        except Exception as e:
            logger.warning("[%s] failed to read cache (%s); recomputing", obs.label, e)

    _ensure_scripts_on_path()
    from core import ECTComputer  # noqa: E402

    ect = ECTComputer(
        n_dirs=int(bounds["n_dirs"]),
        xpoints=int(bounds["xpoints"]),
        x_min=float(bounds["x_min"]),
        x_max=float(bounds["x_max"]),
    )
    # Synthetic observations skip the bbox/resize/pad step, like the sim
    # cache. Theta angles are drawn fresh per call, matching the cached
    # simulator-side ECT computation. Experimental images are size-normalised
    # instead, by exp_distances.py.
    feats: List[np.ndarray] = []
    ect_list: List[np.ndarray] = []
    for frame in obs.frames:
        curves = compute_ect_curves(frame, ect=ect, bounds=bounds)
        feats.append(vectorise_sampeuler(curves, ect=ect, bounds=bounds))
        ect_list.append(curves)
    X = np.stack(feats).astype(np.float32)
    ect_arr = np.stack(ect_list).astype(np.float32) if ect_list else None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        features=X,
        label=np.array(obs.label),
        truth=(np.array(obs.truth, dtype=np.float64) if obs.truth is not None
               else np.array([np.nan, np.nan])),
        n_obs_frames=np.int64(len(feats)),
        delta_x=np.float64(ect.delta_x),
        **{k: np.array(v) for k, v in fp_now.items()},
    )
    if ect_arr is not None:
        save_kwargs["ect_curves"] = ect_arr
    np.savez_compressed(cache_path, **save_kwargs)
    logger.info("[%s] cached %d observation features to %s",
                obs.label, len(feats), cache_path)
    return X, (ect_arr if return_ect else None)


# Distance loop

def _hungarian_l1_pair(curves_a: np.ndarray,
                       curves_b: np.ndarray,
                       delta_x: float) -> float:
    """Hungarian-matched L1 distance between two single-frame ECT curve sets."""
    from scipy.optimize import linear_sum_assignment

    C = np.sum(np.abs(curves_a[:, None, :] - curves_b[None, :, :]),
               axis=2) * delta_x
    r, c = linear_sum_assignment(C)
    return float(np.mean(C[r, c]))


def _build_pool_index(paths: "list[Path]") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """First-pass cache scan: build a flat index of every cached frame."""
    path_idx_list, frame_idx_list = [], []
    theta_idx_list, tauV_list, xi_list, t_list = [], [], [], []
    for pi, p in enumerate(paths):
        try:
            blob = np.load(p, allow_pickle=False)
            sim_t = np.asarray(blob["t"], dtype=np.int64)
            theta_idx = int(blob["theta_idx"])
            tauV = float(blob["tauV"])
            xi = float(blob["xi"])
        except Exception as e:
            logger.warning("pool-scan skipping %s: %s", p.name, e)
            continue
        n_frames = int(len(sim_t))
        if n_frames == 0:
            continue
        path_idx_list.append(np.full(n_frames, pi, dtype=np.int64))
        frame_idx_list.append(np.arange(n_frames, dtype=np.int64))
        theta_idx_list.append(np.full(n_frames, theta_idx, dtype=np.int64))
        tauV_list.append(np.full(n_frames, tauV, dtype=np.float64))
        xi_list.append(np.full(n_frames, xi, dtype=np.float64))
        t_list.append(sim_t)
    if not path_idx_list:
        raise RuntimeError("Pool index is empty — no usable cache NPZs.")
    return (
        np.concatenate(path_idx_list),
        np.concatenate(frame_idx_list),
        np.concatenate(theta_idx_list),
        np.concatenate(tauV_list),
        np.concatenate(xi_list),
        np.concatenate(t_list),
    )


def compute_distances_for_observation(
    obs_features: np.ndarray,
    cache_dir: Path,
    *,
    distance: str = "l2",
    aggregation: str = "single",
    bounds: "Optional[dict]" = None,
    obs_ect: "Optional[np.ndarray]" = None,
    n_pool_samples: "Optional[int]" = None,
    pool_seed: int = 42,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Pool-sample the cache and compute observation-vs-sim distances."""
    if obs_features.ndim != 2:
        raise ValueError(f"obs_features must be 2D, got {obs_features.shape}")
    n_obs = obs_features.shape[0]
    if aggregation == "single" and n_obs != 1:
        raise ValueError(
            f"aggregation='single' requires n_obs == 1, got {n_obs}; "
            "use --aggregation min / mean for multi-frame observations."
        )
    if aggregation == "movie-min":
        raise NotImplementedError(
            "movie-min aggregation collapses to per-(tau_V, xi) particles; "
            "this is incompatible with the per-frame parquet layout."
        )
    if distance == "hungarian":
        if obs_ect is None:
            raise ValueError(
                "distance='hungarian' requires obs_ect (the (n_obs, n_dirs, "
                "xpoints) raw ECT curves of the observation); pass return_ect=True "
                "to compute_observation_features to get them."
            )
        if obs_ect.ndim != 3 or obs_ect.shape[0] != n_obs:
            raise ValueError(
                f"obs_ect shape {obs_ect.shape} inconsistent with n_obs={n_obs}"
            )

    paths = sorted(cache_dir.glob("abcrej_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cache NPZs in {cache_dir}")

    # Optional invariant check vs the obs's bounds.
    expected_n_chi = int(bounds["n_chi"]) if bounds else None
    expected_xpoints = int(bounds["xpoints"]) if bounds else None
    expected_D = (None if bounds is None
                  else expected_n_chi * expected_xpoints)
    if expected_D is not None and obs_features.shape[1] != expected_D:
        raise ValueError(
            f"obs_features dim {obs_features.shape[1]} != "
            f"n_chi * xpoints = {expected_n_chi} * {expected_xpoints} = {expected_D}"
        )

    # Pre-compute observation norm² for fast L2.
    obs_norm2 = (obs_features ** 2).sum(axis=1)  # (n_obs,)

    # ── Pass 1: build the global pool of cached frames ──
    t_scan = time.time()
    pool_path_idx, pool_frame_idx, pool_theta_idx, pool_tauV, pool_xi, pool_t = \
        _build_pool_index(paths)
    pool_size = int(len(pool_path_idx))
    logger.info(
        "pool scan: %d frames across %d cache NPZs in %.1fs",
        pool_size, len(paths), time.time() - t_scan,
    )

    # ── Sample pool indices uniformly without replacement ──
    # Note: with the 3-stage workflow, the cache is *already* the
    # Stage-A2 pool sample, so any further sub-sampling here is a
    # sample-of-a-sample with mismatched semantics. If a user passes a
    # smaller --n-pool-samples they should know what they're getting.
    if n_pool_samples is None or n_pool_samples <= 0 or n_pool_samples >= pool_size:
        sel = np.arange(pool_size, dtype=np.int64)
        logger.info("pool sampling disabled; computing distances against all "
                    "%d cached frames", pool_size)
    else:
        rng = np.random.default_rng(int(pool_seed))
        sel = rng.choice(pool_size, size=int(n_pool_samples), replace=False)
        sel.sort()  # contiguous-by-path access is cheaper
        logger.warning(
            "WARNING: drew %d / %d *cached* frames (seed=%d). The cache "
            "is already a uniform pool sample (Stage A2 output), so this "
            "is a sample-of-a-sample. To score every cached frame, "
            "pass --n-pool-samples 0 (or omit; the default 20000 is "
            "usually >= cache size and short-circuits to all).",
            len(sel), pool_size, int(pool_seed),
        )

    sel_path_idx = pool_path_idx[sel]
    sel_frame_idx = pool_frame_idx[sel]
    sel_theta_idx = pool_theta_idx[sel]
    sel_tauV = pool_tauV[sel]
    sel_xi = pool_xi[sel]
    sel_t = pool_t[sel]

    # ── Pass 2: group sampled rows by cache path, compute distances ──
    # Distances are written into `dist_out` at positions matching `sel`'s
    # original order so the output flat arrays line up.
    dist_out = np.full(len(sel), np.nan, dtype=np.float64)
    n_skipped_shape = 0

    # Map path_idx -> rows-into-sel that reference it
    unique_paths, inverse = np.unique(sel_path_idx, return_inverse=True)
    t0 = time.time()
    for k, pi in enumerate(unique_paths):
        p = paths[int(pi)]
        try:
            blob = np.load(p, allow_pickle=False)
        except Exception as e:
            logger.warning("skipping %s: %s", p.name, e)
            continue
        sim_X = blob["sampeuler"]                 # (n_sim_frames, D)
        sim_n_chi = int(blob["n_chi"]) if "n_chi" in blob.files else None
        sim_xpoints = int(blob["xpoints"]) if "xpoints" in blob.files else None

        if sim_X.ndim != 2 or sim_X.shape[1] != obs_features.shape[1]:
            logger.warning("skipping %s: D mismatch (sim %s vs obs %s)",
                           p.name, sim_X.shape, obs_features.shape)
            n_skipped_shape += 1
            continue
        # Stricter: also check (n_chi, xpoints) scalar match if both sides
        # know their own values, because two different (n_chi, xpoints)
        # pairs can multiply to the same flat dimension D.
        if (expected_n_chi is not None and sim_n_chi is not None
                and (sim_n_chi != expected_n_chi
                     or sim_xpoints != expected_xpoints)):
            logger.warning(
                "skipping %s: (n_chi, xpoints) mismatch -- sim "
                "(%s, %s) vs obs (%s, %s)",
                p.name, sim_n_chi, sim_xpoints,
                expected_n_chi, expected_xpoints,
            )
            n_skipped_shape += 1
            continue

        rows_for_this_path = np.where(inverse == k)[0]
        local_frame_idx = sel_frame_idx[rows_for_this_path]
        # Slice out only the frames we'll actually score.
        sim_X_sub = sim_X[local_frame_idx]                  # (n_pick, D)

        if distance in ("l2", "l2sq"):
            # Batched L2² via the dot-product identity:
            # ||a - b||² = ||a||² + ||b||² - 2 a·b
            sim_norm2 = (sim_X_sub ** 2).sum(axis=1)
            cross = obs_features @ sim_X_sub.T              # (n_obs, n_pick)
            d2 = obs_norm2[:, None] + sim_norm2[None, :] - 2.0 * cross
            np.maximum(d2, 0.0, out=d2)
            d_full = np.sqrt(d2) if distance == "l2" else d2
        elif distance == "hungarian":
            # Pair-wise Hungarian (linear sum assignment) on the raw ECT
            # curves. This is exactly `core.WassersteinDistance.hungarian_l1`
            # applied to single-frame inputs. Costly: O(n_dirs³) per pair,
            # O(n_obs * n_pick) pairs per cache NPZ.
            if "ect_curves" not in blob.files:
                logger.warning(
                    "skipping %s: distance='hungarian' but cache has no "
                    "ect_curves (regenerate the cache to enable Hungarian).",
                    p.name,
                )
                continue
            sim_ect = blob["ect_curves"]               # (n_sim_frames, n_dirs, xpoints)
            if sim_ect.shape[1:] != obs_ect.shape[1:]:
                logger.warning(
                    "skipping %s: ECT shape mismatch (sim %s vs obs %s)",
                    p.name, sim_ect.shape, obs_ect.shape,
                )
                continue
            sim_ect_sub = sim_ect[local_frame_idx]      # (n_pick, n_dirs, xpoints)
            delta_x = (float(blob["delta_x"]) if "delta_x" in blob.files
                       else (float(bounds["x_max"]) - float(bounds["x_min"]))
                            / max(1, int(bounds["xpoints"]) - 1))
            n_pick = sim_ect_sub.shape[0]
            d_full = np.empty((n_obs, n_pick), dtype=np.float64)
            for ii in range(n_obs):
                for jj in range(n_pick):
                    d_full[ii, jj] = _hungarian_l1_pair(
                        obs_ect[ii], sim_ect_sub[jj], delta_x,
                    )
        else:
            raise ValueError(f"Unsupported distance: {distance}")

        # Aggregation across observation frames.
        if aggregation in ("single", "min"):
            d_per = d_full.min(axis=0)
        elif aggregation == "mean":
            d_per = d_full.mean(axis=0)
        else:  # pragma: no cover -- already validated above
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        dist_out[rows_for_this_path] = d_per.astype(np.float64)

        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info("scored %d / %d cache NPZs in %.1fs",
                        k + 1, len(unique_paths), elapsed)

    if n_skipped_shape:
        logger.warning("skipped %d cache NPZs due to shape/bounds mismatch",
                       n_skipped_shape)

    # Drop rows whose path was skipped (NaN distance).
    keep = np.isfinite(dist_out)
    if not keep.all():
        n_drop = int((~keep).sum())
        logger.warning("dropping %d rows whose cache NPZ was unusable",
                       n_drop)

    return (sel_theta_idx[keep],
            sel_frame_idx[keep],
            sel_tauV[keep],
            sel_xi[keep],
            sel_t[keep],
            dist_out[keep])


# Output writer

def save_distances(
    out_path: Path,
    theta_idx: np.ndarray,
    frame_idx: np.ndarray,
    tauV: np.ndarray,
    xi: np.ndarray,
    t: np.ndarray,
    distance: np.ndarray,
) -> None:
    """Write a flat per-row table; parquet preferred, CSV fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None

    data = dict(
        theta_idx=theta_idx,
        frame_idx=frame_idx,
        tauV=tauV,
        xi=xi,
        t=t,
        distance=distance,
    )
    if pd is not None:
        df = pd.DataFrame(data)
        try:
            df.to_parquet(out_path)
            logger.info("wrote parquet %s (%d rows)", out_path, len(df))
            return
        except Exception as e:
            logger.warning("parquet write failed (%s); falling back to CSV", e)
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            # Remove any stale parquet so plot_posterior doesn't pick it up.
            try:
                if out_path.exists():
                    out_path.unlink()
            except Exception:
                pass
            logger.info("wrote CSV %s (%d rows)", csv_path, len(df))
            return

    # Plain-stdlib fallback: CSV.
    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(data.keys()))
        n = len(theta_idx)
        for i in range(n):
            w.writerow([int(theta_idx[i]), int(frame_idx[i]),
                        float(tauV[i]), float(xi[i]), int(t[i]),
                        float(distance[i])])
    logger.info("wrote CSV %s (%d rows)", csv_path, len(theta_idx))


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fakexp-idx", type=int,
                     help="Compute distances for FAKEXP synthetic idx (0/1/2).")
    src.add_argument("--obs-index", type=int,
                     help="Alias for --fakexp-idx; mirrors fakexp_parallel idiom.")

    p.add_argument("--fakexp-search-roots", nargs="+", type=str,
                   default=[
                       str(CONFIG.project_root / "data" / "DATA_stride5" / "fakexp"),
                       "/scratch/abc_fakexp/DATA",
                   ],
                   help="Where to look for fakexp_<idx>_* dirs.")
    p.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    p.add_argument("--obs-cache-dir", type=str, default=str(OBS_DIR))
    p.add_argument("--out-dir", type=str, default=str(DIST_DIR))
    p.add_argument("--bounds", type=str, default=str(DEFAULT_BOUNDS_FILE))
    p.add_argument("--canvas-side", type=int, default=1024)
    p.add_argument("--n-frames", type=int, default=1,
                   help="Number of observation frames (default 1 = single "
                        "image at --obs-t, matching 3-D ABC rejection with "
                        "t as a parameter).")
    p.add_argument("--obs-t", type=int, default=None,
                   help="Simulator-time index for the single observation "
                        "frame (default: last available frame). Only used "
                        "when --n-frames=1.")
    p.add_argument("--u-thresh", type=float, default=CONFIG.u_thresh)
    p.add_argument("--s-thresh", type=float, default=CONFIG.s_thresh)
    p.add_argument("--distance", choices=("l2", "l2sq", "hungarian"),
                   default=CONFIG.distance,
                   help="'hungarian' (default): Wasserstein-1 "
                        "Hungarian-matched distance between raw ECT curves "
                        "(~minutes per observation across the 1000-sim "
                        "cache). 'l2': "
                        "Euclidean L2 between flattened SampEuler images "
                        "-- fast, batched. 'l2sq' is the squared variant.")
    p.add_argument("--aggregation", choices=("single", "min", "mean"),
                   default="single",
                   help="How to combine distances across observation frames "
                        "when --n-frames > 1. Default 'single' requires "
                        "--n-frames=1.")
    p.add_argument("--n-pool-samples", type=int, default=CONFIG.n_pool_samples,
                   help="Draw this many (theta_idx, frame_idx) candidates "
                        "uniformly without replacement from the global pool "
                        "of cached frames before computing distances. "
                        "Set to 0 to disable pool sampling and score every "
                        "cached frame.")
    p.add_argument("--pool-seed", type=int, default=CONFIG.pool_seed,
                   help="RNG seed for the pool draw. Same seed across "
                        "observations guarantees each observation is "
                        "scored against the same 20 000-frame subset.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    obs_idx = args.fakexp_idx if args.fakexp_idx is not None else args.obs_index
    if obs_idx is None:
        sys.exit("Need --fakexp-idx or --obs-index")

    bounds = load_sampeuler_bounds(Path(args.bounds))
    obs = load_fakexp_observation(
        obs_idx,
        [Path(r) for r in args.fakexp_search_roots],
        n_frames=args.n_frames,
        u_thresh=args.u_thresh,
        s_thresh=args.s_thresh,
        obs_t=args.obs_t,
    )
    obs_cache_path = Path(args.obs_cache_dir) / f"{obs.label}__features.npz"
    obs_features, obs_ect = compute_observation_features(
        obs, bounds=bounds, canvas_side=args.canvas_side,
        cache_path=obs_cache_path,
        return_ect=(args.distance == "hungarian"),
    )

    arrays = compute_distances_for_observation(
        obs_features, Path(args.cache_dir),
        distance=args.distance,
        aggregation=args.aggregation,
        bounds=bounds,
        obs_ect=obs_ect,
        n_pool_samples=(None if args.n_pool_samples <= 0
                        else int(args.n_pool_samples)),
        pool_seed=int(args.pool_seed),
    )

    suffix = (f"__distances_{args.distance}.parquet"
              if args.distance != CONFIG.distance
              else "__distances.parquet")
    out_path = Path(args.out_dir) / f"{obs.label}{suffix}"
    save_distances(out_path, *arrays)
    logger.info("done")


if __name__ == "__main__":
    main()
