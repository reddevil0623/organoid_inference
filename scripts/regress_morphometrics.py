#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict classical morphological scalars from the SampEuler descriptor.

Targets are computed from each frame's {0,1,2} label image: lumen count, lumen
area ratio, and organoid radius. Fits a StandardScaler to estimator pipeline
with k-fold cross-validation grouped by (tau_V, xi), so no parameter cell leaks
across folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import paths as _p
_p.ensure_eucalc()

from sampeuler import SampEulerVectorization
from core import (
    ECTComputer,
    iter_run_frames, lumen_count, lumen_area_ratio, organoid_radius,
)


_PARAM_RE = re.compile(r"tauV(?P<tauV>\d+)_xi(?P<xi>\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bounds", type=Path, required=True,
                   help="ECT bounds npz from compute_sampeuler_bounds.py")
    p.add_argument("--sweep-root", type=Path, required=True)
    p.add_argument("--sweep-pattern", default="dt002_tauV*_xi*",
                   help="Glob for sweep run subdir names "
                        "(default matches DATA_stride5/sweep/).")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.01, 0.1, 1.0, 10.0, 100.0],
                   help="Ridge regularisation strengths for RidgeCV inner search.")
    p.add_argument("--models", nargs="+", default=["linear", "ridge"],
                   choices=["linear", "ridge"],
                   help="Models to evaluate. Linear is reported first as the "
                        "baseline; ridge follows if regularisation helps.")
    p.add_argument("--cache", type=Path, default=None,
                   help="npz cache for features+targets; default <out>/dataset.npz")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def parse_label_params(name: str) -> Optional[Tuple[int, int]]:
    m = _PARAM_RE.search(name)
    if not m:
        return None
    return int(m.group("tauV")), int(m.group("xi"))


def discover_sweep_runs(root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def featurise(img: np.ndarray, ect: ECTComputer, thetas: np.ndarray,
              y_min: float, y_max: float, n_chi: int) -> np.ndarray:
    """SampEuler image (shape (n_chi, xpoints)) flattened to 1D."""
    if img.sum() == 0:
        return np.zeros(n_chi * ect.xpoints, dtype=np.float32)
    curves = ect.compute(img, thetas=thetas)
    image = SampEulerVectorization(
        precomputed=curves,
        xinterval=(ect.x_min, ect.x_max), xpoints=ect.xpoints,
        yinterval=(y_min, y_max),         ypoints=n_chi,
    ).image
    return image.astype(np.float32).flatten()


def build_dataset(runs: List[Path], ect: ECTComputer, thetas: np.ndarray,
                  cache_path: Path, bounds_meta: Dict) -> Dict:
    y_min = bounds_meta["y_min"]
    y_max = bounds_meta["y_max"]
    n_chi = bounds_meta["n_chi"]
    if cache_path.exists():
        d = np.load(str(cache_path), allow_pickle=False)
        ok = (int(d["xpoints"]) == ect.xpoints and
              int(d["n_dirs"]) == len(thetas) and
              float(d["x_min"]) == bounds_meta["x_min"] and
              float(d["x_max"]) == bounds_meta["x_max"] and
              float(d["y_min"]) == y_min and
              float(d["y_max"]) == y_max and
              int(d["n_chi"]) == n_chi and
              int(d["seed"]) == bounds_meta["seed"])
        if ok:
            print(f"[cache] reusing dataset from {cache_path}")
            return {k: d[k] for k in ("X", "y_count", "y_ratio",
                                      "y_radius", "tauV", "xi_pct", "t")}
        print(f"[cache] {cache_path} bounds mismatch — recomputing")

    feats: List[np.ndarray] = []
    y_count: List[int] = []
    y_ratio: List[float] = []
    y_radius: List[float] = []
    tauVs: List[int] = []
    xis: List[int] = []
    ts: List[int] = []
    t0 = time.time(); n = 0
    for r in runs:
        params = parse_label_params(r.name)
        if params is None:
            continue
        tauV, xi_pct = params
        for t, lbl in iter_run_frames(r):
            feats.append(featurise(lbl, ect, thetas, y_min, y_max, n_chi))
            y_count.append(lumen_count(lbl))
            y_ratio.append(lumen_area_ratio(lbl))
            y_radius.append(organoid_radius(lbl))
            tauVs.append(tauV); xis.append(xi_pct); ts.append(t)
            n += 1
            if n % 100 == 0:
                dt = time.time() - t0
                print(f"  built {n} samples in {dt:.1f}s "
                      f"({n / max(dt, 1e-6):.1f} f/s)")
    if not feats:
        sys.exit("No samples built.")
    out = {
        "X":       np.stack(feats).astype(np.float32),
        "y_count": np.asarray(y_count, dtype=np.float32),
        "y_ratio": np.asarray(y_ratio, dtype=np.float32),
        "y_radius": np.asarray(y_radius, dtype=np.float32),
        "tauV":   np.asarray(tauVs, dtype=np.int32),
        "xi_pct": np.asarray(xis, dtype=np.int32),
        "t":      np.asarray(ts, dtype=np.int32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path),
             x_min=np.float64(bounds_meta["x_min"]),
             x_max=np.float64(bounds_meta["x_max"]),
             y_min=np.float64(y_min),
             y_max=np.float64(y_max),
             n_chi=np.int64(n_chi),
             xpoints=np.int64(ect.xpoints),
             n_dirs=np.int64(len(thetas)),
             seed=np.int64(bounds_meta["seed"]),
             **out)
    print(f"[cache] saved {n} samples to {cache_path}")
    return out


def make_model_factory(name: str, alphas: List[float]) -> Callable[[], Pipeline]:
    """Return a zero-arg callable that produces a fresh sklearn Pipeline."""
    if name == "linear":
        return lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("est",    LinearRegression()),
        ])
    if name == "ridge":
        return lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("est",    RidgeCV(alphas=alphas)),
        ])
    raise ValueError(f"Unknown model: {name}")


def _fold_alpha(model: Pipeline) -> Optional[float]:
    est = model.named_steps.get("est")
    return float(est.alpha_) if hasattr(est, "alpha_") else None


def evaluate_target(X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                    k: int, factory: Callable[[], Pipeline]) -> Dict:
    gkf = GroupKFold(n_splits=k)
    fold_metrics: List[Dict] = []
    preds = np.zeros_like(y, dtype=np.float64)
    for f, (idx_tr, idx_te) in enumerate(gkf.split(X, y, groups=group_ids)):
        model = factory()
        model.fit(X[idx_tr], y[idx_tr])
        yhat = model.predict(X[idx_te])
        preds[idx_te] = yhat
        fold_metrics.append({
            "fold":     f,
            "alpha":    _fold_alpha(model),
            "n_train":  int(len(idx_tr)),
            "n_test":   int(len(idx_te)),
            "r2_test":  float(r2_score(y[idx_te], yhat)),
            "mae_test": float(mean_absolute_error(y[idx_te], yhat)),
        })

    overall = {
        "r2_overall":       float(r2_score(y, preds)),
        "mae_overall":      float(mean_absolute_error(y, preds)),
        "baseline_mae_mean": float(np.mean(np.abs(y - np.mean(y)))),
    }
    return {"folds": fold_metrics, "overall": overall, "preds": preds}


def main() -> int:
    args = parse_args()
    if not args.bounds.exists():
        sys.exit(f"Bounds file not found: {args.bounds}")
    bd = np.load(str(args.bounds))
    bounds_meta = {
        "x_min": float(bd["x_min"]),
        "x_max": float(bd["x_max"]),
        "y_min": float(bd["y_min"]),
        "y_max": float(bd["y_max"]),
        "n_chi": int(bd["n_chi"]),
        "seed":  int(bd["seed"]),
    }
    n_dirs = int(bd["n_dirs"])
    xpoints = int(bd["xpoints"])
    thetas = np.asarray(bd["thetas"], dtype=float)
    ect = ECTComputer(
        n_dirs=n_dirs, xpoints=xpoints,
        x_min=bounds_meta["x_min"], x_max=bounds_meta["x_max"],
    )
    feature_dim = bounds_meta["n_chi"] * xpoints
    print(f"[bounds] x=[{bounds_meta['x_min']}, {bounds_meta['x_max']}] @ "
          f"{xpoints}, y=[{bounds_meta['y_min']}, {bounds_meta['y_max']}] @ "
          f"{bounds_meta['n_chi']}, n_dirs={n_dirs}, "
          f"feature_dim={feature_dim}")

    runs = discover_sweep_runs(args.sweep_root.resolve(), args.sweep_pattern)
    if not runs:
        sys.exit(f"No sweep runs matching {args.sweep_pattern}")
    print(f"[sweep] {len(runs)} run dirs")

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.out / "dataset.npz")
    ds = build_dataset(runs, ect, thetas, cache_path, bounds_meta)
    X = ds["X"]
    targets = {
        "lumen_count":      ds["y_count"],
        "lumen_area_ratio": ds["y_ratio"],
        "organoid_radius":  ds["y_radius"],
    }
    pair_array = np.stack([ds["tauV"], ds["xi_pct"]], axis=1)
    _, group_ids = np.unique(pair_array, axis=0, return_inverse=True)
    print(f"[data] X={X.shape}, n_groups={len(set(group_ids.tolist()))}")

    results: Dict[str, Dict] = {}
    for tname, y in targets.items():
        results[tname] = {}
        print(f"\n[regress] target = {tname}")
        for mname in args.models:
            factory = make_model_factory(mname, args.alphas)
            res = evaluate_target(X, y, group_ids, args.kfolds, factory)
            print(f"  model = {mname}")
            for fm in res["folds"]:
                a_str = f"alpha={fm['alpha']:g} " if fm["alpha"] is not None else ""
                print(f"    fold {fm['fold']}: {a_str}"
                      f"R2={fm['r2_test']:.3f} MAE={fm['mae_test']:.4g}")
            ov = res["overall"]
            print(f"    OVERALL  R2={ov['r2_overall']:.3f}  "
                  f"MAE={ov['mae_overall']:.4g}  "
                  f"(baseline mean MAE={ov['baseline_mae_mean']:.4g})")
            results[tname][mname] = {"folds": res["folds"], "overall": ov}

            out_csv = args.out / f"{tname}__{mname}__predictions.csv"
            with out_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tauV", "xi_pct", "t", "y_true", "y_pred"])
                for i in range(len(y)):
                    w.writerow([int(ds["tauV"][i]), int(ds["xi_pct"][i]),
                                int(ds["t"][i]),
                                float(y[i]), float(res["preds"][i])])

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
