#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared classes and frame helpers for phase-field model inference.

Classes:
    Simulator           – Run the C++ MCPFM model
    ECTComputer         – Euler Characteristic Transform via eucalc
    WassersteinDistance – Distance metrics on ECT curves
"""

from __future__ import annotations

import math, sys, time, subprocess, shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import paths as _p
_p.ensure_eucalc()
import eucalc as ec


def xi_tag(xi: float) -> str:
    return f"{int(round(xi * 100)):03d}"


def load_field(path: Path) -> np.ndarray:
    return np.loadtxt(str(path))


def list_field_indices(run_dir: Path) -> List[int]:
    out = []
    for p in run_dir.glob("u_*.dat"):
        try:
            out.append(int(p.stem.split("_", 1)[1]))
        except Exception:
            pass
    out.sort()
    return out


def even_sample(ts: List[int], n: int) -> List[int]:
    if not ts:
        return []
    if n >= len(ts):
        return ts
    idxs = np.linspace(0, len(ts) - 1, num=n)
    return sorted({ts[int(round(x))] for x in idxs})


def u_s_to_labels(U, S, u_thresh=1e-4, s_thresh=0.5):
    L = np.zeros_like(U, dtype=np.uint8)
    L[U > u_thresh] = 1
    L[S > s_thresh] = 2
    return L


def labels_to_gray(labels):
    lut = np.array([0, 255, 128], dtype=np.uint8)
    return lut[np.clip(labels, 0, 2).astype(int)]


def nonzero_bbox(arr):
    mask = arr > 0
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def pad_to(arr, h, w, fill=0, center=False):
    out = np.full((h, w), fill, dtype=arr.dtype)
    oh, ow = arr.shape[:2]
    hh, ww = min(oh, h), min(ow, w)
    if center:
        y0 = (h - hh) // 2
        x0 = (w - ww) // 2
    else:
        y0, x0 = 0, 0
    out[y0:y0+hh, x0:x0+ww] = arr[:hh, :ww]
    return out


def resize_nearest(arr, new_w, new_h):
    pil = Image.fromarray(arr, mode="L")
    pil = pil.resize((new_w, new_h), resample=Image.NEAREST)
    return np.array(pil, dtype=np.uint8)


def organoid_diameter(lbl):
    """Minimum enclosing circle diameter of the whole organoid (lbl > 0)."""
    ys, xs = np.where(lbl > 0)
    if ys.size == 0:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    try:
        import cv2
        _, radius = cv2.minEnclosingCircle(pts)
        return 2.0 * float(radius)
    except ImportError:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist
        if pts.shape[0] < 2:
            return 0.0
        if pts.shape[0] == 2:
            return float(pdist(pts).max())
        try:
            hull = ConvexHull(pts)
            return float(pdist(pts[hull.vertices]).max())
        except Exception:
            return float(pdist(pts).max())


def rescale_to_diameter(lbl, target_diameter):
    """Rescale a label image so its organoid MEC diameter matches target_diameter."""
    cur = organoid_diameter(lbl)
    if cur <= 0 or target_diameter <= 0:
        return lbl
    scale = float(target_diameter) / float(cur)
    new_h = max(1, int(round(lbl.shape[0] * scale)))
    new_w = max(1, int(round(lbl.shape[1] * scale)))
    return resize_nearest(lbl, new_w, new_h)


def iter_run_frames(run_dir: Path):
    """Yield (t, label) for every (u_t, s_t) pair under ``run_dir``."""
    ts = sorted(int(p.stem.split("_", 1)[1])
                for p in run_dir.glob("u_*.dat"))
    for t in ts:
        u = run_dir / f"u_{t}.dat"
        s = run_dir / f"s_{t}.dat"
        if not s.exists():
            continue
        yield t, u_s_to_labels(load_field(u), load_field(s))


# Morphometric scalars, following Lee et al. (Nat. Cell Biol. 28, 113-124, 2026).

def lumen_count(label: np.ndarray) -> int:
    """Number of 4-connected components in the lumen mask (label == 2)."""
    from scipy.ndimage import label as cc_label
    _, n = cc_label((label == 2).astype(np.uint8))
    return int(n)


def lumen_area_ratio(label: np.ndarray) -> float:
    """A_lumen / A_organoid (Lee et al.'s 2D lumen occupancy from the
    mid-plane). Returns 0 if no organoid foreground."""
    organ = int((label > 0).sum())
    if organ == 0:
        return 0.0
    return float((label == 2).sum()) / organ


def organoid_radius(label: np.ndarray) -> float:
    """Minimum-enclosing-circle radius of the whole organoid (label > 0),
    in pixels. Wraps ``organoid_diameter`` so callers can request the
    radius directly."""
    return 0.5 * organoid_diameter(label)


class Simulator:
    """Manages C++ phase-field model execution and scratch directory setup."""

    def __init__(self, model_repo: Path,
                 scratch: Optional[Path] = None,
                 n_cells: Optional[int] = None):
        self.model_repo = model_repo

        if scratch is not None:
            self.cwd, self.data_dir = self._setup_scratch(scratch, n_cells)
        else:
            self.cwd = model_repo
            self.data_dir = model_repo / "DATA"
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._setup_local()

        self._validate()

    def _setup_scratch(self, scratch_base: Path,
                       n_cells: Optional[int]) -> Tuple[Path, Path]:
        if n_cells is not None:
            scratch = scratch_base / f"N{n_cells}"
        else:
            scratch = scratch_base
        scratch.mkdir(parents=True, exist_ok=True)
        data = scratch / "DATA"
        data.mkdir(exist_ok=True)

        symlinks = [
            ("inputs", self.model_repo / "inputs"),
            ("IN", self.model_repo / "inputs"),
            ("INPUT", self.model_repo / "inputs"),
            ("run", self.model_repo / "run"),
        ]
        if n_cells is not None:
            symlinks.append(
                (f"init_cells_{n_cells}",
                 self.model_repo / "inputs" / f"init_cells_{n_cells}"))
        else:
            symlinks.append(
                ("init_cells_4",
                 self.model_repo / "inputs" / "init_cells_4"))

        for name, target in symlinks:
            link = scratch / name
            if not link.exists():
                try:
                    link.symlink_to(target)
                except FileExistsError:
                    pass
        return scratch, data

    def _setup_local(self):
        for src, tgt in [("IN", "inputs"), ("INPUT", "inputs"),
                         ("init_cells_4", "inputs/init_cells_4")]:
            p = self.model_repo / src
            if not p.exists():
                try:
                    p.symlink_to(tgt)
                except Exception:
                    pass

    def _validate(self):
        if not (self.cwd / "run").exists() and not (self.model_repo / "run").exists():
            sys.exit("ERROR: ./run not found. Compile first.")

    def run(self, label: str, n_cells: int, tauV: int, xi: float,
            rs: float, overwrite: bool = False) -> Path:
        outdir = self.data_dir / label
        outdir.mkdir(parents=True, exist_ok=True)

        if not overwrite and (outdir / ".done").exists():
            return outdir

        log = outdir / "out"
        (outdir / ".started").write_text(str(time.time()))
        with log.open("w") as lf:
            ret = subprocess.run(
                ["./run", label, str(n_cells), str(int(round(tauV))),
                 f"{xi:.2f}", f"{rs:.2f}"],
                cwd=str(self.cwd), stdout=lf, stderr=subprocess.STDOUT
            ).returncode
        if ret == 0:
            (outdir / ".done").write_text(str(time.time()))
        return outdir

    def run_checked(self, label: str, n_cells: int, tauV: int, xi: float,
                    rs: float) -> Path:
        outdir = self.run(label, n_cells, tauV, xi, rs)
        if not (outdir / ".done").exists():
            raise RuntimeError(f"Simulator failed for {label}")
        return outdir

    def find_completed(self, prefix: str) -> Optional[Path]:
        for d in self.data_dir.glob(f"{prefix}*"):
            if d.is_dir() and (d / ".done").exists():
                return d
        return None

    def clean_incomplete(self):
        if not self.data_dir.is_dir():
            return 0
        removed = []
        for d in self.data_dir.iterdir():
            if d.is_dir() and (d / ".started").exists() \
               and not (d / ".done").exists():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name)
        if removed:
            print(f"  Cleaned {len(removed)} incomplete sim dir(s)")
        return len(removed)

    def load_label_frames(self, run_dir: Path, n_frames: int,
                          u_thresh=1e-4, s_thresh=0.5):
        indices = list_field_indices(run_dir)
        if not indices:
            raise RuntimeError(f"No u_*.dat frames in {run_dir}")
        chosen = even_sample(indices, n_frames)
        frames = []
        for t in chosen:
            U = load_field(run_dir / f"u_{t}.dat")
            S = load_field(run_dir / f"s_{t}.dat")
            frames.append(u_s_to_labels(U, S, u_thresh, s_thresh))
        return frames, chosen


class ECTComputer:
    """Euler Characteristic Transform computation via eucalc."""

    def __init__(self, n_dirs=720, xpoints=3000,
                 x_min=-1.5, x_max=1.5,):
        self.n_dirs = n_dirs
        self.xpoints = xpoints
        self.x_min = x_min
        self.x_max = x_max
        self.T = np.linspace(x_min, x_max, xpoints)
        self.delta_x = (x_max - x_min) / max(1, xpoints - 1)

    def compute(self, img_uint8: np.ndarray,
                rng: Optional[np.random.Generator] = None,
                thetas: Optional[np.ndarray] = None) -> np.ndarray:
        if thetas is None:
            if rng is None:
                raise ValueError("Either rng or thetas must be provided.")
            thetas = rng.uniform(0, 2 * np.pi, self.n_dirs)

        cplx = ec.EmbeddedComplex(img_uint8)
        cplx.preproc_ect()
        out = np.empty((len(thetas), self.xpoints), dtype=float)
        for i, theta in enumerate(thetas):
            d = np.array((np.sin(theta), np.cos(theta)))
            euler = cplx.compute_euler_characteristic_transform(d)
            out[i] = [euler.evaluate(t) for t in self.T]
        return out

    def compute_with_seed(self, img_uint8: np.ndarray,
                          seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.compute(img_uint8, rng=rng)


class WassersteinDistance:
    """Wasserstein-type distances between ECT curve sets."""

    def __init__(self, delta_x: float, method: str = "hungarian"):
        self.delta_x = delta_x
        self.method = method

    def __call__(self, ects_a, ects_b) -> float:
        if self.method == "hungarian":
            return self.hungarian_l1(ects_a, ects_b)
        elif self.method == "elliptical":
            return self.elliptical(ects_a, ects_b)
        elif self.method == "sliced":
            return self.sliced(ects_a, ects_b)
        raise ValueError(f"Unknown method: {self.method}")

    def hungarian_l1(self, A, B) -> float:
        # Callers pass ects as a Python list of 2D arrays (one per frame),
        # each shaped (n_dirs, xpoints). np.vstack collapses them to
        # (total_curves, xpoints) and is a no-op on arrays already in that
        # shape, so both single-frame and multi-frame inputs work.
        A = np.vstack(A)
        B = np.vstack(B)
        C = np.sum(np.abs(A[:, None, :] - B[None, :, :]),
                   axis=2) * self.delta_x
        r, c = linear_sum_assignment(C)
        return float(np.mean(C[r, c]))

    def elliptical(self, ects_a: list, ects_b: list) -> float:
        A = np.vstack(ects_a) * self.delta_x
        B = np.vstack(ects_b) * self.delta_x
        mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
        std_a = A.std(axis=0) + 1e-12
        std_b = B.std(axis=0) + 1e-12
        w2 = float(np.sum((mu_a - mu_b)**2) + np.sum((std_a - std_b)**2))
        return math.sqrt(max(0.0, w2))

    def sliced(self, ects_a: list, ects_b: list) -> float:
        n = min(len(ects_a), len(ects_b))
        if n == 0:
            return float("inf")
        total = 0.0
        for f in range(n):
            total += float(np.mean(np.sum(
                np.abs(ects_a[f] - ects_b[f]), axis=1)) * self.delta_x)
        return total / n
