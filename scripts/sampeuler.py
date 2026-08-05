#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SampEuler featuriser for phase-field organoid label images.

Summarises each frame as a fixed-length real vector. For each of three binary
masks (cells, lumen, organoid) and each direction, computes the Euler
characteristic curve of the cubical complex and vectorises it on a per-channel
grid. Bounds are per channel because the masks have different filtration
supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

import paths as _p
_p.ensure_eucalc()
import eucalc as ec


CHANNELS: Tuple[str, ...] = ("cells", "lumen", "organoid")

# Directions on which to compute the EC transform. For approximately
# rotationally symmetric organoid shapes, two orthogonal directions
# already capture most of the discriminative content; extending to more
# directions only doubles the feature length, never reduces it.
DIRECTIONS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
)

DEFAULT_CANVAS = 192       # pixels (square, after centred padding)
DEFAULT_XPOINTS = 64       # samples along each EC curve
DEFAULT_U_THRESH = 1e-4
DEFAULT_S_THRESH = 0.5


def frame_to_label(u_path: Path, s_path: Path,
                   u_thresh: float = DEFAULT_U_THRESH,
                   s_thresh: float = DEFAULT_S_THRESH) -> np.ndarray:
    U = np.loadtxt(str(u_path))
    S = np.loadtxt(str(s_path))
    L = np.zeros_like(U, dtype=np.uint8)
    L[U > u_thresh] = 1
    L[S > s_thresh] = 2
    return L


def _bbox(arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(arr > 0)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def centre_pad(label: np.ndarray, canvas: int = DEFAULT_CANVAS) -> np.ndarray:
    """Crop to nonzero bbox, then centre on a square canvas of given side."""
    bb = _bbox(label)
    if bb is None:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    y0, y1, x0, x1 = bb
    crop = label[y0:y1+1, x0:x1+1]
    h, w = crop.shape
    if h > canvas or w > canvas:
        from PIL import Image
        scale = min(canvas / h, canvas / w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        crop = np.array(
            Image.fromarray(crop, mode="L").resize((new_w, new_h),
                                                   resample=Image.NEAREST),
            dtype=np.uint8)
        h, w = crop.shape
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    yo = (canvas - h) // 2
    xo = (canvas - w) // 2
    out[yo:yo+h, xo:xo+w] = crop
    return out


def channel_masks(label: np.ndarray) -> List[np.ndarray]:
    return [
        (label == 1).astype(np.uint8),
        (label == 2).astype(np.uint8),
        (label >  0).astype(np.uint8),
    ]


def _ec_support(mask: np.ndarray, direction: np.ndarray
                ) -> Optional[Tuple[float, float]]:
    """Return (min, max) finite filtration values where chi changes, or
    None if the mask is empty."""
    if mask.sum() == 0:
        return None
    cplx = ec.EmbeddedComplex(mask)
    cplx.preproc_ect()
    ect = cplx.compute_euler_characteristic_transform(direction)
    heights = np.array(ect.get_attributes()[0], dtype=float)
    finite = heights[np.isfinite(heights)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def compute_bounds(label_iter: Iterable[np.ndarray],
                   directions: Tuple[Tuple[float, float], ...] = DIRECTIONS,
                   quantile: Tuple[float, float] = (0.0, 1.0),
                   ) -> np.ndarray:
    """Pass over labels, return per-channel global bounds."""
    n_ch = len(CHANNELS)
    mins = [[] for _ in range(n_ch)]
    maxs = [[] for _ in range(n_ch)]
    for L in label_iter:
        masks = channel_masks(L)
        for c, m in enumerate(masks):
            for d in directions:
                supp = _ec_support(m, np.array(d, dtype=float))
                if supp is None:
                    continue
                mins[c].append(supp[0])
                maxs[c].append(supp[1])
    out = np.zeros((n_ch, 2), dtype=float)
    q_lo, q_hi = quantile
    for c in range(n_ch):
        if not mins[c]:
            out[c] = (-0.5, 0.5)
            continue
        out[c, 0] = float(np.quantile(mins[c], q_lo))
        out[c, 1] = float(np.quantile(maxs[c], q_hi))
    return out


class SampEulerVectoriser:
    """Featurise label images into fixed-length SampEuler vectors."""

    def __init__(self, bounds: np.ndarray,
                 xpoints: int = DEFAULT_XPOINTS,
                 directions: Tuple[Tuple[float, float], ...] = DIRECTIONS,
                 canvas: int = DEFAULT_CANVAS):
        if bounds.shape != (len(CHANNELS), 2):
            raise ValueError(
                f"bounds must have shape ({len(CHANNELS)}, 2), got {bounds.shape}")
        self.bounds = bounds.astype(float)
        self.xpoints = int(xpoints)
        self.directions = tuple(directions)
        self.canvas = int(canvas)

    @property
    def feature_dim(self) -> int:
        return len(CHANNELS) * len(self.directions) * self.xpoints

    def __call__(self, label: np.ndarray) -> np.ndarray:
        L = label if label.shape[0] == self.canvas else centre_pad(
            label, self.canvas)
        masks = channel_masks(L)
        out = np.zeros(self.feature_dim, dtype=np.float32)
        offset = 0
        for c, mask in enumerate(masks):
            x_min, x_max = self.bounds[c]
            if mask.sum() == 0:
                offset += len(self.directions) * self.xpoints
                continue
            cplx = ec.EmbeddedComplex(mask)
            cplx.preproc_ect()
            for d in self.directions:
                ect = cplx.compute_euler_characteristic_transform(
                    np.array(d, dtype=float))
                vec = np.array(ect.vectorize(x_min, x_max, self.xpoints),
                               dtype=np.float32)
                out[offset:offset + self.xpoints] = vec
                offset += self.xpoints
        return out

    def save(self, path: Path):
        np.savez(str(path),
                 bounds=self.bounds,
                 xpoints=np.int64(self.xpoints),
                 directions=np.array(self.directions, dtype=float),
                 canvas=np.int64(self.canvas))

    @classmethod
    def load(cls, path: Path) -> "SampEulerVectoriser":
        d = np.load(str(path))
        return cls(
            bounds=d["bounds"],
            xpoints=int(d["xpoints"]),
            directions=tuple(map(tuple, d["directions"].tolist())),
            canvas=int(d["canvas"]),
        )


def lumen_count(label: np.ndarray) -> int:
    """Number of connected components in the lumen mask (4-connectivity)."""
    from scipy.ndimage import label as cc_label
    _, n = cc_label((label == 2).astype(np.uint8))
    return int(n)


def lumen_area_ratio(label: np.ndarray) -> float:
    organ = int((label > 0).sum())
    if organ == 0:
        return 0.0
    return float((label == 2).sum()) / organ


def iter_run_frames(run_dir: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (t, label) pairs for every (u_t, s_t) pair found under run_dir."""
    ts = sorted(int(p.stem.split("_", 1)[1])
                for p in run_dir.glob("u_*.dat"))
    for t in ts:
        u = run_dir / f"u_{t}.dat"
        s = run_dir / f"s_{t}.dat"
        if not s.exists():
            continue
        yield t, frame_to_label(u, s)


def stride_run_frames(run_dir: Path, stride: int = 5
                      ) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield every k-th frame from run_dir (output_time-ordered)."""
    ts = sorted(int(p.stem.split("_", 1)[1])
                for p in run_dir.glob("u_*.dat"))
    for t in ts[::max(1, stride)]:
        u = run_dir / f"u_{t}.dat"
        s = run_dir / f"s_{t}.dat"
        if not s.exists():
            continue
        yield t, frame_to_label(u, s)
