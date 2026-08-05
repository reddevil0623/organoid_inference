"""Load an observation (synthetic target or experimental image) for inference.

Returns a uniform object exposing its label, its frames as {0,1,2} label
arrays, and its ground truth where known.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# FAKEXP truth lookup

# Ground truths of the synthetic targets, drawn under a fixed seed.
FAKEXP_TRUTH = {
    # idx 0-2: xi is the 2-dp value the simulator actually receives
    # (f"{xi:.2f}"), i.e. the simulated truth. The pre-rounding RNG draws were
    # xi = 0.1966 / 0.2534 / 0.3146 (tau_V are integers, so unchanged).
    0: (70.0, 0.20),
    1: (77.0, 0.25),
    2: (9.0, 0.31),
    # idx 3-5: new validation points (2026-06-06); integer tau_V + 2-dp xi, so
    # the truth equals exactly what the simulator receives.
    3: (25.0, 0.15),
    4: (45.0, 0.20),
    5: (60.0, 0.28),
}

# Directory naming convention: "fakexp_<idx>_tauV<int>_xi<int*100>_<6-hex>".
_FAKEXP_RE = re.compile(
    r"fakexp_(?P<idx>\d+)_tauV(?P<tauV>\d+)_xi(?P<xi>\d+)_[0-9a-f]+"
)


def find_fakexp_dir(idx: int, search_roots: "list[Path]") -> Optional[Path]:
    """Find a completed fakexp_<idx>_* directory under any of the search roots."""
    for root in search_roots:
        if not root.is_dir():
            continue
        for d in sorted(root.glob(f"fakexp_{idx}_*")):
            if not d.is_dir():
                continue
            if not (d / ".done").exists():
                continue
            if not any(d.glob("u_*.dat")):
                continue
            m = _FAKEXP_RE.match(d.name)
            if m and int(m.group("idx")) == idx:
                return d
    return None


# Observation object

@dataclass(frozen=True)
class Observation:
    label: str
    frames: List[np.ndarray]                 # list of 2-D label arrays
    truth: Optional[Tuple[float, float]]     # (tau_V, xi) or None
    fakexp_idx: Optional[int] = None
    source_dir: Optional[Path] = None


def load_fakexp_observation(
    fakexp_idx: int,
    search_roots: "list[Path]",
    *,
    n_frames: int = 1,
    u_thresh: float = 1e-4,
    s_thresh: float = 0.5,
    obs_t: Optional[int] = None,
) -> Observation:
    """Load a FAKEXP simulator-output dir as a 'pseudo-experimental' observation."""
    import sys as _sys
    from . import _ensure_scripts_on_path
    _ensure_scripts_on_path()

    from core import Simulator, list_field_indices, even_sample, load_field, u_s_to_labels  # noqa: E402
    import paths as _p  # noqa: E402

    outdir = find_fakexp_dir(fakexp_idx, search_roots)
    if outdir is None:
        raise FileNotFoundError(
            f"fakexp idx {fakexp_idx} not found under any of: {search_roots}"
        )

    indices = list_field_indices(outdir)
    if not indices:
        raise RuntimeError(f"No u_*.dat frames in {outdir}")

    if n_frames == 1:
        if obs_t is None:
            t_picked = indices[-1]  # final frame: most-mature state
        else:
            # Snap to the closest available frame.
            t_picked = min(indices, key=lambda x: abs(x - obs_t))
        chosen = [t_picked]
        label = f"fakexp_idx{fakexp_idx}_t{t_picked}"
    else:
        chosen = even_sample(indices, n_frames)
        label = f"fakexp_idx{fakexp_idx}"

    frames: List[np.ndarray] = []
    for t in chosen:
        U = load_field(outdir / f"u_{t}.dat")
        S = load_field(outdir / f"s_{t}.dat")
        frames.append(u_s_to_labels(U, S, u_thresh, s_thresh))

    return Observation(
        label=label,
        frames=frames,
        truth=FAKEXP_TRUTH.get(fakexp_idx),
        fakexp_idx=fakexp_idx,
        source_dir=outdir,
    )
