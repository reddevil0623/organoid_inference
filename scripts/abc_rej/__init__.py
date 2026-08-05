"""ABC rejection on a shared simulator grid.

Draws simulator runs i.i.d. from a uniform prior and caches every output
frame's SampEuler descriptor. For each observation, distances to all cached
frames are computed and the closest fixed fraction accepted. The simulation
pool is shared across observations, so every run contributes to every
posterior. Each frame is one candidate (tau_V, xi, t) at equal weight, which
targets the uniform prior restricted to the reachable region.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path


_PROJECT_ROOT = _Path(__file__).resolve().parents[2]


def _ensure_scripts_on_path() -> None:
    """Add the project's scripts/ to sys.path so `import core, paths` works."""
    s = str(_PROJECT_ROOT / "scripts")
    if s not in _sys.path:
        _sys.path.insert(0, s)


_ensure_scripts_on_path()
