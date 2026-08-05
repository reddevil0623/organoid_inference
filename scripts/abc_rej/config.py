"""Settings, prior sampling, and cache paths for the ABC rejection module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


# Static layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "ABC_REJ"

# Sub-directories under OUTPUTS_DIR (NFS-shared across hosts):
#   cache/             featurised per-sim NPZs    (one per sim, sampled frames only)
#   cache/meta/        per-sim metadata NPZs       (Stage A1 output)
#   cache/manifests/   per-host CSV manifests      (NFS-safe append-only)
#   cache/work_lists/  per-host featurisation lists (Stage A2 controller output)
#   observations/      per-observation SampEuler features (cached)
#   distances/         per-observation distance tables
#   posteriors/        per-observation 2D KDE plots
CACHE_DIR = OUTPUTS_DIR / "cache"
META_DIR = CACHE_DIR / "meta"
WORK_LIST_DIR = CACHE_DIR / "work_lists"
OBS_DIR = OUTPUTS_DIR / "observations"
DIST_DIR = OUTPUTS_DIR / "distances"
POST_DIR = OUTPUTS_DIR / "posteriors"

SIM_MANIFEST = CACHE_DIR / "sims_manifest.csv"  # merged manifest (read-only); built lazily from per-host files
PER_HOST_MANIFEST_DIR = CACHE_DIR / "manifests"  # per-host CSVs to avoid NFS-concurrent append corruption
SIM_MANIFEST_COLS = [
    "theta_idx",       # int, ABC sim index
    "tauV",            # float (continuous prior); int actually fed to simulator
    "tauV_int",        # int actually used
    "xi",              # float, two decimal places
    "T_end",           # int, last simulator-time index produced (matches u_<t>.dat)
    "n_frames",        # int, total number of (u, s) frames the simulator wrote
    "wall_seconds",    # float, simulator wall time
    "outdir",          # str, simulator output directory on the per-host scratch
    "hostname",        # str, host that ran this sim (needed for Stage A2 host-aware dispatch)
    "meta_path",       # str, NFS path to the sim_meta_NNNNN.npz produced in Stage A1
    "error",           # str or empty
]


# Configuration

@dataclass(frozen=True)
class Config:
    # Priors (matches FAKEXP CONFIG)
    prior_tauV: Tuple[float, float] = (1.0, 90.0)
    prior_xi: Tuple[float, float] = (0.10, 0.32)

    # Simulator settings (matches FAKEXP CONFIG)
    n_cells: int = 4
    rs: float = 0.70
    u_thresh: float = 1e-4
    s_thresh: float = 0.5

    # Frame retention at simulation time: we featurise EVERY disk frame
    # the simulator wrote. Sub-sampling is deferred to the distance step
    # (see `n_pool_samples` below), where a uniform random subset is
    # drawn from the global pool of (theta_idx, frame_idx) pairs across
    # the entire cache. This preserves per-sim cache independence
    # (idempotent + resumable) and lets the same cache serve different
    # pool-sampling seeds / sizes without re-running simulations.

    # Pool sampling at distance time: draw this many (theta_idx,
    # frame_idx) pairs uniformly without replacement from the global
    # pool across the whole cache.
    n_pool_samples: int = 30000
    # Fixed seed for the pool draw so every observation processes the
    # same candidates, and so runs are reproducible.
    pool_seed: int = 42

    # SampEuler / ECT (matches data/sampeuler_bounds.npz)
    n_dirs: int = 100
    xpoints: int = 600
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -10.0
    y_max: float = 70.0
    n_chi: int = 80
    sampeuler_seed: int = 42  # matches bounds file

    # ABC rejection
    n_sims: int = 3000
    seed: int = 42
    accept_quantile: float = 0.05  # epsilon = 5th percentile of distances

    # Sampling design: 'lhs' or 'uniform' (i.i.d. uniform from prior).
    # Default 'uniform' is the ABC rejection design; 'lhs' is space-filling
    # but needs pyDOE.
    sampling: str = "uniform"

    # Per-host parallelism
    n_jobs: int = 25

    # Paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # Distance metric. Default 'hungarian' = Hungarian-matched Wasserstein-1
    # between the raw ECT curves underlying the SampEuler vectorisation.
    # Set to 'l2' for the cheaper Euclidean baseline on flattened
    # SampEuler images.
    distance: str = "hungarian"

    @property
    def model_repo(self) -> Path:
        return self.project_root / "MCPFM_tauV-model"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs" / "ABC_REJ"


CONFIG = Config()


def prior_box(cfg: Config) -> "list[Tuple[float, float]]":
    return [cfg.prior_tauV, cfg.prior_xi]


def lhs_samples(n: int, cfg: Config, *, seed: int = None) -> np.ndarray:
    """Latin hypercube samples in (tau_V, xi) prior box."""
    try:
        from pyDOE import lhs
    except ImportError as e:
        raise ImportError(
            "pyDOE is required for LHS sampling: pip install pyDOE"
        ) from e

    rng_seed = cfg.seed + 1 if seed is None else seed
    np.random.seed(rng_seed)  # pyDOE uses np.random
    unit = lhs(2, samples=n, criterion="maximin", iterations=20)
    lo = np.array([cfg.prior_tauV[0], cfg.prior_xi[0]])
    hi = np.array([cfg.prior_tauV[1], cfg.prior_xi[1]])
    return lo + (hi - lo) * unit


def uniform_samples(n: int, cfg: Config, *, seed: int = None) -> np.ndarray:
    """i.i.d. uniform samples in (tau_V, xi) prior box."""
    rng = np.random.default_rng(cfg.seed + 1 if seed is None else seed)
    out = np.empty((n, 2))
    out[:, 0] = rng.uniform(cfg.prior_tauV[0], cfg.prior_tauV[1], size=n)
    out[:, 1] = rng.uniform(cfg.prior_xi[0], cfg.prior_xi[1], size=n)
    return out


def make_thetas(n: int, cfg: Config) -> np.ndarray:
    """Choose LHS or i.i.d. uniform sampling based on cfg.sampling."""
    if cfg.sampling == "lhs":
        return lhs_samples(n, cfg)
    if cfg.sampling == "uniform":
        return uniform_samples(n, cfg)
    raise ValueError(f"Unknown sampling: {cfg.sampling}")


# SampEuler bounds I/O

DEFAULT_BOUNDS_FILE = PROJECT_ROOT / "data" / "sampeuler_bounds.npz"


def load_sampeuler_bounds(path: Path = DEFAULT_BOUNDS_FILE) -> dict:
    """Load the SampEuler bounds spec."""
    if not path.exists():
        return {
            "x_min": float(CONFIG.x_min),
            "x_max": float(CONFIG.x_max),
            "xpoints": int(CONFIG.xpoints),
            "y_min": float(CONFIG.y_min),
            "y_max": float(CONFIG.y_max),
            "n_chi": int(CONFIG.n_chi),
            "n_dirs": int(CONFIG.n_dirs),
            "seed": int(CONFIG.sampeuler_seed),
            "thetas": None,
        }
    blob = np.load(str(path), allow_pickle=False)
    return {
        "x_min": float(blob["x_min"]),
        "x_max": float(blob["x_max"]),
        "xpoints": int(blob["xpoints"]),
        "y_min": float(blob["y_min"]),
        "y_max": float(blob["y_max"]),
        "n_chi": int(blob["n_chi"]),
        "n_dirs": int(blob["n_dirs"]),
        "seed": int(blob["seed"]),
        "thetas": np.asarray(blob["thetas"], dtype=float),
    }
