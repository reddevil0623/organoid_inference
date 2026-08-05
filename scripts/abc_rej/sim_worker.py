"""Per-theta workers for the ABC rejection grid.

``simulate_only`` runs one simulation at (tau_V, xi) and writes its metadata.
``featurise_sampled`` later computes ECT + SampEuler for only the frames that
survived pool sampling.
"""
from __future__ import annotations

import os
import socket
import sys as _sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make the abc_rej package import the project scripts/ modules cleanly
# both in the parent process and in forked workers.
from . import _ensure_scripts_on_path
_ensure_scripts_on_path()

from .config import (
    Config,
    CONFIG,
    DEFAULT_BOUNDS_FILE,
    load_sampeuler_bounds,
)


# Pure helpers (no simulator access)

def xi_tag(xi: float) -> str:
    """Match the FAKEXP labelling convention (xi is rounded to 2 d.p.)."""
    return f"{int(round(xi * 100)):03d}"


def compute_ect_curves(
    frame_label: np.ndarray,
    *,
    ect,
    bounds: dict,
    rng: "Optional[np.random.Generator]" = None,
) -> np.ndarray:
    """Compute the raw ECT curves of a single (synthetic-simulator) label frame."""
    if frame_label.sum() == 0:
        return np.zeros((int(bounds["n_dirs"]), int(bounds["xpoints"])),
                        dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()
    curves = ect.compute(frame_label, rng=rng)  # (n_dirs, xpoints), {0,1,2} labels
    return curves.astype(np.float32)


def vectorise_sampeuler(
    curves: np.ndarray,
    *,
    ect,
    bounds: dict,
) -> np.ndarray:
    """Flattened vectorised-SampEuler image -- DISABLED placeholder."""
    return np.zeros(int(bounds["n_chi"]) * int(bounds["xpoints"]),
                    dtype=np.float32)


def featurise_frame(
    frame_label: np.ndarray,
    *,
    canvas_side: int = 1024,    # kept for backward compat; unused for synthetic frames
    ect,
    bounds: dict,
    rng: "Optional[np.random.Generator]" = None,
) -> np.ndarray:
    """Compute ECT + SampEuler vectorisation in one go."""
    curves = compute_ect_curves(frame_label, ect=ect, bounds=bounds, rng=rng)
    return vectorise_sampeuler(curves, ect=ect, bounds=bounds)


def normalize_label_frame(
    frame_label: np.ndarray,
    canvas_side: int,
):
    """Normalise a label frame to the common canvas."""
    from core import nonzero_bbox, resize_nearest, pad_to, labels_to_gray  # noqa: E402

    bb = nonzero_bbox(frame_label)
    crop = (frame_label[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1] if bb is not None
            else frame_label)
    h, w = crop.shape
    if h > canvas_side or w > canvas_side:
        scale = canvas_side / max(h, w)
        crop = resize_nearest(crop, max(1, int(w * scale)),
                              max(1, int(h * scale)))
    L_pad = pad_to(crop, canvas_side, canvas_side, center=True)
    return labels_to_gray(L_pad)


# Stage A1: simulate_only — run sim, write metadata-only NPZ

def simulate_only(
    theta_idx: int,
    tauV: float,
    xi: float,
    scratch_root: str,
    cache_root: str,
    worker_id: int,
    *,
    n_cells: int = 4,
    rs: float = 0.70,
) -> dict:
    """Run one simulation; write only its metadata to NFS."""
    _ensure_scripts_on_path()

    cache_root = Path(cache_root)
    meta_dir = cache_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"sim_meta_{theta_idx:05d}.npz"
    hostname = socket.gethostname().split(".")[0]

    # Resume: cached metadata exists.
    # Note: we report `hostname` as the *original* sim's host (from the
    # NPZ), not the current one. That's the authoritative owner of the
    # scratch outdir — Stage A2's host-aware dispatch needs to send the
    # featurisation work to that host. The local per-host manifest will
    # therefore contain rows whose `hostname` column refers to a
    # different host; that's intentional.
    if meta_path.exists():
        try:
            blob = np.load(meta_path, allow_pickle=False)
            return {
                "theta_idx": int(theta_idx),
                "tauV": float(tauV),
                "tauV_int": int(blob.get("tauV_int", int(round(tauV)))),
                "xi": float(xi),
                "T_end": int(blob["T_end"]),
                "n_frames": int(blob["n_frames"]),
                "wall_seconds": float(blob.get("wall_seconds", 0.0)),
                "outdir": str(blob["outdir"]),
                "hostname": str(blob.get("hostname", hostname)),
                "meta_path": str(meta_path),
                "error": "",
            }
        except Exception:
            # Corrupt metadata; regenerate.
            try:
                meta_path.unlink()
            except FileNotFoundError:
                pass

    tauV_int = int(round(tauV))
    label_prefix = f"abcrej_{theta_idx:05d}_tauV{tauV_int}_xi{xi_tag(xi)}_"

    t0 = time.time()
    try:
        from core import Simulator, list_field_indices  # noqa: E402
    except Exception as e:  # pragma: no cover
        return _err_meta_manifest(theta_idx, tauV, xi, t0, hostname,
                                  f"core import failed: {e}")

    worker_scratch = Path(scratch_root) / f"w{worker_id:02d}"
    worker_scratch.mkdir(parents=True, exist_ok=True)

    # ── Run (or reuse) the simulator ──
    try:
        import paths as _p
        sim = Simulator(model_repo=_p.MODEL_REPO,
                        scratch=worker_scratch, n_cells=n_cells)
        outdir = sim.find_completed(label_prefix)
        if outdir is None:
            label = label_prefix + uuid.uuid4().hex[:6]
            outdir = sim.run_checked(label, n_cells, tauV_int, float(xi), rs)
    except Exception as e:
        return _err_meta_manifest(theta_idx, tauV, xi, t0, hostname,
                                  f"simulator failed: {e}")

    # ── Inventory frames written to disk ──
    try:
        all_indices = list_field_indices(outdir)
    except Exception as e:
        return _err_meta_manifest(theta_idx, tauV, xi, t0, hostname,
                                  f"list_field_indices failed: {e}")
    if not all_indices:
        return _err_meta_manifest(theta_idx, tauV, xi, t0, hostname,
                                  f"no u_*.dat frames in {outdir}")

    t_arr = np.asarray(all_indices, dtype=np.int64)
    T_end = int(t_arr.max())
    n_frames = int(len(t_arr))
    wall_seconds = time.time() - t0

    # ── Atomic write of metadata NPZ ──
    # Tmp filename MUST end in '.npz' — otherwise np.savez_compressed
    # silently auto-appends '.npz' to the path we passed, and our
    # subsequent os.replace(tmp, final) fails because tmp doesn't exist
    # under the name we gave it.
    try:
        tmp_path = meta_path.with_suffix(
            f".tmp.{hostname}.{os.getpid()}.{uuid.uuid4().hex[:6]}.npz"
        )
        np.savez_compressed(
            tmp_path,
            theta_idx=np.int64(theta_idx),
            tauV=np.float64(tauV),
            tauV_int=np.int64(tauV_int),
            xi=np.float64(xi),
            T_end=np.int64(T_end),
            n_frames=np.int64(n_frames),
            t_indices=t_arr,
            outdir=np.array(str(outdir)),
            hostname=np.array(hostname),
            wall_seconds=np.float64(wall_seconds),
        )
        os.replace(tmp_path, meta_path)
    except Exception as e:
        try:
            tmp_path.unlink()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return _err_meta_manifest(theta_idx, tauV, xi, t0, hostname,
                                  f"meta save failed: {e}")

    return {
        "theta_idx": int(theta_idx),
        "tauV": float(tauV),
        "tauV_int": int(tauV_int),
        "xi": float(xi),
        "T_end": T_end,
        "n_frames": n_frames,
        "wall_seconds": float(wall_seconds),
        "outdir": str(outdir),
        "hostname": hostname,
        "meta_path": str(meta_path),
        "error": "",
    }


def _err_meta_manifest(theta_idx, tauV, xi, t0, hostname, msg) -> dict:
    return {
        "theta_idx": int(theta_idx),
        "tauV": float(tauV),
        "tauV_int": int(round(float(tauV))),
        "xi": float(xi),
        "T_end": -1,
        "n_frames": 0,
        "wall_seconds": float(time.time() - t0),
        "outdir": "",
        "hostname": hostname,
        "meta_path": "",
        "error": msg,
    }


# Stage A2: featurise_sampled — featurise a specific list of frames

def featurise_sampled(
    theta_idx: int,
    sampled_t_indices: "List[int]",
    outdir: str,
    cache_root: str,
    *,
    u_thresh: float = 1e-4,
    s_thresh: float = 0.5,
    canvas_side: int = 1024,
    bounds_path: Optional[str] = None,
) -> dict:
    """Featurise only the requested model-time frames from one sim."""
    _ensure_scripts_on_path()

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    npz_path = cache_root / f"abcrej_{theta_idx:05d}.npz"
    meta_path = cache_root / "meta" / f"sim_meta_{theta_idx:05d}.npz"

    sorted_t = sorted(int(t) for t in sampled_t_indices)
    if not sorted_t:
        return _err_feat(theta_idx, "empty sampled_t_indices")

    # Resume / idempotency: cached NPZ already has these exact frames.
    if npz_path.exists():
        try:
            cached = np.load(npz_path, allow_pickle=False)
            cached_t = np.asarray(cached["t"], dtype=np.int64).tolist()
            if sorted(cached_t) == sorted_t:
                return {
                    "theta_idx": int(theta_idx),
                    "n_frames": int(cached["n_frames"]),
                    "npz_path": str(npz_path),
                    "error": "",
                }
        except Exception:
            try:
                npz_path.unlink()
            except FileNotFoundError:
                pass

    # Load metadata for tauV / xi / T_end.
    try:
        meta = np.load(meta_path, allow_pickle=False)
        tauV = float(meta["tauV"])
        tauV_int = int(meta["tauV_int"])
        xi = float(meta["xi"])
        meta_T_end = int(meta["T_end"])
    except Exception as e:
        return _err_feat(theta_idx, f"meta load failed: {e}")

    try:
        from core import (
            ECTComputer,
            load_field,
            u_s_to_labels,
        )
    except Exception as e:  # pragma: no cover
        return _err_feat(theta_idx, f"core import failed: {e}")

    bounds = load_sampeuler_bounds(
        Path(bounds_path) if bounds_path else DEFAULT_BOUNDS_FILE
    )
    try:
        ect = ECTComputer(
            n_dirs=int(bounds["n_dirs"]),
            xpoints=int(bounds["xpoints"]),
            x_min=float(bounds["x_min"]),
            x_max=float(bounds["x_max"]),
        )
    except Exception as e:
        return _err_feat(theta_idx, f"ECTComputer init failed: {e}")

    outdir_path = Path(outdir)
    feats: List[np.ndarray] = []
    ect_list: List[np.ndarray] = []
    frame_t_indices: List[int] = []
    # Synthetic-only: skip the bbox-crop / resize / pad-to-canvas step --
    # the simulator output is already at canonical fixed size with the
    # organoid initialised at the centre. Theta angles are drawn fresh per
    # call from `np.random.default_rng()`.
    for t in sorted_t:
        try:
            U = load_field(outdir_path / f"u_{t}.dat")
            S = load_field(outdir_path / f"s_{t}.dat")
            label_img = u_s_to_labels(U, S, u_thresh, s_thresh)
            curves = compute_ect_curves(label_img, ect=ect, bounds=bounds)
            f = vectorise_sampeuler(curves, ect=ect, bounds=bounds)
        except Exception as e:
            print(f"[abcrej feat idx={theta_idx} t={t}] failed: {e}",
                  file=_sys.stderr)
            continue
        feats.append(f)
        ect_list.append(curves)
        frame_t_indices.append(int(t))

    if not feats:
        return _err_feat(theta_idx, "no frames featurised successfully")

    feats_arr = np.stack(feats).astype(np.float32)      # (n_sampled, n_chi*xpoints)
    ect_arr = np.stack(ect_list).astype(np.float32)     # (n_sampled, n_dirs, xpoints)
    t_arr = np.array(frame_t_indices, dtype=np.int64)
    # T_end is the original simulator's last-frame index (from the meta
    # NPZ), NOT the max of sampled t. The cache's `t` array carries the
    # sampled frames' model times; downstream consumers wanting "did
    # the sim reach time T?" should read `T_end`, not `t.max()`.
    T_end = int(meta_T_end)
    n_sampled = int(len(t_arr))

    # Tmp filename MUST end in '.npz' (see comment in simulate_only).
    try:
        host_tag = socket.gethostname().split(".")[0]
        tmp_path = npz_path.with_suffix(
            f".tmp.{host_tag}.{os.getpid()}.{uuid.uuid4().hex[:6]}.npz"
        )
        np.savez_compressed(
            tmp_path,
            sampeuler=feats_arr,
            ect_curves=ect_arr,
            t=t_arr,
            tauV=np.float64(tauV),
            tauV_int=np.int64(tauV_int),
            xi=np.float64(xi),
            T_end=np.int64(T_end),
            n_frames=np.int64(n_sampled),
            theta_idx=np.int64(theta_idx),
            n_chi=np.int64(bounds["n_chi"]),
            xpoints=np.int64(bounds["xpoints"]),
            x_min=np.float64(bounds["x_min"]),
            x_max=np.float64(bounds["x_max"]),
            y_min=np.float64(bounds["y_min"]),
            y_max=np.float64(bounds["y_max"]),
            n_dirs=np.int64(bounds["n_dirs"]),
            seed=np.int64(bounds["seed"]),
            canvas_side=np.int64(canvas_side),
            delta_x=np.float64(ect.delta_x),
            outdir=np.array(str(outdir)),
        )
        os.replace(tmp_path, npz_path)
    except Exception as e:
        try:
            tmp_path.unlink()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return _err_feat(theta_idx, f"npz save failed: {e}")

    return {
        "theta_idx": int(theta_idx),
        "n_frames": n_sampled,
        "npz_path": str(npz_path),
        "error": "",
    }


def _err_feat(theta_idx, msg) -> dict:
    return {
        "theta_idx": int(theta_idx),
        "n_frames": 0,
        "npz_path": "",
        "error": msg,
    }
