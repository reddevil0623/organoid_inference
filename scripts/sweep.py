#!/usr/bin/env python3
import argparse, os, sys, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import paths as _p
REPO = _p.MODEL_REPO
DATA = _p.DATA


def ensure_inputs():
    os.chdir(REPO)
    if not (REPO / "IN").exists():
        (REPO / "IN").symlink_to("inputs")
    if not (REPO / "INPUT").exists():
        (REPO / "INPUT").symlink_to("inputs")
    if not (REPO / "init_cells_4").exists():
        (REPO / "init_cells_4").symlink_to("inputs/init_cells_4")
    if not (REPO / "run").exists():
        sys.exit("ERROR: ./run not found. Compile first, then retry.")

def xi_tag(xi: float) -> str:
    # 0.16 -> "016"
    return f"{int(round(xi*100)):03d}"

def label_for(tauV: int, xi: float) -> str:
    # Your code uses dt=0.02 by default; keep that in the label for clarity
    return f"dt002_tauV{tauV}_xi{xi_tag(xi)}"

def frange(start: float, stop: float, step: float) -> List[float]:
    vals=[]; x=start
    while x <= stop + 1e-12:
        vals.append(round(x, 10)); x += step
    return vals


def parse_header_tend_dt(logpath: Path) -> Tuple[Optional[int], Optional[float]]:
    tend = None; dt = None
    try:
        with logpath.open("r", errors="ignore") as f:
            for line in f:
                if "tend=" in line and "tmax=" in line and "dt=" in line:
                    parts = line.strip().split()
                    for p in parts:
                        if p.startswith("tend="): tend = int(p.split("=")[1])
                        if p.startswith("dt="):   dt   = float(p.split("=")[1])
                    break
    except FileNotFoundError:
        pass
    return tend, dt

def latest_Tphys(label_dir: Path) -> int:
    """Latest physical time: prefer timestamp.dat, else last 'output t=' in log."""
    ts = label_dir / "timestamp.dat"
    if ts.exists():
        try:
            return int(ts.read_text().strip().split()[0])
        except Exception:
            pass
    log = label_dir / "out"
    try:
        last = 0
        with log.open("r", errors="ignore") as f:
            for line in f:
                if line.startswith("output t="):
                    try: last = int(line.split("=")[1])
                    except: pass
        return last
    except FileNotFoundError:
        return 0


def run_one(tauV: int, xi: float, rs: float, cells: int,
            make_video_cmd: List[str], overwrite: bool=False) -> Dict:
    label = label_for(tauV, xi)
    outdir = DATA / label
    outdir.mkdir(parents=True, exist_ok=True)

    log = outdir / "out"
    video = outdir / f"video_tauV{tauV}_xi{xi_tag(xi)}.mp4"

    if video.exists() and not overwrite:
        return {"label": label, "tauV": tauV, "xi": xi,
                "status": "SKIPPED", "video": str(video)}

    # launch simulation and mark as started (to show ONLY actually running jobs)
    with log.open("w") as lf:
        proc = subprocess.Popen(
            ["./run", label, str(cells), str(tauV), f"{xi:.2f}", f"{rs:.2f}"],
            cwd=REPO, stdout=lf, stderr=subprocess.STDOUT
        )
    # marker that the process really started (monitor filters on this)
    (outdir / ".started").write_text(str(time.time()))

    ret = proc.wait()

    if ret == 0:
        # make per-run video
        vid_cmd = make_video_cmd + ["--label", label, "--outfile", str(video)]
        vret = subprocess.call(vid_cmd, cwd=REPO)
        vstatus = "OK" if vret == 0 else f"VIDEO_FAIL({vret})"
        return {"label": label, "tauV": tauV, "xi": xi,
                "status": "OK", "video": str(video), "video_status": vstatus}
    else:
        return {"label": label, "tauV": tauV, "xi": xi,
                "status": f"FAIL({ret})", "video": None}

# Shows ONLY jobs that actually started (have .started), excludes queued and finished.
# Also prints sweep completion across ALL jobs:
#   done=100%, running=current %, not-yet-started=0%.

def monitor_loop(running: Dict[str, Dict], stop_event: threading.Event,
                 refresh: float=3.0, rows: int=24, sort_key: str="pct_asc",
                 summary_only: bool=False, progress_file: Optional[Path]=None,
                 state: Optional[dict]=None):
    while not stop_event.is_set():
        # snapshot to avoid dict mutation during iteration
        snapshot = list(running.items())

        items = []  # (pct, label, T, tend)
        for label, info in snapshot:
            label_dir = info["log"].parent
            # only display processes that actually started
            if not (label_dir / ".started").exists():
                continue
            tend, _ = parse_header_tend_dt(info["log"])
            T = latest_Tphys(label_dir) if tend else 0
            pct = (100.0 * T / tend) if tend and tend > 0 else 0.0
            items.append((pct, label, T, tend or 0))

        # Optional: write all active-started jobs to TSV
        if progress_file:
            try:
                with progress_file.open("w") as f:
                    f.write("label\tT\tTend\tPercent\n")
                    for pct, label, T, tend in sorted(items, key=lambda x: x[1]):
                        f.write(f"{label}\t{T}\t{tend}\t{pct:.2f}\n")
            except Exception:
                pass

        # Sweep completion
        active = len(items)
        if state:
            total = max(1, state.get("total", 0))
            done  = state.get("done", 0)
            overall = (100.0*done + sum(p for p, *_ in items)) / total
        else:
            overall = sum(p for p, *_ in items)/active if active else 0.0

        # sort & trim
        if sort_key == "label":
            items.sort(key=lambda x: x[1])
        elif sort_key == "pct_desc":
            items.sort(key=lambda x: (-x[0], x[1]))
        else:  # pct_asc
            items.sort(key=lambda x: (x[0], x[1]))
        shown = items if rows <= 0 else items[:rows]

        try: os.system("clear")
        except Exception: pass
        print("== Parameter sweep progress ==")
        if state:
            total = state.get("total", 0); done = state.get("done", 0)
            print(f"Running (started): {active:>3}   Finished: {done:>3}/{total}   "
                  f"Sweep completion: {overall:5.2f}%")
        else:
            print(f"Active (started) jobs: {active:>3}   Overall avg(active): {overall:5.2f}%")

        if not summary_only:
            print()
            for pct, label, T, tend in shown:
                print(f"{label:28s}  t={T:>8}/{tend:<8}  {pct:6.2f}%")
            if rows > 0 and active > rows:
                print(f"\n… {active-rows} more running (use --monitor-rows or --progress-file).")

        time.sleep(refresh)


def main():
    p = argparse.ArgumentParser(
        description="Run τV × ξ sweep (parallel), show ONLY running jobs, and build per-run videos.")
    p.add_argument("--tauV", nargs="+", type=int,
                   default=[1,10,20,30,40,50,60,70,80,90])
    p.add_argument("--xi-list", nargs="*", type=float, default=None)
    p.add_argument("--xi-range", type=str, default=None,
                   help="start:stop:step, e.g. 0.10:0.32:0.02")
    p.add_argument("-j", "--jobs", type=int,
                   default=max(1, (os.cpu_count() or 2)//2),
                   help="Max concurrent simulations")
    p.add_argument("--cells", type=int, default=4)
    p.add_argument("--rs", type=float, default=0.70)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--video", choices=["solid","overlay"], default="solid")
    p.add_argument("--video-fps", type=int, default=20)

    # solid-cells appearance
    p.add_argument("--soft-edge", type=float, default=0.0)
    p.add_argument("--cells-color", default="#2ecc71")
    p.add_argument("--lumen-alpha", type=float, default=0.30)
    p.add_argument("--lumen-thresh", type=float, default=0.5)

    # monitor controls
    p.add_argument("--monitor-rows", type=int, default=24,
                   help="max running rows to display (<=0 = all)")
    p.add_argument("--monitor-sort", choices=["pct_asc","pct_desc","label"],
                   default="pct_asc", help="row sort order")
    p.add_argument("--monitor-summary-only", action="store_true",
                   help="only print the header summary")
    p.add_argument("--monitor-refresh", type=float, default=3.0,
                   help="seconds between monitor refreshes")
    p.add_argument("--progress-file", type=str, default=None,
                   help="optional TSV written each refresh with running jobs")

    args = p.parse_args()
    ensure_inputs()

    # Build xi list
    if args.xi_list:
        xis = args.xi_list
    elif args.xi_range:
        a,b,c = args.xi_range.split(":")
        xis = frange(float(a), float(b), float(c))
    else:
        xis = frange(0.10, 0.32, 0.02)

    # Video maker
    if args.video == "solid":
        vid_py = _p.SCRIPTS / "make_solid_cells_video.py"
        if not vid_py.exists():
            sys.exit("Missing scripts/make_solid_cells_video.py.")
        make_video_cmd = [
            sys.executable, str(vid_py),
            "--fps", str(args.video_fps),
            "--soft-edge", str(args.soft_edge),
            "--cells-color", str(args.cells_color),
            "--lumen-alpha", str(args.lumen_alpha),
            "--lumen-thresh", str(args.lumen_thresh),
        ]
    else:
        vid_py = _p.SCRIPTS / "make_combo_video.py"
        if not vid_py.exists():
            sys.exit("Missing scripts/make_combo_video.py.")
        make_video_cmd = [
            sys.executable, str(vid_py),
            "--mode", "overlay",
            "--fps", str(args.video_fps),
        ]

    jobs = [(tv, x) for tv in args.tauV for x in xis]
    total_jobs = len(jobs)
    print(f"Planned runs: {total_jobs}  (tauV={args.tauV}, xi={[f'{v:.2f}' for v in xis]})")
    DATA.mkdir(parents=True, exist_ok=True)

    # monitor thread state (for true sweep completion)
    running: Dict[str, Dict] = {}
    state = {"total": total_jobs, "done": 0}
    stop_event = threading.Event()
    progress_file = Path(args.progress_file) if args.progress_file else None
    mon = threading.Thread(
        target=monitor_loop,
        args=(running, stop_event, args.monitor_refresh, args.monitor_rows,
              args.monitor_sort, args.monitor_summary_only, progress_file, state),
        daemon=True,
    )
    mon.start()

    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            future_map = {}
            for tauV, xi in jobs:
                label = label_for(tauV, xi)
                outdir = DATA / label
                outdir.mkdir(parents=True, exist_ok=True)
                log = outdir / "out"
                running[label] = {"log": log, "tauV": tauV, "xi": xi}
                fut = ex.submit(run_one, tauV, xi, args.rs, args.cells, make_video_cmd, args.overwrite)
                future_map[fut] = label

            for fut in as_completed(future_map):
                res = fut.result()
                results.append(res)
                running.pop(res["label"], None)   # finished -> remove from “running”
                state["done"] += 1                # finished -> counts as 100% in completion
    finally:
        stop_event.set()

    # summary
    ok = sum(1 for r in results if r["status"].startswith("OK"))
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    failed = [r for r in results if (not r["status"].startswith("OK")) and r["status"]!="SKIPPED"]

    print("\n== Sweep summary ==")
    print(f"OK: {ok}   SKIPPED: {skipped}   FAIL: {len(failed)}")
    for r in failed:
        print(f"  {r['label']}: {r['status']}")
    example = next((r for r in results if r.get("video")), None)
    if example:
        print("\nExample copy from a remote host:")
        print(f"scp <host>:{example['video']} .")

if __name__ == "__main__":
    main()
