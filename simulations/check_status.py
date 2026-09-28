#!/usr/bin/env python
"""Quick status check for run_random_simulations.py runs.

Usage:
    python check_status.py                     # default: ./runs
    python check_status.py --run-dir /path/to/runs
    python check_status.py --watch 30          # refresh every 30s
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def get_damask_procs() -> list[dict]:
    """Return running DAMASK_grid processes with PID, CPU%, MEM%, elapsed, cmd."""
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    procs = []
    for line in out.strip().splitlines()[1:]:
        if "DAMASK_grid" not in line or "grep" in line:
            continue
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        procs.append({
            "pid": parts[1],
            "cpu%": parts[2],
            "mem%": parts[3],
            "rss_mb": round(int(parts[5]) / 1024),
            "time": parts[9],
            "cmd": parts[10],
        })
    return procs


def extract_sim_id_from_cmd(cmd: str) -> str | None:
    """Extract sim_id from DAMASK_grid command line."""
    # --geom AZ31_extruded_1182.vti -> 1182
    for part in cmd.split():
        if part.startswith("AZ31_extruded_") and part.endswith(".vti"):
            return part.replace("AZ31_extruded_", "").replace(".vti", "")
    return None


def report(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"No manifest.json in {run_dir}")
        return

    m = json.load(manifest_path.open())
    entries = m["entries"]
    total = len(entries)

    ok = [e for e in entries if e.get("status") == "ok" or
          (e.get("status") == "skipped_existing" and e.get("properties"))]
    skipped = [e for e in entries if e.get("status") == "skipped_existing"
               and not e.get("properties")]
    failed = [e for e in entries if e.get("status") and
              e["status"] not in ("ok", "skipped_existing") and "status" in e]
    pending = [e for e in entries if "status" not in e]

    procs = get_damask_procs()
    running_ids = set()
    for p in procs:
        sid = extract_sim_id_from_cmd(p["cmd"])
        if sid:
            running_ids.add(sid)

    print("=" * 70)
    print(f"  DAMASK Simulation Status  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Run dir: {run_dir}")
    print("=" * 70)
    print(f"  Total:     {total}")
    print(f"  OK:        {len(ok)}")
    print(f"  Failed:    {len(failed)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Pending:   {len(pending)}")
    print(f"  Running:   {len(procs)} DAMASK_grid process(es)")
    print("-" * 70)

    # --- Running processes ---
    if procs:
        print("\n  RUNNING NOW:")
        for p in procs:
            sid = extract_sim_id_from_cmd(p["cmd"]) or "?"
            print(f"    sim_{sid:>5}  PID={p['pid']}  CPU={p['cpu%']}%  "
                  f"RSS={p['rss_mb']}MB  elapsed={p['time']}")

    # --- Recently completed (last 5) ---
    if ok:
        print(f"\n  LAST COMPLETED (showing up to 5 of {len(ok)}):")
        recent = sorted(ok, key=lambda e: e.get("elapsed_s", 0), reverse=False)[-5:]
        for e in recent:
            props = e.get("properties", {})
            sy = props.get("sigma_y", "?")
            su = props.get("sigma_u", "?")
            n = props.get("n", "?")
            sy_str = f"{sy:.1f}" if isinstance(sy, (int, float)) else sy
            su_str = f"{su:.1f}" if isinstance(su, (int, float)) else su
            n_str = f"{n:.3f}" if isinstance(n, (int, float)) else n
            print(f"    sim_{e['sim_id']:>5}  {e.get('elapsed_s','?')}s  "
                  f"σ_y={sy_str} MPa  σ_u={su_str} MPa  n={n_str}")

    # --- Failures ---
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for e in failed:
            reason = e.get("status", "unknown")
            # Truncate long reasons
            if len(reason) > 80:
                reason = reason[:77] + "..."
            print(f"    sim_{e['sim_id']:>5}  {reason}")

    # --- Timing estimate ---
    if ok:
        times = [e["elapsed_s"] for e in ok if "elapsed_s" in e]
        avg_s = sum(times) / len(times) if times else 0
        remaining = len(pending)
        workers = m.get("workers", 3)
        if remaining > 0 and avg_s > 0:
            est_min = (remaining / workers * avg_s) / 60
            print(f"\n  ETA: ~{est_min:.0f} min remaining "
                  f"({remaining} pending, {workers} workers, avg {avg_s:.0f}s/sim)")
    elif pending:
        print(f"\n  ETA: waiting for first completion to estimate...")

    # --- Properties summary ---
    if len(ok) >= 3:
        all_sy = [e["properties"]["sigma_y"] for e in ok if e.get("properties", {}).get("sigma_y")]
        all_su = [e["properties"]["sigma_u"] for e in ok if e.get("properties", {}).get("sigma_u")]
        all_n = [e["properties"]["n"] for e in ok if e.get("properties", {}).get("n")]
        if all_sy:
            print(f"\n  PROPERTY RANGES (from {len(ok)} successful sims):")
            print(f"    σ_y:  {min(all_sy):.1f} – {max(all_sy):.1f} MPa  (mean {sum(all_sy)/len(all_sy):.1f})")
        if all_su:
            print(f"    σ_u:  {min(all_su):.1f} – {max(all_su):.1f} MPa  (mean {sum(all_su)/len(all_su):.1f})")
        if all_n:
            print(f"    n:    {min(all_n):.3f} – {max(all_n):.3f}       (mean {sum(all_n)/len(all_n):.3f})")

    print()


def main():
    p = argparse.ArgumentParser(description="Check DAMASK simulation progress.")
    p.add_argument("--run-dir", type=Path,
                   default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh every N seconds (0 = once).")
    args = p.parse_args()

    if args.watch > 0:
        try:
            while True:
                os.system("clear")
                report(args.run_dir)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        report(args.run_dir)


if __name__ == "__main__":
    main()
