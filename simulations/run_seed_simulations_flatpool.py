"""Seed simulations for the 2026-07 flat AZ31_extruded PNG pool.

The RVE pool (e.g. rve/AZ31_extruded) is a FLAT directory
of codec-encoded PNGs named AZ31_extruded_orientation_XXXX.png with no
pre-built .dream3d files. This driver:

  Phase A (convert): shuffle the pool with --seed, walk candidates and convert
    PNG -> .dream3d via meridian.simulation.codec.png_to_dream3d
    (colour_tol=None -> exact tol=1 connected-component segmentation, valid
    for clean encoder PNGs; spacing 2um). Accept only RVEs whose grain count
    lies in [--band-lo, --band-hi]; stop when --n accepted.
    Conversions land in <output-dir>/sim_<id>/AZ31_extruded_<id>.dream3d.

  Phase B (simulate): run DAMASK + postproc + Hollomon extraction for each
    accepted RVE using run_one/_run_one_impl from run_random_simulations.py
    (load.yaml + AZ31_Phenopower.yaml in this folder), in parallel.

Outputs (same shape as run_random_simulations.py):
  <output-dir>/manifest.json, properties_all.csv, failures.txt,
  pool_n_grains.json (id -> n_grains for every converted candidate).

Layout is compatible with scripts/encode_runs_to_seed_cache.py:
runs contain sim_<id>/properties.json and PNGs resolve flat as
<pool-dir>/AZ31_extruded_<id>.png.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from run_random_simulations import run_one, write_aggregate_csv  # noqa: E402


def discover_pool(pool_dir: Path, class_key: str) -> list[tuple[str, Path]]:
    png_re = re.compile(
        rf"^{re.escape(class_key)}_(?P<id>orientation_\d+)\.png$"
    )
    items: list[tuple[str, Path]] = []
    for p in sorted(pool_dir.iterdir(), key=lambda q: q.name):
        m = png_re.match(p.name)
        if m and p.is_file():
            items.append((m.group("id"), p))
    return items


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool-dir", type=Path, required=True,
                   help="flat folder of encoded RVE PNGs, e.g. rve/AZ31_extruded")
    p.add_argument("--output-dir", type=Path, default=HERE / "runs")
    p.add_argument("--class-means", type=Path,
                   default=REPO_ROOT / "assets" / "class_means.json")
    p.add_argument("--class-key", default="AZ31_extruded")
    p.add_argument("--load-yaml", type=Path, default=HERE / "load.yaml")
    p.add_argument("--phase-yaml", type=Path, default=HERE / "AZ31_Phenopower.yaml")
    p.add_argument("--target-phase", default="AZ31",
                   help="Phase key inside --phase-yaml (e.g. 'Mg-5gd').")
    p.add_argument("--n", type=int, default=100, help="Accepted RVEs to simulate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--band-lo", type=int, default=250)
    p.add_argument("--band-hi", type=int, default=1100)
    p.add_argument("--spacing", type=float, nargs=3, default=[2.0, 2.0, 2.0])
    p.add_argument("--workers", type=int, default=21)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--loading-direction", default="x", choices=["x", "y", "z"])
    p.add_argument("--youngs-strain-max", type=float, default=1.0e-3)
    p.add_argument("--yield-tolerance", type=float, default=0.4)
    p.add_argument("--keep-hdf5-sample", type=int, default=10)
    p.add_argument("--convert-only", action="store_true",
                   help="Stop after Phase A (PNG -> dream3d conversion).")
    args = p.parse_args()

    if shutil.which("DAMASK_grid") is None and not args.convert_only:
        print("ERROR: DAMASK_grid not on PATH. Export the damask env first, e.g.\n"
              "  conda activate damask   # or: export PATH=<damask-env>/bin:$PATH")
        return 2

    import numpy as np
    from meridian.simulation.codec import png_to_dream3d

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    pool = discover_pool(args.pool_dir, args.class_key)
    if not pool:
        print(f"ERROR: no flat {args.class_key}_orientation_*.png in {args.pool_dir}")
        return 1
    print(f"[pool] {len(pool)} PNGs in {args.pool_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    spacing = np.asarray(args.spacing, dtype=float)
    pool_n_grains_path = out_root / "pool_n_grains.json"
    pool_n_grains: dict[str, int] = {}
    if pool_n_grains_path.is_file():
        pool_n_grains = json.loads(pool_n_grains_path.read_text())

    # ---- Phase A: convert until --n accepted in band ----
    accepted: list[tuple[str, Path, Path]] = []  # (id, png, dream3d)
    scanned = 0
    t0 = time.time()
    for rve_id, png in pool:
        if len(accepted) >= args.n:
            break
        scanned += 1
        sim_dir = out_root / f"sim_{rve_id}"
        d3d = sim_dir / f"{args.class_key}_{rve_id}.dream3d"
        known = pool_n_grains.get(rve_id)
        if known is not None and not (args.band_lo <= known <= args.band_hi):
            continue  # cached out-of-band
        if d3d.is_file() and known is not None:
            accepted.append((rve_id, png, d3d))
            continue
        sim_dir.mkdir(parents=True, exist_ok=True)
        try:
            _, n_grains = png_to_dream3d(
                png_path=png,
                class_means_json=args.class_means,
                class_key=args.class_key,
                output_dream3d=d3d,
                colour_tol=None,          # exact tol=1 CC segmentation
                spacing=spacing,
            )
        except Exception as e:
            print(f"[convert] {rve_id}: FAILED {type(e).__name__}: {e}")
            pool_n_grains[rve_id] = -1
            shutil.rmtree(sim_dir, ignore_errors=True)
            continue
        pool_n_grains[rve_id] = int(n_grains)
        if args.band_lo <= n_grains <= args.band_hi:
            accepted.append((rve_id, png, d3d))
            print(f"[convert] {rve_id}: n_grains={n_grains} ACCEPT "
                  f"({len(accepted)}/{args.n}, scanned {scanned})")
        else:
            print(f"[convert] {rve_id}: n_grains={n_grains} out-of-band, skip")
            shutil.rmtree(sim_dir, ignore_errors=True)
        if scanned % 20 == 0:
            pool_n_grains_path.write_text(json.dumps(pool_n_grains, indent=1))
    pool_n_grains_path.write_text(json.dumps(pool_n_grains, indent=1))

    print(f"[phase A] accepted {len(accepted)}/{args.n} after scanning {scanned} "
          f"({time.time()-t0:.0f}s)")
    if len(accepted) < args.n:
        print("WARNING: pool exhausted before reaching --n")
    if args.convert_only:
        return 0

    # ---- Phase B: DAMASK in parallel ----
    manifest_path = out_root / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())

    keep_ids = {rid for rid, _, _ in accepted[: args.keep_hdf5_sample]}
    jobs = []
    for rve_id, png, d3d in accepted:
        prev = manifest.get(rve_id)
        if prev and prev.get("status") == "ok":
            continue
        sim_dir = out_root / f"sim_{rve_id}"
        # bind_cpus=False: with many concurrent DAMASK processes, OMP core
        # pinning would stack every process onto the same cores.
        jobs.append((rve_id, str(d3d), str(sim_dir), str(args.load_yaml),
                     str(args.phase_yaml), args.threads, args.timeout,
                     args.loading_direction, args.youngs_strain_max,
                     args.yield_tolerance, rve_id in keep_ids,
                     args.workers <= 1, args.target_phase))
        manifest.setdefault(rve_id, {})
        manifest[rve_id].update({
            "png": str(png), "dream3d": str(d3d), "sim_dir": str(sim_dir),
            "n_grains": pool_n_grains.get(rve_id),
        })

    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "2"
    print(f"[phase B] {len(jobs)} DAMASK jobs "
          f"({len(accepted) - len(jobs)} already ok), "
          f"workers={args.workers} threads={args.threads}")

    def _commit(res: dict, idx: int, total: int) -> None:
        rid = res.get("sim_id") or res.get("rve_id")
        manifest[rid].update(res)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{idx}/{total}] {rid}: {res.get('status')} "
              f"({res.get('elapsed_s', '?')}s)")

    t0 = time.time()
    if args.workers <= 1:
        for i, j in enumerate(jobs, 1):
            _commit(run_one(j), i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"sim_id": futures[fut],
                           "status": f"executor_exception: {type(e).__name__}: {e}"}
                _commit(res, i, len(futures))

    n_ok = write_aggregate_csv(
        {"entries": [{"rve_dir": "", "sim_id": rid, **m}
                     for rid, m in manifest.items()]},
        out_root / "properties_all.csv")
    failures = [f"{rid}\t{m.get('status')}" for rid, m in manifest.items()
                if m.get("status") != "ok"]
    (out_root / "failures.txt").write_text("\n".join(failures) + "\n" if failures else "")
    print(f"[done] ok={n_ok} failed={len(failures)} elapsed={time.time()-t0:.0f}s")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
