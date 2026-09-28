"""Run DAMASK_grid on a random sample of AZ31_extruded RVEs.

For each sampled RVE directory <rve_root>/<id>/ we:
  1. Run DAMASK on AZ31_extruded_<id>.dream3d using
     AZ31_Phenopower.yaml + load.yaml in this folder.
  2. Post-process the DAMASK HDF5 -> engineering stress-strain CSV.
  3. Extract Hollomon mechanical properties (E, sigma_y, sigma_u, n, K, ...).

Per-simulation outputs land in <output_dir>/sim_<id>/:
  - AZ31_extruded_<id>_material.yaml   # DAMASK material file (orientations)
  - AZ31_extruded_<id>.vti             # DAMASK geometry (compressed)
  - load.yaml                          # boundary conditions (copied)
  - damask.log                         # full DAMASK stdout/stderr
  - AZ31_extruded_<id>_load_AZ31_extruded_<id>_material.hdf5   # raw result
  - damask_stress_strain.csv           # step,strain_eng,stress_eng (MPa)
  - properties.json                    # extracted Hollomon properties

Top-level outputs in <output_dir>/:
  - manifest.json    # sim_id -> rve_dir, png, dream3d, sim_dir, csv, props, status
  - properties_all.csv   # one row per successful simulation, columns =
                         # all MechanicalProperties fields + sim_id, png path
  - failures.txt     # one line per failed sim with reason

The png path is preserved per-sim so a future trained image encoder can
encode each microstructure into a latent z; combined with the extracted
properties this yields a (Z, properties) seed_cache.npz that the four
optimizers (MERIDIAN/DANTE/TuRBO/BAxUS) consume to warm-start their
surrogate models -- see scripts/precompute_seeds.py in
the repository root for the cache format.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Make meridian.simulation imports available without installing the package
sys.path.insert(0, str(REPO_ROOT))


def discover_rves(rve_root: Path) -> list[tuple[str, Path, Path, Path]]:
    """Return list of (rve_id, rve_dir, dream3d_path, png_path) for valid RVEs."""
    items: list[tuple[str, Path, Path, Path]] = []
    for sub in sorted(rve_root.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        rve_id = sub.name
        d3d = sub / f"AZ31_extruded_{rve_id}.dream3d"
        png = sub / f"AZ31_extruded_{rve_id}.png"
        if d3d.is_file() and png.is_file():
            items.append((rve_id, sub, d3d, png))
    return items


def _run_one_impl(
    rve_id: str,
    dream3d_str: str,
    sim_dir_str: str,
    load_yaml: str,
    phase_yaml: str,
    n_threads: int,
    timeout: int,
    loading_direction: str,
    youngs_strain_max: float,
    yield_tolerance: float,
    keep_hdf5: bool,
    bind_cpus: bool = True,
    target_phase: str = "AZ31",
) -> dict:
    """Run DAMASK + postproc + property extraction for one RVE."""
    from meridian.simulation.damask import DAMASKRunner
    from meridian.simulation.extractor import PropertyExtractor
    from meridian.simulation.postproc import StressStrainProcessor

    sim_dir = Path(sim_dir_str)
    t0 = time.time()
    out: dict = {"sim_id": rve_id, "sim_dir": sim_dir_str}

    runner = DAMASKRunner(
        load_yaml=load_yaml,
        phase_yaml=phase_yaml,
        target_phase=target_phase,
        target_homog="SX",
        n_threads=n_threads,
        timeout_seconds=timeout,
        bind_cpus=bind_cpus,
    )

    try:
        result = runner.run(dream3d=dream3d_str, sim_dir=sim_dir_str)
    except Exception as e:
        out["status"] = f"damask_launch_exception: {type(e).__name__}: {e}"
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    out["returncode"] = int(result.returncode)
    out["log"] = str(result.log_path)
    out["hdf5"] = str(result.hdf5_path) if result.hdf5_path else None

    if result.returncode != 0 or result.hdf5_path is None:
        out["status"] = f"damask_failed_rc={result.returncode}"
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    csv_path = sim_dir / "damask_stress_strain.csv"
    try:
        StressStrainProcessor(loading_direction=loading_direction).process_to_csv(
            result.hdf5_path, csv_path
        )
        out["csv"] = str(csv_path)
    except Exception as e:
        out["status"] = f"postproc_failed: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=4)
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    try:
        props = PropertyExtractor(
            youngs_strain_max=youngs_strain_max,
            yield_tolerance=yield_tolerance,
        ).extract(csv_path)
    except Exception as e:
        out["status"] = f"extractor_exception: {type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=4)
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    if props is None:
        out["status"] = "extractor_returned_none"
        out["elapsed_s"] = round(time.time() - t0, 2)
        return out

    props_dict = props.to_dict()
    props_path = sim_dir / "properties.json"
    with props_path.open("w") as f:
        json.dump(props_dict, f, indent=2)
    out["properties"] = props_dict
    out["properties_json"] = str(props_path)

    if not keep_hdf5:
        try:
            Path(result.hdf5_path).unlink()
            out["hdf5"] = None
        except OSError:
            pass

    out["status"] = "ok"
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def run_one(args: tuple) -> dict:
    """Picklable wrapper for ProcessPoolExecutor."""
    return _run_one_impl(*args)


def write_aggregate_csv(manifest: dict, out_path: Path) -> int:
    """Write one row per successful sim with all properties + provenance."""
    import csv as _csv

    rows = []
    for e in manifest["entries"]:
        if e.get("status") in ("ok", "skipped_existing") and e.get("properties"):
            row = {
                "sim_id": e["sim_id"],
                "rve_dir": e["rve_dir"],
                "png": e["png"],
                "dream3d": e["dream3d"],
                "sim_dir": e["sim_dir"],
                "csv": e.get("csv"),
                "elapsed_s": e.get("elapsed_s"),
                **e["properties"],
            }
            rows.append(row)

    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _format_progress(res: dict, idx: int, total: int) -> str:
    base = f"[{idx}/{total}] sim_{res['sim_id']:>5} -> {res['status']} ({res.get('elapsed_s', '?')}s)"
    props = res.get("properties") or {}
    if props and all(isinstance(props.get(k), (int, float)) for k in ("sigma_y", "sigma_u", "n")):
        base += (f"  sigma_y={props['sigma_y']:.1f} MPa  "
                 f"sigma_u={props['sigma_u']:.1f} MPa  n={props['n']:.3f}")
    return base


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rve-root", type=Path, required=True,
                   help="folder of RVE samples, e.g. rve/AZ31_extruded")
    p.add_argument("--output-dir", type=Path, default=HERE / "runs")
    p.add_argument("--load-yaml", type=Path, default=HERE / "load.yaml")
    p.add_argument("--phase-yaml", type=Path, default=HERE / "AZ31_Phenopower.yaml")
    p.add_argument("--n", type=int, default=100, help="Number of RVEs to sample.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=3,
                   help="Parallel DAMASK processes.")
    p.add_argument("--threads", type=int, default=12,
                   help="OMP_NUM_THREADS per simulation.")
    p.add_argument("--timeout", type=int, default=7200, help="Per-sim timeout (s).")
    p.add_argument("--loading-direction", default="x", choices=["x", "y", "z"])
    p.add_argument("--youngs-strain-max", type=float, default=1.0e-3)
    p.add_argument("--yield-tolerance", type=float, default=0.4)
    p.add_argument("--keep-hdf5", action="store_true",
                   help="Retain ALL per-sim HDF5 files (default: delete after CSV+props extracted).")
    p.add_argument("--keep-hdf5-sample", type=int, default=10,
                   help="Randomly retain HDF5 for this many sims (ignored if --keep-hdf5).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip sims whose properties.json already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Sample + write manifest only; do not run DAMASK.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = discover_rves(args.rve_root)
    if not pool:
        print(f"ERROR: no valid RVEs (dream3d+png) under {args.rve_root}", file=sys.stderr)
        return 2
    if args.n > len(pool):
        print(f"WARN: requested {args.n} but only {len(pool)} valid; using all.",
              file=sys.stderr)
        args.n = len(pool)

    rng = random.Random(args.seed)
    sample = rng.sample(pool, args.n)

    # Pre-select which sim_ids will retain their HDF5.
    sample_ids = [rid for rid, _, _, _ in sample]
    if args.keep_hdf5:
        keep_hdf5_ids = set(sample_ids)
    elif args.keep_hdf5_sample > 0:
        k = min(args.keep_hdf5_sample, len(sample_ids))
        keep_hdf5_ids = set(rng.sample(sample_ids, k))
    else:
        keep_hdf5_ids = set()

    manifest_path = args.output_dir / "manifest.json"
    manifest: dict = {
        "rve_root": str(args.rve_root),
        "load_yaml": str(args.load_yaml),
        "phase_yaml": str(args.phase_yaml),
        "n_requested": args.n,
        "seed": args.seed,
        "workers": args.workers,
        "threads_per_sim": args.threads,
        "loading_direction": args.loading_direction,
        "extraction": {
            "youngs_strain_max": args.youngs_strain_max,
            "yield_tolerance": args.yield_tolerance,
        },
        "keep_hdf5_ids": sorted(keep_hdf5_ids),
        "entries": [],
    }
    sim_jobs: list[tuple] = []
    for rve_id, rve_dir, d3d, png in sample:
        sim_dir = args.output_dir / f"sim_{rve_id}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "sim_id": rve_id,
            "rve_dir": str(rve_dir),
            "dream3d": str(d3d),
            "png": str(png),
            "sim_dir": str(sim_dir),
        }
        manifest["entries"].append(entry)
        if args.skip_existing and (sim_dir / "properties.json").is_file():
            entry["status"] = "skipped_existing"
            try:
                entry["properties"] = json.loads((sim_dir / "properties.json").read_text())
            except Exception:
                pass
            continue
        sim_jobs.append((
            rve_id, str(d3d), str(sim_dir),
            str(args.load_yaml), str(args.phase_yaml),
            args.threads, args.timeout, args.loading_direction,
            args.youngs_strain_max, args.yield_tolerance,
            rve_id in keep_hdf5_ids,
        ))

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path} "
          f"({len(manifest['entries'])} entries, {len(sim_jobs)} to run)")

    if args.dry_run:
        return 0

    by_id = {e["sim_id"]: e for e in manifest["entries"]}
    t_start = time.time()

    def _commit(res: dict, idx: int, total: int) -> None:
        by_id[res["sim_id"]].update(res)
        print(_format_progress(res, idx, total), flush=True)
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

    if args.workers <= 1:
        for i, job in enumerate(sim_jobs, 1):
            _commit(run_one(job), i, len(sim_jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, j): j[0] for j in sim_jobs}
            for done, fut in enumerate(as_completed(futures), 1):
                _commit(fut.result(), done, len(sim_jobs))

    n_ok = write_aggregate_csv(manifest, args.output_dir / "properties_all.csv")
    failures = [e for e in manifest["entries"]
                if e.get("status") not in ("ok", "skipped_existing")]
    if failures:
        with (args.output_dir / "failures.txt").open("w") as f:
            for e in failures:
                f.write(f"{e['sim_id']}\t{e.get('status', 'unknown')}\n")

    elapsed = round(time.time() - t_start, 1)
    print(f"\nDone in {elapsed}s. Successful: {n_ok}/{len(manifest['entries'])}")
    print(f"  manifest:        {manifest_path}")
    print(f"  properties_all:  {args.output_dir / 'properties_all.csv'}")
    if failures:
        print(f"  failures:        {args.output_dir / 'failures.txt'}  ({len(failures)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
