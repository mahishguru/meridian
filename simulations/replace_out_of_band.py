"""Replace out-of-band sims with in-band ones, parallel DAMASK with retry."""
from __future__ import annotations
import argparse, json, random, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_random_simulations import run_one, write_aggregate_csv, _format_progress  # type: ignore


def _make_job(rve_id, args, keep):
    rve_dir = Path(args.rve_root) / rve_id
    d3d = rve_dir / f"AZ31_extruded_{rve_id}.dream3d"
    sim_dir = Path(args.output_dir) / f"sim_{rve_id}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    return (
        rve_id, str(d3d), str(sim_dir),
        str(args.load_yaml), str(args.phase_yaml),
        args.threads, args.timeout, args.loading_direction,
        args.youngs_strain_max, args.yield_tolerance,
        keep,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rve-root", type=Path, required=True, help="folder of RVE samples, e.g. rve/AZ31_extruded")
    p.add_argument("--output-dir", type=Path, default=HERE / "runs")
    p.add_argument("--pool-cache", type=Path, default=HERE / "rve_pool_n_grains.json")
    p.add_argument("--load-yaml", type=Path, default=HERE / "load.yaml")
    p.add_argument("--phase-yaml", type=Path, default=HERE / "AZ31_Phenopower.yaml")
    p.add_argument("--band-lo", type=int, default=50)
    p.add_argument("--band-hi", type=int, default=300)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--loading-direction", default="x", choices=["x","y","z"])
    p.add_argument("--youngs-strain-max", type=float, default=1.0e-3)
    p.add_argument("--yield-tolerance", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-rounds", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pool_n_grains = json.loads(args.pool_cache.read_text())
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    LO, HI = args.band_lo, args.band_hi

    keep_entries, drop_entries = [], []
    for e in manifest["entries"]:
        sid = str(e["sim_id"])
        ng = pool_n_grains.get(sid, -1)
        e["n_grains"] = ng
        if LO <= ng <= HI and e.get("status") in ("ok","skipped_existing"):
            keep_entries.append(e)
        else:
            drop_entries.append(e)

    n_keep = len(keep_entries)
    n_target = len(manifest["entries"])
    n_replace = n_target - n_keep
    print(f"In-band [{LO},{HI}]: keep {n_keep}/{n_target}, need {n_replace} replacements")

    used = {str(e["sim_id"]) for e in manifest["entries"]}
    available = sorted(rid for rid,ng in pool_n_grains.items()
                       if LO <= ng <= HI and rid not in used)
    print(f"Pool (in-band, unused): {len(available)}")
    if len(available) < n_replace:
        print(f"ERROR: pool too small", file=sys.stderr); return 2

    rng = random.Random(args.seed)

    for e in drop_entries:
        d = Path(e["sim_dir"])
        if d.exists():
            print(f"  removing {d.name} (n_grains={e['n_grains']})")
            if not args.dry_run:
                shutil.rmtree(d, ignore_errors=True)

    if args.dry_run:
        print("[dry-run] stop"); return 0

    successful = list(keep_entries)
    keep_hdf5_ids = set(str(x) for x in manifest.get("keep_hdf5_ids", []))
    keep_hdf5_quota = 10
    round_idx = 0
    t_start = time.time()

    while len(successful) < n_target and round_idx < args.max_rounds:
        round_idx += 1
        gap = n_target - len(successful)
        if gap > len(available):
            print(f"ERROR: pool exhausted", file=sys.stderr); break
        picks = rng.sample(available, gap)
        available = [r for r in available if r not in set(picks)]

        round_keep = set()
        slots = max(0, keep_hdf5_quota - len(keep_hdf5_ids))
        if slots:
            k = min(slots, len(picks))
            round_keep = set(rng.sample(picks, k))
            keep_hdf5_ids.update(round_keep)

        jobs = [_make_job(rid, args, rid in round_keep) for rid in picks]
        print(f"\n=== round {round_idx}: {len(jobs)} sims, "
              f"{args.workers}w x {args.threads}t ===", flush=True)

        round_results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, j): j[0] for j in jobs}
            for done, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                round_results[res["sim_id"]] = res
                print(_format_progress(res, done, len(jobs)), flush=True)

        for rid in picks:
            res = round_results.get(rid, {"sim_id": rid, "status": "no_result"})
            ng = pool_n_grains.get(rid, -1)
            entry = {
                "sim_id": rid,
                "rve_dir": str(Path(args.rve_root) / rid),
                "dream3d": str(Path(args.rve_root) / rid / f"AZ31_extruded_{rid}.dream3d"),
                "png": str(Path(args.rve_root) / rid / f"AZ31_extruded_{rid}.png"),
                "sim_dir": str(Path(args.output_dir) / f"sim_{rid}"),
                "n_grains": ng,
            }
            entry.update(res)
            if res.get("status") == "ok":
                successful.append(entry)
            else:
                d = Path(entry["sim_dir"])
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

        manifest["entries"] = successful
        manifest["band"] = {"lo": LO, "hi": HI}
        manifest["keep_hdf5_ids"] = sorted(keep_hdf5_ids)
        manifest["round"] = round_idx
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        n_ok = write_aggregate_csv(manifest, args.output_dir / "properties_all.csv")
        print(f"  round {round_idx}: {len(successful)}/{n_target} ok "
              f"(csv rows={n_ok})", flush=True)

    elapsed = time.time() - t_start
    if len(successful) < n_target:
        print(f"\nWARN: {len(successful)}/{n_target} after {round_idx} rounds, {elapsed:.0f}s",
              file=sys.stderr); return 1
    print(f"\nAll {n_target} in-band done in {elapsed:.0f}s ({round_idx} rounds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
