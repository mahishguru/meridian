#!/usr/bin/env python
"""Launch one or more optimizer ablations with distinct output directories.

Builds on `meridian-ablation` (see meridian/cli.py) but adds three things:

  1. Pre-built seed-cache support (skip the 100 DAMASK seed evaluations and
     warm-start every surrogate from the same shared (Z, Y) pool —
     fair-comparison ready).
  2. Explicit per-cell output directory naming
        results/<group>/<decoder>__<opt>__<obj>__r<seed>/
     so DANTE / TuRBO / BAxUS / MERIDIAN never collide.
  3. A `--only OPT[,OPT,...]` filter so a single optimizer (default: dante)
     can be run in isolation while keeping the decoder/objective grid intact.

Example
-------
    python scripts/run_ablation.py \\
        --config configs/acta2026/az31_fmdit.yaml \\
        --seed-cache results/seed_cache_AZ31_v1.npz \\
        --group AZ31_v1 \\
        --only dante \\
        --decoders fmdit \\
        --objectives v1 \\
        --repeats 1

Outputs land in
    results/AZ31_v1/fmdit__dante__v1__r0/
        ├── images/       (decoded PNGs per iteration)
        ├── sims/         (per-iteration DAMASK working dirs)
        ├── log.jsonl
        ├── state_iterNNNN.npz
        └── result.json   (best objective + best z + run config)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config  # noqa: E402
from meridian.loop import OptimizationLoop  # noqa: E402


def _split_csv(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]


def _objective_fingerprint(cfg) -> tuple[dict, str]:
    """Resolved objective parameters plus a short hash of them.

    The v2 Mg-5Gd campaign was invalidated because a grain-penalty setting was
    edited into the YAML part-way through and nothing recorded it, so cells run
    before and after the edit were scored differently and looked comparable.
    Writing the resolved block into every cell makes such a split detectable
    (group by the hash) instead of silent.
    """
    name = str(cfg.objective.name)
    spec = {"name": name, **json.loads(json.dumps(cfg.objective.get(name, {})))}
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return spec, hashlib.sha256(blob.encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="Base YAML config.")
    p.add_argument("--override", "-o", nargs="*", default=[],
                   help="Dotted-key overrides, e.g. loop.n_iterations=10")
    p.add_argument("--seed-cache", default=None,
                   help="Path to seed_cache.npz (overrides loop.seed_cache).")
    p.add_argument("--group", default="run",
                   help="Sub-folder under results/ that groups this ablation set.")

    p.add_argument("--decoders", default=None,
                   help="CSV of decoder names. Default: ablation.decoders.")
    p.add_argument("--optimizers", default=None,
                   help="CSV of optimizer names. Default: ablation.optimizers.")
    p.add_argument("--objectives", default=None,
                   help="CSV of objective names. Default: ablation.objectives.")
    p.add_argument("--only", default=None,
                   help="Restrict optimizers to this CSV (e.g. 'dante').")
    p.add_argument("--repeats", type=int, default=None,
                   help="Repeat count per cell (overrides ablation.n_repeats).")
    p.add_argument("--rep-start", type=int, default=0,
                   help="First repeat index to use (default: 0). Combined with "
                        "--repeats this writes r{rep_start}..r{rep_start+repeats-1}.")

    p.add_argument("--dry-run-list", action="store_true",
                   help="Print the planned cells and exit (no execution).")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Keep going after a cell fails (default: True).")
    p.add_argument("--resume", action="store_true",
                   help="If a run_dir already contains state_iterNNNN.npz "
                        "checkpoints (>= --resume-min-iter), continue from "
                        "the latest one instead of starting over from the "
                        "seed cache. Without this flag the run_dir is wiped "
                        "to avoid cross-run contamination.")
    p.add_argument("--resume-min-iter", type=int, default=5,
                   help="Minimum checkpoint iteration to bother resuming "
                        "from. Cells with only iter<this are wiped and "
                        "started fresh from the seed cache. Default: 5.")
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip any cell whose run_dir already contains a "
                        "result.json (i.e. completed successfully).")
    args = p.parse_args(argv)

    base = apply_overrides(load_config(args.config), args.override)

    decoders = _split_csv(args.decoders) or list(base.ablation.decoders)
    optimizers = _split_csv(args.optimizers) or list(base.ablation.optimizers)
    objectives = _split_csv(args.objectives) or list(base.ablation.objectives)
    if args.only:
        keep = set(_split_csv(args.only) or [])
        optimizers = [o for o in optimizers if o in keep]
        if not optimizers:
            print(f"ERROR: --only={args.only} filtered out all optimizers.", file=sys.stderr)
            return 1
    repeats = int(args.repeats if args.repeats is not None else base.ablation.n_repeats)

    seed_cache = args.seed_cache or getattr(base.loop, "seed_cache", None)
    if seed_cache and not Path(seed_cache).is_file():
        print(f"ERROR: seed cache not found: {seed_cache}", file=sys.stderr)
        return 1

    out_root = Path(base.experiment.output_dir) / args.group
    out_root.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(
        decoders, optimizers, objectives,
        range(int(args.rep_start), int(args.rep_start) + repeats),
    ))
    print(f"Planned {len(cells)} cells under {out_root}/")
    for dec, opt, obj, rep in cells:
        print(f"  - {dec}__{opt}__{obj}__r{rep}")
    if args.dry_run_list:
        return 0

    summary: list[dict] = []
    for dec, opt, obj, rep in cells:
        run_name = f"{dec}__{opt}__{obj}__r{rep}"
        run_dir = out_root / run_name
        # `experiment.name` is joined with `experiment.output_dir` in OptimizationLoop.
        cell_exp_name = f"{args.group}/{run_name}"

        cfg = apply_overrides(load_config(args.config), args.override + [
            f"decoder.name={dec}",
            f"optimizer.name={opt}",
            f"objective.name={obj}",
            f"experiment.seed={int(base.experiment.seed) + rep}",
            f"experiment.name={cell_exp_name}",
        ])

        print("\n" + "=" * 72)
        print(f"[{run_name}] starting → {run_dir}")
        print("=" * 72)

        # --skip-completed: if this cell already has a result.json (i.e. it
        # finished successfully in a previous sweep) leave it alone. This is
        # what makes a re-launched sweep idempotent for already-done cells.
        if args.skip_completed and (run_dir / "result.json").is_file():
            print(f"[skip-completed] {run_name}: result.json present, leaving as-is")
            try:
                rec = json.loads((run_dir / "result.json").read_text())
                summary.append({k: rec.get(k) for k in (
                    "run", "decoder", "optimizer", "objective", "repeat",
                    "output_dir", "status", "best_objective", "elapsed_s")})
            except Exception:
                pass
            continue

        # Decide whether to resume. We only resume if --resume was passed AND
        # there is a state_iterNNNN.npz with iter >= resume_min_iter; otherwise
        # the dir is wiped to avoid cross-run contamination.
        resume_ckpt: Path | None = None
        if args.resume and run_dir.is_dir():
            ckpts = sorted(run_dir.glob("state_iter[0-9][0-9][0-9][0-9].npz"))
            if ckpts:
                latest = ckpts[-1]
                try:
                    latest_iter = int(latest.stem.split("iter")[-1])
                except ValueError:
                    latest_iter = -1
                if latest_iter >= args.resume_min_iter:
                    resume_ckpt = latest
                    print(f"[resume] {run_name}: continuing from {latest.name} "
                          f"(iter {latest_iter}). NOTE: optimizer-internal state "
                          f"(TR radius / MCTS tree / BAxUS embedding) is rebuilt "
                          f"from observations; not bit-identical to an "
                          f"uninterrupted run.")
                else:
                    print(f"[resume] {run_name}: latest checkpoint is iter "
                          f"{latest_iter} < {args.resume_min_iter}; wiping and "
                          f"starting fresh.")

        if resume_ckpt is None and run_dir.exists():
            import shutil
            for child in run_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            print(f"[clean] wiped stale contents of {run_dir}")

        t0 = time.time()
        obj_spec, obj_hash = _objective_fingerprint(cfg)
        cell_record = {
            "run": run_name,
            "decoder": dec, "optimizer": opt, "objective": obj, "repeat": rep,
            "output_dir": str(run_dir),
            "seed_cache": str(seed_cache) if seed_cache else None,
            "resumed_from": str(resume_ckpt) if resume_ckpt else None,
            "objective_spec": obj_spec,
            "objective_hash": obj_hash,
        }
        try:
            res = OptimizationLoop(cfg).run(
                seed_cache=seed_cache,
                resume_checkpoint=resume_ckpt,
            )
            cell_record.update({
                "status": "ok",
                "best_objective": float(res["best_objective"]),
                "elapsed_s": round(time.time() - t0, 1),
            })
            # Persist a tidy result.json next to the loop's checkpoints.
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(json.dumps({
                **cell_record,
                "best_z": res["best_z"],
            }, indent=2))
            print(f"[OK] {run_name}: best={res['best_objective']:.4f}  "
                  f"({cell_record['elapsed_s']}s)")
        except Exception as exc:
            cell_record.update({
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": round(time.time() - t0, 1),
            })
            print(f"[FAIL] {run_name}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                summary.append(cell_record)
                out_root.mkdir(parents=True, exist_ok=True)
                (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
                return 1
        summary.append(cell_record)
        # Refresh after every cell so a Ctrl-C still leaves a useful summary.
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nFinished {len(cells)} cells. Summary → {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
