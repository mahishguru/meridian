"""Console entry points: meridian-run, meridian-ablation."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from meridian.config import apply_overrides, load_config
from meridian.loop import OptimizationLoop


def _common_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--override", "-o", nargs="*", default=[],
                   help="Overrides like decoder.name=mock optimizer.name=dante")
    p.add_argument("--seed-cache", type=str, default=None,
                   help="Path to seed_cache.npz (skip seed simulation).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to state_iterNNNN.npz checkpoint to resume from.")
    return p


def run_experiment(argv: list[str] | None = None) -> int:
    p = _common_parser("meridian-run")
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)

    # CLI flags take priority over config
    seed_cache = args.seed_cache or getattr(cfg.loop, "seed_cache", None)
    resume_ckpt = args.resume or getattr(cfg.loop, "resume_checkpoint", None)

    loop = OptimizationLoop(cfg)
    result = loop.run(seed_cache=seed_cache, resume_checkpoint=resume_ckpt)
    print(json.dumps(result, indent=2))
    return 0


def run_ablation(argv: list[str] | None = None) -> int:
    p = _common_parser("meridian-ablation")
    p.add_argument("--n-repeats", type=int, default=None)
    args = p.parse_args(argv)
    base = load_config(args.config)
    base = apply_overrides(base, args.override)

    decoders = list(base.ablation.decoders)
    optimizers = list(base.ablation.optimizers)
    objectives = list(base.ablation.objectives)
    repeats = int(args.n_repeats if args.n_repeats is not None else base.ablation.n_repeats)

    out_root = Path(base.experiment.output_dir) / "ablation"
    out_root.mkdir(parents=True, exist_ok=True)

    # Seed cache: CLI flag > config
    seed_cache = args.seed_cache or getattr(base.loop, "seed_cache", None)

    summary = []
    for dec, opt, obj, rep in itertools.product(decoders, optimizers, objectives, range(repeats)):
        run_name = f"{dec}__{opt}__{obj}__r{rep}"
        cfg = load_config(args.config)
        apply_overrides(cfg, args.override + [
            f"decoder.name={dec}",
            f"optimizer.name={opt}",
            f"objective.name={obj}",
            f"experiment.seed={int(base.experiment.seed) + rep}",
            f"experiment.name=ablation/{run_name}",
        ])
        try:
            res = OptimizationLoop(cfg).run(seed_cache=seed_cache)
            summary.append({"run": run_name, **res})
            print(f"[OK] {run_name}: best={res['best_objective']:.4f}")
        except Exception as e:
            summary.append({"run": run_name, "error": repr(e)})
            print(f"[FAIL] {run_name}: {e}", file=sys.stderr)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        sys.exit(run_ablation(sys.argv[2:]))
    sys.exit(run_experiment(sys.argv[1:]))
