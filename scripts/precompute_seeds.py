#!/usr/bin/env python
"""Pre-compute seed evaluations and save to .npz for reuse across optimizers.

Runs N seed latent vectors through the full pipeline:
  z → Decoder → PNG → Orientation codec → Dream3D → DAMASK → Properties → Objective

Saves:
  seed_cache.npz  with keys: Z, Y, properties, success
  where Z is (N, dim), Y is (N,), properties is (N,) object array of dicts,
  and success is (N,) bool array.

Usage:
    python scripts/precompute_seeds.py --config configs/base.yaml [--n-seeds 100]

All optimizer ablations can then load this cache instead of re-running DAMASK.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pre-compute seed simulations for optimizer warm-start.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--override", "-o", nargs="*", default=[])
    p.add_argument("--n-seeds", type=int, default=None,
                   help="Number of seeds (overrides latent.init.n_seed).")
    p.add_argument("--output", type=str, default=None,
                   help="Output .npz path. Default: {output_dir}/seed_cache.npz")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)

    if args.n_seeds is not None:
        cfg.latent.init.n_seed = args.n_seeds

    n_seed = int(cfg.latent.init.n_seed)
    dim = int(cfg.latent.dim)

    # Determine output path
    out_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "seed_cache.npz"
    images_dir = out_dir / "seed_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sims_dir = out_dir / "seed_sims"
    sims_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:    {args.config}")
    print(f"Seeds:     {n_seed}")
    print(f"Output:    {out_path}")
    print(f"Work dir:  {out_dir}")
    print(f"Dry run:   {bool(cfg.loop.dry_run)}")

    # --- Build components ---
    import torch
    np.random.seed(int(cfg.experiment.seed))
    torch.manual_seed(int(cfg.experiment.seed))

    from meridian.decoders import get_decoder
    from meridian.latent import BoxConstraint, build_initial_population
    from meridian.objectives import get_objective

    decoder = get_decoder(cfg)
    objective = get_objective(cfg)
    box = BoxConstraint(*cfg.latent.bounds) if cfg.latent.use_box_constraint else None

    sim_pipeline = None
    if not bool(cfg.loop.dry_run):
        from meridian.simulation import SimulationPipeline
        from meridian.simulation.damask import DAMASKRunner
        from meridian.simulation.extractor import PropertyExtractor
        from meridian.simulation.postproc import StressStrainProcessor

        s = cfg.simulation
        damask = DAMASKRunner(
            load_yaml=s.load_yaml, phase_yaml=s.phase_yaml,
            target_phase=s.target_phase, target_homog=s.target_homog,
            damask_binary=s.damask_binary, n_threads=int(s.n_threads),
            timeout_seconds=int(s.timeout_seconds),
        )
        post = StressStrainProcessor(loading_direction=s.loading_direction)
        ext = PropertyExtractor(
            youngs_strain_max=float(cfg.extraction.youngs_strain_max),
            yield_tolerance=float(cfg.extraction.yield_tolerance),
        )
        template = getattr(cfg.codec, "template_dream3d", None)
        colour_tol_raw = getattr(cfg.codec, "colour_tol", None)
        if isinstance(colour_tol_raw, str):
            colour_tol = colour_tol_raw
        elif colour_tol_raw is not None:
            colour_tol = int(colour_tol_raw)
        else:
            colour_tol = None
        spacing_raw = getattr(cfg.codec, "spacing", None)
        spacing = np.array(spacing_raw, dtype=np.float32) if spacing_raw else None
        min_grains = getattr(cfg.codec, "min_grains", None)
        target_size = getattr(cfg.codec, "target_size", None)
        min_pixel_std = getattr(cfg.codec, "min_pixel_std", None)

        sim_pipeline = SimulationPipeline(
            damask=damask, postproc=post, extractor=ext,
            class_means_json=cfg.codec.class_means_json,
            class_key=cfg.codec.class_key,
            template_dream3d=template,
            colour_tol=colour_tol,
            spacing=spacing,
            cleanup_hdf5=bool(s.cleanup_hdf5),
            min_grains=int(min_grains) if min_grains is not None else None,
            target_size=int(target_size) if target_size is not None else None,
            min_pixel_std=float(min_pixel_std) if min_pixel_std is not None else None,
        )

    # --- Generate initial population ---
    Z = build_initial_population(cfg)
    if box is not None:
        Z = box.clamp_np(Z)
    print(f"Generated {len(Z)} seed latent vectors (dim={Z.shape[1]})")

    # --- Decode all ---
    z_t = torch.from_numpy(Z).float()
    images = decoder.decode(z_t)
    png_paths = []
    for j, img in enumerate(images):
        p = images_dir / f"seed_{j:04d}.png"
        img.save(p)
        png_paths.append(p)
    print(f"Decoded {len(png_paths)} images")

    # --- Evaluate ---
    Y = np.full(n_seed, np.nan, dtype=np.float32)
    success = np.zeros(n_seed, dtype=bool)
    properties = np.empty(n_seed, dtype=object)
    errors = np.empty(n_seed, dtype=object)
    manifest: list[dict] = []

    for i, (z, png) in enumerate(tqdm(list(zip(Z, png_paths)), desc="DAMASK seeds")):
        t0 = time.time()
        sim_dir = sims_dir / f"seed_{i:04d}"
        props_path = sim_dir / "properties.json"
        entry = {
            "sim_id": f"seed_{i:04d}",
            "png": str(png),
            "props": str(props_path),
            "success": False,
            "objective": float("nan"),
            "error": None,
        }
        try:
            if bool(cfg.loop.dry_run):
                from meridian.simulation.mock import mock_simulate
                outcome = mock_simulate(z)
            else:
                outcome = sim_pipeline.evaluate_png(png, work_dir=sim_dir)

            if outcome.success and outcome.properties is not None:
                score = objective(outcome.properties)
                Y[i] = score
                success[i] = True
                properties[i] = outcome.properties.to_dict()
                sim_dir.mkdir(parents=True, exist_ok=True)
                props_path.write_text(json.dumps(properties[i], indent=2, default=float))
                errors[i] = None
                entry["success"] = True
                entry["objective"] = float(score)
                entry["n_grains"] = int(getattr(outcome.properties, "n_grains", -1))
            else:
                errors[i] = outcome.error or "unknown"
                entry["error"] = errors[i]
        except Exception as exc:
            errors[i] = repr(exc)
            entry["error"] = errors[i]
        manifest.append(entry)

        elapsed = time.time() - t0
        status = "OK" if success[i] else "FAIL"
        tqdm.write(f"  [{i+1:3d}/{n_seed}] {status}  y={Y[i]:.3f}  ({elapsed:.1f}s)  {errors[i] or ''}")

    n_ok = int(success.sum())
    print(f"\nCompleted: {n_ok}/{n_seed} successful ({n_ok/n_seed*100:.0f}%)")

    if n_ok == 0:
        print("ERROR: All seeds failed. Not saving cache.")
        return 1

    # --- Save ---
    # Convert properties to a JSON-serializable list for storage
    props_list = [properties[i] if properties[i] is not None else {} for i in range(n_seed)]

    np.savez(
        out_path,
        Z=Z,
        Y=Y,
        success=success,
    )

    # Save properties separately as JSON (np.savez doesn't handle dicts well)
    props_json_path = out_path.with_suffix(".properties.json")
    with open(props_json_path, "w") as f:
        json.dump(props_list, f, indent=2, default=float)

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print(f"Saved seed cache → {out_path}")
    print(f"Saved properties  → {props_json_path}")
    print(f"Saved manifest    → {manifest_path}")
    print(f"  Z:       ({Z.shape[0]}, {Z.shape[1]})")
    print(f"  Y range: [{np.nanmin(Y[success]):.3f}, {np.nanmax(Y[success]):.3f}]")
    print(f"  Valid:   {n_ok}/{n_seed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
