#!/usr/bin/env python
"""Build a `seed_cache.npz` from completed DAMASK runs.

For every `runs/sim_<id>/` directory under
`automated_simulations/runs/` we:

  1. Locate the matching microstructure PNG in
     `<rve_root>/<id>/AZ31_extruded_<id>.png`
  2. Encode the PNG through the pretrained Compressor (ViT-H/14 → 512-D)
     loaded from a FMDiT checkpoint (`checkpoint["encoder"]`).
  3. Read `properties.json` and score it through the configured objective
     (V1 toughness or V2 target-driven distance).

Result: a `seed_cache.npz` with keys (Z, Y, success) plus a sibling
`seed_cache.properties.json` and `seed_cache.manifest.json`. Drop the
.npz path into `loop.seed_cache` (or pass `--seed-cache` to meridian-run)
and *every* optimizer (DANTE / TuRBO / BAxUS / MERIDIAN) trains its
surrogate on the same shared evaluations — fair-comparison ready.

Usage
-----
    python scripts/encode_runs_to_seed_cache.py \\
        --config configs/acta2026/az31_fmdit_meridian_v1.yaml \\
        --runs-dir   simulations/runs \\
        --rve-root   /path/to/rve/AZ31_extruded \\
        --encoder-ckpt weights/checkpoints/vitfmdit.pth \\
        --output data/seed_caches/seed_cache_AZ31_v1_fmdit.npz \\
        --sample 100 --sample-seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config  # noqa: E402
from meridian.objectives import get_objective  # noqa: E402
from meridian.simulation.extractor import MechanicalProperties, PropertyExtractor  # noqa: E402

PNG_NAME_TMPL = "{prefix}_{id}.png"
PROPS_NAME = "properties.json"
SIM_DIR_RE = re.compile(r"^sim_(?P<id>.+)$")


# ----------------------------------------------------------------------------
# Encoder loader
# ----------------------------------------------------------------------------
def load_encoder(ckpt_path: Path, encoder_repo: Path, device: str,
                 target_dim: int | None = None, spatial_tokens: int = 0):
    """Instantiate the pretrained Compressor and load its weights.

    `target_dim` selects the bottleneck width (canonical 512, or 768/1024 for
    the wider FMDiT variants); ``None`` keeps the Compressor default (512).
    """
    # `encoder_repo` is kept for command-line compatibility; the encoder comes
    # from the installed microstructure-encoder-decoder package.
    from microstructure_ed.encoder_arch_pretrained import Compressor

    if target_dim is None:
        print("[encoder] instantiating Compressor (ViT-H/14, default 512-D bottleneck)")
        encoder = Compressor(use_gradient_checkpointing=False, trainable_blocks=0,
                             spatial_tokens=int(spatial_tokens))
    else:
        print(f"[encoder] instantiating Compressor (ViT-H/14, {target_dim}-D "
              f"bottleneck, spatial_tokens={int(spatial_tokens)})")
        encoder = Compressor(use_gradient_checkpointing=False, trainable_blocks=0,
                             target_dim=int(target_dim),
                             spatial_tokens=int(spatial_tokens))

    print(f"[encoder] loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_state = state.get("encoder", state)  # accept bare state_dict too

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"[encoder] {len(missing)} missing keys (expected for frozen-only buffers): "
              f"{missing[:4]}{' …' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[encoder] {len(unexpected)} unexpected keys "
              f"(harmless if from training-only modules): "
              f"{unexpected[:4]}{' …' if len(unexpected) > 4 else ''}")

    encoder = encoder.to(device).eval()
    return encoder


def png_to_tensor(path: Path, image_size: int = 512) -> torch.Tensor:
    """PNG → (1, 3, H, W) tensor in [-1, 1] (Compressor's expected input range)."""
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        # Use NEAREST to preserve 1-pixel grain boundaries (no smoothing).
        img = img.resize((image_size, image_size), resample=Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # [0, 1]
    arr = arr.transpose(2, 0, 1)                              # (3, H, W)
    t = torch.from_numpy(arr).unsqueeze(0)                    # (1, 3, H, W)
    return t * 2.0 - 1.0                                      # [-1, 1]


# ----------------------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------------------
def discover_runs(runs_dir: Path, rve_root: Path,
                  png_prefix: str = "AZ31_extruded") -> list[tuple[str, Path, Path]]:
    """Return [(sim_id, properties.json, png), ...] for every valid run."""
    items: list[tuple[str, Path, Path]] = []
    for sub in sorted(runs_dir.iterdir(), key=lambda p: p.name):
        m = SIM_DIR_RE.match(sub.name)
        if not (sub.is_dir() and m):
            continue
        sid = m.group("id")
        props = sub / PROPS_NAME
        png = rve_root / sid / PNG_NAME_TMPL.format(prefix=png_prefix, id=sid)
        if not png.is_file():
            # Flat pool layout (2026-07): <rve_root>/<prefix>_<sid>.png
            png = rve_root / PNG_NAME_TMPL.format(prefix=png_prefix, id=sid)
        if not props.is_file():
            print(f"  [skip] {sub.name}: no {PROPS_NAME}")
            continue
        if not png.is_file():
            print(f"  [skip] {sub.name}: PNG not found at {png}")
            continue
        items.append((sid, props, png))
    return items


def _count_material_grains(sim_dir: Path) -> int:
    for p in sorted(sim_dir.glob("*_material.yaml")):
        try:
            data = yaml.safe_load(p.read_text())
            mats = data.get("material") if isinstance(data, dict) else None
            if isinstance(mats, list) and mats:
                return int(len(mats))
        except Exception:
            continue
    return -1


def _count_png_grains(png_path: Path | None) -> int:
    if png_path is None or not png_path.is_file():
        return -1
    try:
        from orientation_codec.dataset import load_png, segment_grains  # type: ignore
        img = load_png(png_path)
        label_map = segment_grains(img, colour_tol=1)
        uniq = np.unique(label_map)
        return int((uniq != 0).sum()) if 0 in uniq else int(uniq.size)
    except Exception:
        return -1


def load_props(path: Path, extractor: PropertyExtractor | None = None,
               png_path: Path | None = None) -> MechanicalProperties | None:
    try:
        sim_dir = path.parent
        csv_path = sim_dir / "damask_stress_strain.csv"
        mp = extractor.extract(csv_path) if extractor is not None and csv_path.is_file() else None
        if mp is None:
            with open(path) as f:
                d = json.load(f)
            mp = MechanicalProperties.from_dict(d)
        if int(getattr(mp, "n_grains", -1) or -1) <= 0:
            ng = _count_material_grains(sim_dir)
            if ng <= 0:
                ng = _count_png_grains(png_path)
            if ng > 0:
                mp.n_grains = ng
        return mp
    except Exception as e:
        print(f"  [warn] could not parse {path}: {e}")
        return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="Path to YAML config (used only for objective).")
    p.add_argument("--override", "-o", nargs="*", default=[],
                   help="Dotted-key overrides, e.g. objective.name=v1")
    p.add_argument("--runs-dir", required=True,
                   help="Directory holding sim_<id>/ subfolders (with properties.json).")
    p.add_argument("--rve-root", required=True,
                   help="Root holding <id>/AZ31_extruded_<id>.png microstructure images.")
    p.add_argument("--png-prefix", default="AZ31_extruded",
                   help="Filename prefix of pool PNGs, e.g. Mg-5Gd_extruded.")
    p.add_argument("--encoder-ckpt", required=True,
                   help="Pretrained FMDiT checkpoint .pth (encoder weights under key 'encoder').")
    p.add_argument("--encoder-repo", default=".",
                   help="Unused; kept for compatibility (the encoder is imported from microstructure_ed).")
    p.add_argument("--output", default=None,
                   help="Output .npz path. Default: <experiment.output_dir>/<experiment.name>/seed_cache.npz")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--spatial-tokens", type=int, default=None,
                   help="Compressor spatial-token layout (v4 encoders use 16). "
                        "Default: cfg latent.spatial_tokens or 0.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only encode first N runs (debug).")
    p.add_argument("--sample", type=int, default=None,
                   help="Deterministically sample N runs after discovery (useful for 100-row seed caches).")
    p.add_argument("--sample-seed", type=int, default=42,
                   help="Random seed for --sample.")
    p.add_argument("--pool-cache", default=None,
                   help="Optional JSON {sim_id: n_grains} to inject n_grains into props "
                        "(properties.json on disk has -1 because the DAMASK extractor "
                        "doesn't compute grain count).")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    objective = get_objective(cfg)
    extractor = PropertyExtractor(
        youngs_strain_max=float(cfg.extraction.youngs_strain_max),
        yield_tolerance=float(cfg.extraction.yield_tolerance),
    )
    print(f"[obj] using {objective.name}")

    pool_n_grains: dict[str, int] = {}
    if args.pool_cache:
        with open(args.pool_cache) as f:
            pool_n_grains = {str(k): int(v) for k, v in json.load(f).items()}
        print(f"[pool] loaded n_grains for {len(pool_n_grains)} sim_ids from {args.pool_cache}")

    runs_dir = Path(args.runs_dir).resolve()
    rve_root = Path(args.rve_root).resolve()
    items = discover_runs(runs_dir, rve_root, png_prefix=args.png_prefix)
    if args.sample:
        rng = np.random.default_rng(int(args.sample_seed))
        if int(args.sample) < len(items):
            idx = np.sort(rng.choice(len(items), size=int(args.sample), replace=False))
            items = [items[int(i)] for i in idx]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"ERROR: no valid runs found under {runs_dir}")
        return 1
    print(f"Found {len(items)} runs with PNG + properties.json")

    # Output paths
    out_path = Path(args.output) if args.output else (
        Path(cfg.experiment.output_dir) / cfg.experiment.name / "seed_cache.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoder
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    spatial_tokens = args.spatial_tokens
    if spatial_tokens is None:
        spatial_tokens = int(getattr(cfg.latent, "spatial_tokens", 0) or 0)
    encoder = load_encoder(Path(args.encoder_ckpt), Path(args.encoder_repo).resolve(),
                           device, target_dim=int(cfg.latent.dim),
                           spatial_tokens=spatial_tokens)

    # Encode + score
    n = len(items)
    dim = int(cfg.latent.dim)
    Z = np.zeros((n, dim), dtype=np.float32)
    Y = np.full(n, np.nan, dtype=np.float32)
    success = np.zeros(n, dtype=bool)
    props_list: list[dict] = [None] * n  # type: ignore[list-item]
    manifest: list[dict] = []

    for i, (sid, props_path, png_path) in enumerate(tqdm(items, desc="encoding+scoring")):
        entry = {"sim_id": sid, "png": str(png_path), "props": str(props_path),
                 "success": False, "objective": float("nan"), "error": None}
        try:
            x = png_to_tensor(png_path, image_size=args.image_size).to(device)
            with torch.no_grad():
                z = encoder(x).squeeze(0).float().cpu().numpy()
            if z.shape[0] != dim:
                raise RuntimeError(f"encoder output dim {z.shape[0]} != latent.dim {dim}")
            Z[i] = z

            mp = load_props(props_path, extractor=extractor, png_path=png_path)
            if mp is None:
                raise RuntimeError("properties.json parse error")
            if pool_n_grains:
                ng = pool_n_grains.get(str(sid))
                if ng is not None:
                    # The RVE pool cache is the authoritative grain count for
                    # these precomputed simulation IDs. Override extractor or
                    # fallback segmentation counts so seed objective rescoring
                    # applies grain penalties to exactly the same morphology.
                    mp.n_grains = int(ng)
            score = objective(mp)
            if not np.isfinite(score):
                raise RuntimeError(f"non-finite objective ({score})")
            Y[i] = float(score)
            success[i] = True
            props_list[i] = mp.to_dict()
            entry["success"] = True
            entry["objective"] = float(score)
            entry["n_grains"] = int(mp.n_grains) if mp.n_grains is not None else None
        except Exception as exc:
            entry["error"] = repr(exc)
            tqdm.write(f"  [fail] {sid}: {exc}")
        manifest.append(entry)

    n_ok = int(success.sum())
    print(f"\nEncoded {n_ok}/{n} runs successfully")
    if n_ok == 0:
        print("ERROR: no successful encodings — not saving cache.")
        return 1

    # Save
    np.savez(out_path, Z=Z, Y=Y, success=success)

    props_path = out_path.with_suffix(".properties.json")
    with open(props_path, "w") as f:
        json.dump(props_list, f, indent=2, default=float)

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print(f"Saved seed cache       → {out_path}")
    print(f"Saved per-seed props   → {props_path}")
    print(f"Saved per-seed manifest→ {manifest_path}")
    print(f"  Z shape:     {Z.shape}  (dtype={Z.dtype})")
    print(f"  Y range:     [{np.nanmin(Y[success]):.4f}, {np.nanmax(Y[success]):.4f}]")
    print(f"  Y mean/std:  {np.nanmean(Y[success]):.4f} / {np.nanstd(Y[success]):.4f}")
    print(f"  Successful:  {n_ok}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
