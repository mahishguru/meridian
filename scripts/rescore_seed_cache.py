#!/usr/bin/env python
"""Rescore an existing seed_cache.npz under the active objective.

This does not rerun DAMASK and does not re-encode images. It reuses the
existing latent matrix Z, reads per-seed properties from the sibling manifest,
optionally recomputes scalar properties from damask_stress_strain.csv so new
curve-work fields are present, reconstructs n_grains from material.yaml when
available, and writes a versioned seed cache with updated Y/success.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config  # noqa: E402
from meridian.objectives import get_objective  # noqa: E402
from meridian.simulation.extractor import MechanicalProperties, PropertyExtractor  # noqa: E402


def _manifest_path(cache_path: Path) -> Path:
    p = cache_path.with_suffix(".manifest.json")
    if p.is_file():
        return p
    p = Path(str(cache_path) + ".manifest.json")
    if p.is_file():
        return p
    raise FileNotFoundError(f"no manifest next to {cache_path}")


def _resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    candidates = [Path(raw), ROOT / raw]  # absolute, or relative to the repository
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


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


def _load_properties(props_path: Path, png_path: Path | None, extractor: PropertyExtractor) -> MechanicalProperties:
    sim_dir = props_path.parent
    csv_path = sim_dir / "damask_stress_strain.csv"
    mp = extractor.extract(csv_path) if csv_path.is_file() else None
    if mp is None:
        mp = MechanicalProperties.from_dict(json.loads(props_path.read_text()))
    ng = int(getattr(mp, "n_grains", -1) or -1)
    if ng <= 0:
        ng = _count_material_grains(sim_dir)
    if ng <= 0:
        ng = _count_png_grains(png_path)
    if ng > 0:
        mp.n_grains = ng
    return mp


def _sim_id_from_entry(entry: dict) -> str | None:
    sid = entry.get("sim_id")
    if sid is not None:
        s = str(sid)
        return s[4:] if s.startswith("sim_") else s
    props = entry.get("props") or entry.get("properties")
    if props:
        name = Path(str(props)).parent.name
        if name.startswith("sim_"):
            return name[4:]
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML config defining objective/extraction.")
    p.add_argument("--seed-cache", required=True, help="Existing seed_cache.npz to rescore.")
    p.add_argument("--output", required=True, help="New output .npz path. Existing file is not modified.")
    p.add_argument("--override", "-o", nargs="*", default=[], help="Dotted-key config overrides.")
    p.add_argument("--pool-cache", default=None,
                   help="Optional JSON {sim_id: n_grains}; authoritative grain counts used before scoring.")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    objective = get_objective(cfg)
    extractor = PropertyExtractor(
        youngs_strain_max=float(cfg.extraction.youngs_strain_max),
        yield_tolerance=float(cfg.extraction.yield_tolerance),
    )
    pool_n_grains: dict[str, int] = {}
    if args.pool_cache:
        pool_path = _resolve_path(args.pool_cache)
        if pool_path is not None and pool_path.is_file():
            pool_n_grains = {str(k): int(v) for k, v in json.loads(pool_path.read_text()).items()}
            print(f"[pool] loaded n_grains for {len(pool_n_grains)} sim_ids from {pool_path}")

    cache_path = Path(args.seed_cache)
    out_path = Path(args.output)
    if out_path.resolve() == cache_path.resolve():
        print("ERROR: --output must be a new file; refusing in-place overwrite", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path, allow_pickle=False)
    Z = data["Z"].astype(np.float32)
    old_success = data["success"].astype(bool)
    manifest = json.loads(_manifest_path(cache_path).read_text())
    if len(manifest) != len(Z):
        raise RuntimeError(f"manifest length {len(manifest)} != Z rows {len(Z)}")

    Y = np.full(len(Z), np.nan, dtype=np.float32)
    success = np.zeros(len(Z), dtype=bool)
    props_list: list[dict | None] = [None] * len(Z)
    new_manifest: list[dict] = []

    for i, entry in enumerate(manifest):
        rec = dict(entry)
        rec["old_success"] = bool(old_success[i])
        rec["success"] = False
        rec["objective"] = float("nan")
        rec["error"] = None
        try:
            props_path = _resolve_path(entry.get("props") or entry.get("properties"))
            if props_path is None or not props_path.is_file():
                raise FileNotFoundError(f"properties path not found: {entry.get('props')}")
            png_path = _resolve_path(entry.get("png"))
            mp = _load_properties(props_path, png_path, extractor)
            sid = _sim_id_from_entry(entry)
            if sid is not None and sid in pool_n_grains:
                # Authoritative RVE grain count for this precomputed sim_id.
                # Use it before objective scoring so v2 cache labels are
                # consistent with v1 and with the original RVE morphology.
                mp.n_grains = int(pool_n_grains[sid])
            score = float(objective(mp))
            if not math.isfinite(score):
                raise RuntimeError(f"non-finite objective: {score}")
            Y[i] = score
            success[i] = True
            props_list[i] = mp.to_dict()
            rec["success"] = True
            rec["objective"] = score
            rec["n_grains"] = int(mp.n_grains)
        except Exception as exc:
            rec["error"] = repr(exc)
        new_manifest.append(rec)

    n_ok = int(success.sum())
    if n_ok == 0:
        print("ERROR: no seed entries rescored successfully", file=sys.stderr)
        return 1

    np.savez(out_path, Z=Z, Y=Y, success=success)
    out_path.with_suffix(".properties.json").write_text(json.dumps(props_list, indent=2, default=float))
    out_path.with_suffix(".manifest.json").write_text(json.dumps(new_manifest, indent=2, default=float))

    print(f"[obj] {objective.name}")
    print(f"rescored {n_ok}/{len(Z)} seeds")
    print(f"wrote {out_path}")
    print(f"Y range [{np.nanmin(Y[success]):.5f}, {np.nanmax(Y[success]):.5f}] mean={np.nanmean(Y[success]):.5f}")
    missing_ng = sum(1 for p0 in props_list if not p0 or int(p0.get("n_grains", -1)) <= 0)
    print(f"missing/unknown n_grains after repair: {missing_ng}/{len(Z)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
