#!/usr/bin/env python
"""Prepare the Mg-5Gd v2 campaign after the seed DAMASK sims complete.

Two subcommands (both idempotent, both print what they change):

  targets  — read runs_mg5gd/sim_*/properties.json, print the property
             distribution, pick v2 targets (per-property 80th percentile,
             rounded: 5 MPa for stresses/K, 0.01 for n) and write them into
             the config's `objective.v2.targets` block.

  shell    — read the encoded seed cache (Z), print ||z|| statistics and
             rewrite `shell_radius_min/max` (median ∓ band) plus
             `seed_rows` in the MERIDIAN block of the config.

Usage:
    python scripts/mg5gd_prepare_v2.py targets \
        --runs-dir simulations/runs_mg5gd \
        --config configs/acta2026/mg5gd_fmdit_1280_meridian_v2.yaml
    python scripts/mg5gd_prepare_v2.py shell \
        --cache data/seed_caches/seed_cache_Mg5Gd_v2_fmdit_1280.npz \
        --config configs/acta2026/mg5gd_fmdit_1280_meridian_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROPS = ("sigma_y", "n", "K", "sigma_u")


def _load_run_properties(runs_dir: Path) -> list[dict]:
    rows = []
    for pj in sorted(runs_dir.glob("sim_*/properties.json")):
        try:
            d = json.loads(pj.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] unreadable {pj}: {e}")
            continue
        if all(isinstance(d.get(k), (int, float)) for k in PROPS):
            rows.append(d)
    return rows


def _round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 10)


def cmd_targets(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir).resolve()
    rows = _load_run_properties(runs_dir)
    if len(rows) < 20:
        print(f"ERROR: only {len(rows)} usable properties.json under {runs_dir}")
        return 1

    print(f"[targets] {len(rows)} successful sims in {runs_dir}")
    header = f"{'prop':>10} {'min':>9} {'p20':>9} {'p50':>9} {'p80':>9} {'max':>9}"
    print(header)
    targets: dict[str, float] = {}
    for k in PROPS:
        v = np.array([r[k] for r in rows], dtype=float)
        q = np.percentile(v, [0, 20, 50, 80, 100])
        print(f"{k:>10} " + " ".join(f"{x:9.3f}" for x in q))
        t = float(np.percentile(v, args.percentile))
        targets[k] = _round_to(t, 0.01 if k == "n" else 5.0)

    print(f"[targets] chosen (p{args.percentile}, rounded): {targets}")

    cfg_path = Path(args.config).resolve()
    lines = cfg_path.read_text().splitlines(keepends=True)
    in_v2 = in_targets = False
    n_repl = 0
    for i, ln in enumerate(lines):
        if re.match(r"^  v2:", ln):
            in_v2 = True
        elif in_v2 and re.match(r"^  \S", ln):
            in_v2 = False
        if in_v2 and re.match(r"^    targets:", ln):
            in_targets = True
            continue
        if in_targets and re.match(r"^    \S", ln):
            in_targets = False
        if in_targets:
            m = re.match(r"^(\s+)(sigma_y|n|K|sigma_u):\s*\S+(.*)$", ln)
            if m and m.group(2) in targets:
                key = m.group(2)
                unit = "" if key == "n" else "    # MPa"
                lines[i] = (f"{m.group(1)}{key}: {targets[key]}{unit}"
                            f"  # p{args.percentile} of {len(rows)} Mg-5Gd seed sims\n")
                n_repl += 1
    if n_repl != len(PROPS):
        print(f"ERROR: replaced {n_repl}/{len(PROPS)} target lines in {cfg_path}")
        return 1
    txt = "".join(lines).replace(
        "    # PLACEHOLDER until seed sims complete; updated automatically below.\n", "")
    cfg_path.write_text(txt)
    print(f"[targets] wrote objective.v2.targets into {cfg_path}")
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    cache = Path(args.cache).resolve()
    data = np.load(cache, allow_pickle=True)
    Z = np.asarray(data["Z"], dtype=np.float64)
    success = np.asarray(data["success"]).astype(bool) if "success" in data else np.ones(len(Z), bool)
    r = np.linalg.norm(Z[success], axis=1)
    med = float(np.median(r))
    print(f"[shell] cache={cache.name} rows={len(Z)} ok={int(success.sum())} "
          f"||z||: min={r.min():.2f} med={med:.2f} max={r.max():.2f} std={r.std():.3f}")

    lo = round(med - args.band, 2)
    hi = round(med + args.band, 2)
    cfg_path = Path(args.config).resolve()
    txt = cfg_path.read_text()
    txt, n1 = re.subn(r"(?m)^(\s*shell_radius_min:)\s*\S+.*$",
                      rf"\g<1> {lo}  # Mg-5Gd seed-cache median ||z||={med:.2f} - {args.band}", txt)
    txt, n2 = re.subn(r"(?m)^(\s*shell_radius_max:)\s*\S+.*$",
                      rf"\g<1> {hi}  # Mg-5Gd seed-cache median ||z||={med:.2f} + {args.band}", txt)
    txt, n3 = re.subn(r"(?m)^(\s*seed_rows:)\s*\S+.*$",
                      rf"\g<1> {len(Z)}  # rows in {cache.name}", txt)
    if not (n1 == n2 == n3 == 1):
        print(f"ERROR: shell rewrite matched min={n1} max={n2} seed_rows={n3} (expected 1 each)")
        return 1
    cfg_path.write_text(txt)
    print(f"[shell] shell_radius=[{lo}, {hi}], seed_rows={len(Z)} -> {cfg_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("targets", help="derive v2 targets from seed sims")
    t.add_argument("--runs-dir", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--percentile", type=float, default=80.0)
    t.set_defaults(fn=cmd_targets)

    s = sub.add_parser("shell", help="calibrate MERIDIAN shell radius from cache")
    s.add_argument("--cache", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--band", type=float, default=0.15)
    s.set_defaults(fn=cmd_shell)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
