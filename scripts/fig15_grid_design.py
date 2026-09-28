#!/usr/bin/env python
"""Design the Fig. 15 target-achievability grid and emit one config per cell.

The reachable (sigma_y, n) region of a given decoder is not a rectangle: the
strength-hardening trade-off makes it a narrow diagonal ribbon (AZ31/FM-DiT-512:
corr = -0.86, principal-axis anisotropy 12.9x). A rectangular target grid
therefore spends most of its cells in a corner no decoder can express, which
measures nothing. This script instead builds the grid in the ribbon's own frame:

    axis 1 (along)  = the trade-off direction        -> probes steerability
    axis 2 (across) = the direction that beats it    -> probes the boundary

Both axes are expressed in seed-population sigma, the same units the V3
objective uses via ``scales``, so a cell's residual is directly comparable to
its grid coordinate.

Two things are emitted per cell:
  * a standalone YAML config carrying that cell's V3 targets, and
  * a trust-region schedule relaxed in proportion to how far the target sits
    from the seed anchor cloud, so that a missed far target cannot be dismissed
    as an under-searched one.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

ALLOYS = {
    "AZ31": dict(
        base_config="configs/acta2026/az31_fmdit_meridian_v2.yaml",
        seed_cache="data/seed_caches/seed_cache_AZ31_v2_fmdit.npz",
        visited=["results/AZ31_v2_fmdit/*/log.jsonl", "results/AZ31_v1_fmdit/*/log.jsonl"],
    ),
    "Mg5Gd": dict(
        base_config="configs/acta2026/mg5gd_fmdit_meridian_v2.yaml",
        seed_cache="data/seed_caches/seed_cache_Mg5Gd_v2_fmdit.npz",
        visited=["results/Mg5Gd_v2_fmdit/*/log.jsonl", "results/Mg5Gd_v3_fmdit/*/log.jsonl"],
    ),
}

# Grid coordinates, in seed-sigma, relative to the visited-cloud centre.
T1_LEVELS = [-5.0, -1.8, 1.4, 4.6]     # along the ribbon: all nominally reachable
T2_LEVELS = [-0.8, 0.6, 2.0, 3.4]      # across it: interior -> edge -> beyond


def load_visited(patterns):
    pts = []
    for pat in patterns:
        for path in sorted(glob.glob(str(ROOT / pat))):
            with open(path) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    p = r.get("props")
                    if r.get("success") and isinstance(p, dict) \
                            and p.get("sigma_y") and p.get("n"):
                        pts.append((p["sigma_y"], p["n"]))
    if not pts:
        raise SystemExit(f"no visited points matched {patterns}")
    return np.asarray(pts, dtype=float)


def load_seed_props(cache_npz: Path):
    props = json.loads(Path(str(cache_npz).replace(".npz", ".properties.json")).read_text())
    rows = []
    for r in props:
        p = r.get("props", r) if isinstance(r, dict) else None
        if isinstance(p, dict) and p.get("sigma_y") and p.get("n"):
            rows.append((p["sigma_y"], p["n"]))
    return np.asarray(rows, dtype=float)


def ribbon_frame(visited, seeds):
    """Seed-standardised frame whose axes are the ribbon's principal directions."""
    mu, sd = seeds.mean(0), seeds.std(0)
    Z = (visited - mu) / sd
    evals, evecs = np.linalg.eigh(np.cov(Z.T))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    u1, u2 = evecs[:, 0], evecs[:, 1]
    if u1[0] > 0:            # u1 -> low sigma_y / high n (soft, ductile end)
        u1 = -u1
    if u2[0] < 0:            # u2 -> high sigma_y AND high n (beats the trade-off)
        u2 = -u2
    return dict(mu=mu, sd=sd, u1=u1, u2=u2, ctr=Z.mean(0),
                sd_along=float(np.sqrt(evals[0])), sd_across=float(np.sqrt(evals[1])),
                anisotropy=float(evals[0] / evals[1]))


def tr_schedule(d_seed: float) -> dict:
    """Relax MERIDIAN's trust region in proportion to target distance.

    ``L_min`` is a *restart* threshold, not a floor: once L falls below it the
    optimizer cold-starts from the class pool, discarding whatever distance it
    had travelled. Far targets therefore want a lower L_min, not a higher one.
    Likewise tau_fail = ceil(k/q * factor) capped by failure_tolerance_max, so
    the factor has to move for the cap to matter at all.
    """
    f = float(np.clip(d_seed / 5.0, 0.0, 1.0))
    return dict(
        L_init=round(0.6 + 0.4 * f, 4),
        L_min=round(0.05 - 0.02 * f, 4),
        L_max=round(1.0 + 0.4 * f, 4),
        success_tolerance=3 if f < 0.5 else 2,
        failure_tolerance_factor=round(1.0 + 1.5 * f, 4),
        failure_tolerance_max=int(round(8 + 6 * f)),
        _distance_fraction=round(f, 4),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", default="AZ31", choices=sorted(ALLOYS))
    ap.add_argument("--tag", default="512", help="decoder tag used in output names")
    ap.add_argument("--outdir", default="configs/acta2026/fig15")
    args = ap.parse_args()

    spec = ALLOYS[args.alloy]
    base_path = ROOT / spec["base_config"]
    cache = ROOT / spec["seed_cache"]

    visited = load_visited(spec["visited"])
    seeds = load_seed_props(cache)
    fr = ribbon_frame(visited, seeds)
    mu, sd, u1, u2, ctr = fr["mu"], fr["sd"], fr["u1"], fr["u2"], fr["ctr"]
    Zv, Zs = (visited - mu) / sd, (seeds - mu) / sd

    base = yaml.safe_load(base_path.read_text())
    scales = {"sigma_y": float(sd[0]), "n": float(sd[1])}

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "data/seed_caches/fig15" / f"{args.alloy}_{args.tag}"

    print(f"{args.alloy} / {args.tag}: {len(visited)} visited, {len(seeds)} seeds")
    print(f"  seed anchor mu=({mu[0]:.3f}, {mu[1]:.5f})  sd=({sd[0]:.4f}, {sd[1]:.6f})")
    print(f"  ribbon anisotropy {fr['anisotropy']:.1f}x   corr={np.corrcoef(visited.T)[0,1]:+.3f}")
    print(f"  u_along=({u1[0]:+.4f},{u1[1]:+.4f})  u_across=({u2[0]:+.4f},{u2[1]:+.4f})\n")
    print(f"{'cell':>4} {'t_along':>8} {'t_across':>9} {'sigma_y':>8} {'n':>8} "
          f"{'d_visit':>8} {'d_seed':>7} {'L_init':>7} {'tau_f':>6}  prediction")

    cells = []
    for i, t1 in enumerate(T1_LEVELS):
        for j, t2 in enumerate(T2_LEVELS):
            idx = len(cells)
            z = ctr + t1 * u1 + t2 * u2
            phys = mu + z * sd
            sigma_y, n = float(phys[0]), float(phys[1])
            d_visit = float(np.linalg.norm(Zv - z, axis=1).min())
            d_seed = float(np.linalg.norm(Zs - z, axis=1).min())
            sched = tr_schedule(d_seed)
            pred = "reachable" if d_visit < 0.35 else ("edge" if d_visit < 1.0 else "beyond")

            cfg = json.loads(json.dumps(base))          # deep copy
            cfg["objective"]["name"] = "v3"
            cfg["objective"]["v3"] = {
                "targets": {"sigma_y": round(sigma_y, 4), "n": round(n, 6)},
                "weights": {"sigma_y": 1.0, "n": 1.0},
                "scales": {k: round(v, 6) for k, v in scales.items()},
                "g_min": 250, "lambda_g": 2.0,
                "g_max": 1100, "lambda_g_hi": 2.0,
            }
            mer = cfg["optimizer"]["meridian"]
            for k, v in sched.items():
                if not k.startswith("_"):
                    mer[k] = v
            cell_cache = cache_dir / f"seed_cache_c{idx:02d}.npz"
            cfg["loop"]["seed_cache"] = str(cell_cache.relative_to(ROOT))
            cfg["experiment"]["name"] = f"fig15/{args.alloy}_{args.tag}_c{idx:02d}"
            cfg["ablation"] = {"decoders": [base["decoder"]["name"]],
                               "optimizers": ["meridian"], "objectives": ["v3"],
                               "n_repeats": 1}
            cfg["fig15_cell"] = {
                "index": idx, "t_along": t1, "t_across": t2,
                "sigma_y": round(sigma_y, 4), "n": round(n, 6),
                "d_nearest_visited_seed_sigma": round(d_visit, 4),
                "d_nearest_seed_seed_sigma": round(d_seed, 4),
                "prediction": pred,
                "tr_schedule_distance_fraction": sched["_distance_fraction"],
            }

            cfg_path = outdir / f"{args.alloy.lower()}_{args.tag}_fig15_c{idx:02d}.yaml"
            header = (
                f"# AUTO-GENERATED by scripts/fig15_grid_design.py -- do not edit by hand.\n"
                f"# Source config : {spec['base_config']}\n"
                f"# Fig.15 cell   : {idx}  (t_along={t1:+.2f}, t_across={t2:+.2f} seed-sigma)\n"
                f"# Target        : sigma_y={sigma_y:.3f} MPa, n={n:.5f}\n"
                f"# Nearest visited point {d_visit:.3f} seed-sigma away -> predicted '{pred}'.\n"
                f"# Nearest seed {d_seed:.3f} seed-sigma away -> trust region relaxed to\n"
                f"#   L_init={sched['L_init']}, L_max={sched['L_max']}, L_min={sched['L_min']},\n"
                f"#   success_tol={sched['success_tolerance']}, "
                f"fail_factor={sched['failure_tolerance_factor']}, "
                f"fail_max={sched['failure_tolerance_max']}\n"
            )
            cfg_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))

            tau_f = min(math.ceil(16 / 4 * sched["failure_tolerance_factor"]),
                        sched["failure_tolerance_max"])
            print(f"{idx:>4} {t1:>8.1f} {t2:>9.1f} {sigma_y:>8.2f} {n:>8.5f} "
                  f"{d_visit:>8.3f} {d_seed:>7.3f} {sched['L_init']:>7.2f} {tau_f:>6d}  {pred}")

            cells.append(dict(index=idx, t_along=t1, t_across=t2,
                              sigma_y=sigma_y, n=n, d_visit=d_visit, d_seed=d_seed,
                              prediction=pred, config=str(cfg_path.relative_to(ROOT)),
                              seed_cache=str(cell_cache.relative_to(ROOT)),
                              tr_schedule={k: v for k, v in sched.items() if not k.startswith("_")}))

    grid = dict(
        alloy=args.alloy, tag=args.tag, base_config=spec["base_config"],
        source_seed_cache=spec["seed_cache"],
        frame=dict(mu=mu.tolist(), sd=sd.tolist(), u_along=u1.tolist(),
                   u_across=u2.tolist(), centre=ctr.tolist(),
                   sd_along=fr["sd_along"], sd_across=fr["sd_across"],
                   anisotropy=fr["anisotropy"],
                   corr_sigma_y_n=float(np.corrcoef(visited.T)[0, 1]),
                   n_visited=int(len(visited)), n_seeds=int(len(seeds))),
        t_along_levels=T1_LEVELS, t_across_levels=T2_LEVELS, cells=cells,
    )
    grid_path = ROOT / "configs/acta2026/fig15" / f"grid_{args.alloy}_{args.tag}.json"
    grid_path.write_text(json.dumps(grid, indent=2))
    n_pred = {k: sum(c["prediction"] == k for c in cells) for k in ("reachable", "edge", "beyond")}
    print(f"\npredicted outcome: {n_pred}")
    print(f"wrote {len(cells)} configs to {outdir}/ and the grid spec to {grid_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
