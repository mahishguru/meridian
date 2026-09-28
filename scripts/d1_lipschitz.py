#!/usr/bin/env python
"""D1: landscape-regularisation diagnostic for the Acta smoothness figure.

Measures the response change |dJ| caused by a controlled perturbation applied in
two different places, both scored by the same DAMASK oracle:

  latent  z -> z + eps*u, then decode -> codec -> DAMASK   (probes J o f o D)
  direct  the RVE itself is perturbed and fed to DAMASK    (probes J o f)

The direct arm has two physically-interpretable knobs:
  orientation  every grain is rotated by an independent random rotation of RMS
               magnitude delta
  morphology   grain boundaries migrate until a fraction p of all pixels
               have changed their grain ID

Star design: each base point is simulated once and that single centre value is
shared by all of its partners, so K pairs cost K+1 simulations instead of 2K.

Results are appended to a JSONL file one endpoint at a time, so an interrupted
shard can be restarted with --resume.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meridian.config import apply_overrides, load_config  # noqa: E402

D3D_BASE = "DataContainers/SyntheticVolumeDataContainer"

# Perturbation budgets. The latent radii bracket the ~0.8-1.2 step length the
# optimizers actually take per iteration; the smaller value is the sub-step
# control. The direct budgets are the physical counterparts.
LATENT_EPS = (0.25, 1.0)
ORIENT_DEG = (5.0, 15.0)
MORPH_FRAC = (0.01, 0.05)


def _localise(p: str) -> Path:
    """Seed-cache manifest paths are absolute or relative to the repository."""
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def perturb_orientation(src: Path, dst: Path, delta_deg: float,
                        rng: np.random.Generator) -> float:
    """Rotate every grain by an independent random rotation of RMS delta_deg."""
    import h5py
    from scipy.spatial.transform import Rotation

    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        g = f[D3D_BASE]
        eul = np.asarray(g["Grain Data/EulerAngles"][:], dtype=np.float64)
        n = eul.shape[0]
        axis = rng.normal(size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        ang = rng.normal(scale=np.deg2rad(delta_deg), size=n)
        new = (Rotation.from_rotvec(axis * ang[:, None])
               * Rotation.from_euler("ZXZ", eul)).as_euler("ZXZ")
        new[:, 0] %= 2.0 * np.pi
        new[:, 2] %= 2.0 * np.pi
        new[0] = 0.0  # row 0 is the unassigned-cell dummy
        g["Grain Data/EulerAngles"][...] = new.astype(np.float32)
        fid = np.asarray(g["CellData/FeatureIds"][:][..., 0])
        g["CellData/EulerAngles"][...] = new[fid].astype(np.float32)
    return float(np.sqrt(np.mean(np.rad2deg(ang[1:]) ** 2)))


def perturb_morphology(src: Path, dst: Path, frac: float,
                       rng: np.random.Generator) -> float:
    """Migrate grain boundaries until `frac` of all pixels have changed ID.

    Only boundary pixels are candidates: an interior pixel would simply adopt
    its own grain's ID, so sampling uniformly wastes almost the whole budget.
    """
    import h5py

    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        g = f[D3D_BASE]
        eul = np.asarray(g["Grain Data/EulerAngles"][:], dtype=np.float64)
        old = np.asarray(g["CellData/FeatureIds"][:][..., 0])[0]
        ny, nx = old.shape
        cur = old.copy()
        target = int(round(frac * old.size))
        dy = np.array([-1, 1, 0, 0])
        dx = np.array([0, 0, -1, 1])

        for _ in range(200):
            need = target - int(np.count_nonzero(cur != old))
            if need <= 0:
                break
            vert = cur[:-1, :] != cur[1:, :]
            horz = cur[:, :-1] != cur[:, 1:]
            bnd = np.zeros_like(cur, dtype=bool)
            bnd[:-1, :] |= vert
            bnd[1:, :] |= vert
            bnd[:, :-1] |= horz
            bnd[:, 1:] |= horz
            ys, xs = np.nonzero(bnd)
            if ys.size == 0:
                break
            take = int(min(ys.size, max(1, need)))
            sel = rng.choice(ys.size, take, replace=False)
            ys, xs = ys[sel], xs[sel]
            d = rng.integers(0, 4, take)
            yy = np.clip(ys + dy[d], 0, ny - 1)
            xx = np.clip(xs + dx[d], 0, nx - 1)
            cur[ys, xs] = cur[yy, xx]

        # A vanished grain would desynchronise FeatureIds from Grain Data, so
        # give every lost grain one pixel of its original body back.
        for gid in np.setdiff1d(np.unique(old), np.unique(cur)):
            py, px = np.argwhere(old == gid)[0]
            cur[py, px] = gid

        changed = float(np.mean(cur != old))
        out = cur[None, :, :]
        g["CellData/FeatureIds"][...] = out[..., None].astype(np.int32)
        g["CellData/EulerAngles"][...] = eul[out].astype(np.float32)
    return changed


def build_latent_jobs(cache, n_base, n_dir, bounds, rng) -> list[dict]:
    Z, Y = cache["Z"], cache["Y"]
    idx = np.flatnonzero(cache["success"].astype(bool) & np.isfinite(Y))
    # Spread the bases over the objective range so the estimate is not a
    # property of one corner of the manifold.
    idx = idx[np.argsort(Y[idx])]
    picks = idx[np.linspace(0, len(idx) - 1, n_base).round().astype(int)]

    jobs: list[dict] = []
    for b, bi in enumerate(picks):
        z0 = Z[bi].astype(np.float64)
        jobs.append({"job_id": "lat_b%02d_centre" % b, "arm": "latent",
                     "kind": "centre", "base": b, "role": "centre",
                     "budget": 0.0, "z": z0.tolist(), "step": 0.0})
        for eps in LATENT_EPS:
            for k in range(n_dir):
                u = rng.normal(size=z0.shape)
                u /= np.linalg.norm(u)
                z1 = z0 + eps * u
                if bounds is not None:
                    z1 = np.clip(z1, bounds[0], bounds[1])
                jobs.append({"job_id": "lat_b%02d_e%g_k%d" % (b, eps, k),
                             "arm": "latent", "kind": "latent", "base": b,
                             "role": "partner", "budget": float(eps),
                             "z": z1.tolist(),
                             "step": float(np.linalg.norm(z1 - z0))})
    return jobs


def build_direct_jobs(manifest, n_base, n_dir, rng) -> list[dict]:
    usable = [e for e in manifest
              if e.get("success") and _localise(e["png"]).is_file()]
    if len(usable) < n_base:
        raise SystemExit("only %d usable base RVEs" % len(usable))
    usable.sort(key=lambda e: e["objective"])
    picks = [usable[i] for i in
             np.linspace(0, len(usable) - 1, n_base).round().astype(int)]

    jobs: list[dict] = []
    for b, e in enumerate(picks):
        png = str(_localise(e["png"]))
        jobs.append({"job_id": "dir_b%02d_centre" % b, "arm": "direct",
                     "kind": "centre", "base": b, "role": "centre",
                     "budget": 0.0, "png": png, "sim_id": e["sim_id"]})
        for kind, budgets in (("orientation", ORIENT_DEG),
                              ("morphology", MORPH_FRAC)):
            for q in budgets:
                for k in range(n_dir):
                    jobs.append({
                        "job_id": "dir_b%02d_%s_q%g_k%d" % (b, kind[:3], q, k),
                        "arm": "direct", "kind": kind, "base": b,
                        "role": "partner", "budget": float(q), "png": png,
                        "sim_id": e["sim_id"],
                        "seed": int(rng.integers(0, 2**31 - 1))})
    return jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--arm", required=True, choices=["latent", "direct"])
    p.add_argument("--alloy", required=True)
    p.add_argument("--seed-cache", required=True)
    p.add_argument("--out", required=True, help="results JSONL path")
    p.add_argument("--work-root", required=True)
    p.add_argument("--n-base", type=int, default=10)
    p.add_argument("--n-dir", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="smoke test: run only the first N jobs of the shard")
    p.add_argument("--override", "-o", nargs="*", default=[])
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    rng = np.random.default_rng(args.seed)
    cache_path = Path(args.seed_cache)

    if args.arm == "latent":
        cache = dict(np.load(cache_path, allow_pickle=True))
        bounds = (tuple(cfg.latent.bounds)
                  if cfg.latent.use_box_constraint else None)
        jobs = build_latent_jobs(cache, args.n_base, args.n_dir, bounds, rng)
    else:
        manifest = json.loads(
            cache_path.with_suffix(".manifest.json").read_text())
        jobs = build_direct_jobs(manifest, args.n_base, args.n_dir, rng)

    # Shard by base point so a centre never lands in a different shard than the
    # partners that need it.
    jobs = [j for j in jobs if j["base"] % args.n_shards == args.shard]
    if args.limit:
        jobs = jobs[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out_path.is_file():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("success"):
                done.add(rec["job_id"])
        jobs = [j for j in jobs if j["job_id"] not in done]

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print("[d1] arm=%s alloy=%s shard=%d/%d jobs=%d skipped=%d"
          % (args.arm, args.alloy, args.shard, args.n_shards, len(jobs),
             len(done)), flush=True)
    if not jobs:
        return 0

    from meridian.objectives import get_objective
    from meridian.simulation import SimulationPipeline
    from meridian.simulation.damask import DAMASKRunner
    from meridian.simulation.extractor import PropertyExtractor
    from meridian.simulation.postproc import StressStrainProcessor

    objective = get_objective(cfg)
    s = cfg.simulation
    damask = DAMASKRunner(
        load_yaml=s.load_yaml, phase_yaml=s.phase_yaml,
        target_phase=s.target_phase, target_homog=s.target_homog,
        damask_binary=s.damask_binary, n_threads=int(s.n_threads),
        timeout_seconds=int(s.timeout_seconds), bind_cpus=False,
    )
    post = StressStrainProcessor(loading_direction=s.loading_direction)
    ext = PropertyExtractor(
        youngs_strain_max=float(cfg.extraction.youngs_strain_max),
        yield_tolerance=float(cfg.extraction.yield_tolerance),
    )
    spacing_raw = getattr(cfg.codec, "spacing", None)
    spacing = np.array(spacing_raw, dtype=np.float32) if spacing_raw else None
    min_grains = getattr(cfg.codec, "min_grains", None)
    target_size = getattr(cfg.codec, "target_size", None)
    min_pixel_std = getattr(cfg.codec, "min_pixel_std", None)

    pipeline = None
    decoder = None
    decode_lock = threading.Lock()
    base_d3d: dict[int, Path] = {}

    if args.arm == "latent":
        import torch
        from meridian.decoders import get_decoder

        torch.manual_seed(int(cfg.experiment.seed))
        decoder = get_decoder(cfg)
        pipeline = SimulationPipeline(
            damask=damask, postproc=post, extractor=ext,
            class_means_json=cfg.codec.class_means_json,
            class_key=cfg.codec.class_key,
            template_dream3d=getattr(cfg.codec, "template_dream3d", None),
            colour_tol=getattr(cfg.codec, "colour_tol", None),
            spacing=spacing, cleanup_hdf5=bool(s.cleanup_hdf5),
            min_grains=int(min_grains) if min_grains is not None else None,
            target_size=int(target_size) if target_size is not None else None,
            min_pixel_std=(float(min_pixel_std)
                           if min_pixel_std is not None else None),
        )
    else:
        # One segmentation per base RVE, reused by the centre and every
        # partner, so the two arms differ only in where the noise is injected.
        from meridian.simulation.codec import png_to_dream3d

        d3d_dir = work_root / "_base_rve"
        d3d_dir.mkdir(parents=True, exist_ok=True)
        for b in sorted({j["base"] for j in jobs}):
            png = Path(next(j["png"] for j in jobs if j["base"] == b))
            out = d3d_dir / ("base_b%02d.dream3d" % b)
            if not out.is_file():
                png_to_dream3d(
                    png_path=png,
                    class_means_json=cfg.codec.class_means_json,
                    class_key=cfg.codec.class_key, output_dream3d=out,
                    template_dream3d=None,
                    colour_tol=getattr(cfg.codec, "colour_tol", None),
                    spacing=spacing,
                    min_grains=(int(min_grains)
                                if min_grains is not None else None),
                    target_size=(int(target_size)
                                 if target_size is not None else None),
                    min_pixel_std=None,
                )
            base_d3d[b] = out
            print("[d1] base RVE b%02d -> %s" % (b, out.name), flush=True)

    write_lock = threading.Lock()
    fh = out_path.open("a")

    def emit(rec: dict) -> None:
        with write_lock:
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()

    def run_job(job: dict) -> dict:
        import h5py

        t0 = time.time()
        rec = {k: v for k, v in job.items() if k != "z"}
        rec.update({"alloy": args.alloy, "success": False,
                    "objective": None, "error": None})
        sim_dir = work_root / job["job_id"]
        try:
            sim_dir.mkdir(parents=True, exist_ok=True)
            if job["arm"] == "latent":
                import torch
                z = np.asarray(job["z"], dtype=np.float32)
                with decode_lock:
                    with torch.no_grad():
                        img = decoder.decode(
                            torch.from_numpy(z[None]).float())[0]
                png = sim_dir / (job["job_id"] + ".png")
                img.save(png)
                outcome = pipeline.evaluate_png(png, work_dir=sim_dir, z=z)
                if not outcome.success:
                    raise RuntimeError(outcome.error or "pipeline_failed")
                props = outcome.properties
            else:
                src = base_d3d[job["base"]]
                d3d = sim_dir / (job["job_id"] + ".dream3d")
                if job["kind"] == "centre":
                    shutil.copyfile(src, d3d)
                    rec["applied"] = 0.0
                else:
                    jrng = np.random.default_rng(job["seed"])
                    fn = (perturb_orientation if job["kind"] == "orientation"
                          else perturb_morphology)
                    rec["applied"] = fn(src, d3d, job["budget"], jrng)
                with h5py.File(d3d, "r") as f:
                    ng = int(len(np.unique(
                        f[D3D_BASE + "/CellData/FeatureIds"][:])))
                try:
                    res = damask.run(d3d, sim_dir=sim_dir)
                    if res.returncode != 0 or res.hdf5_path is None:
                        raise RuntimeError(
                            "damask_failed: rc=%d" % res.returncode)
                    csv = sim_dir / "damask_stress_strain.csv"
                    post.process_to_csv(res.hdf5_path, csv)
                    props = ext.extract(csv)
                finally:
                    if bool(s.cleanup_hdf5):
                        DAMASKRunner.cleanup(sim_dir)
                if props is None:
                    raise RuntimeError("extraction_returned_none")
                props.n_grains = ng

            rec["objective"] = float(objective(props))
            rec["n_grains"] = int(getattr(props, "n_grains", -1))
            rec["props"] = {k: float(v) for k, v in props.to_dict().items()
                            if isinstance(v, (int, float))}
            rec["success"] = True
        except Exception as exc:
            rec["error"] = repr(exc)
        finally:
            for pat in ("*.dream3d", "*.vti", "*.hdf5"):
                for junk in sim_dir.glob(pat):
                    junk.unlink(missing_ok=True)
        rec["seconds"] = round(time.time() - t0, 1)
        return rec

    n_done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for rec in pool.map(run_job, jobs):
            emit(rec)
            n_done += 1
            print("[d1] %d/%d %s %s J=%s %s"
                  % (n_done, len(jobs), "OK " if rec["success"] else "FAIL",
                     rec["job_id"], rec["objective"], rec["error"] or ""),
                  flush=True)
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
