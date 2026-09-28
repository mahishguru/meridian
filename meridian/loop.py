"""Outer optimization loop."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from meridian.config import ConfigNode
from meridian.decoders import get_decoder
from meridian.latent import BoxConstraint, PhysicsPenalty, build_initial_population
from meridian.objectives import get_objective
from meridian.optimizers import get_optimizer
from meridian.simulation import SimulationOutcome


@dataclass
class IterationRecord:
    iteration: int
    z: np.ndarray
    objective: float
    properties: Optional[dict]
    success: bool
    error: Optional[str]
    elapsed_s: float


class OptimizationLoop:
    """Drives suggest -> decode -> simulate -> objective -> update."""

    def __init__(self, cfg: ConfigNode) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "sims").mkdir(exist_ok=True)

        np.random.seed(int(cfg.experiment.seed))
        torch.manual_seed(int(cfg.experiment.seed))

        self.decoder = get_decoder(cfg)
        self.objective = get_objective(cfg)
        self.optimizer = get_optimizer(cfg)

        self.box = BoxConstraint(*cfg.latent.bounds) if cfg.latent.use_box_constraint else None
        self.physics = (
            PhysicsPenalty(
                min_pixel_std=float(cfg.latent.physics_penalty.min_pixel_std),
                weight=float(cfg.latent.physics_penalty.weight),
            )
            if cfg.latent.use_physics_penalty
            else None
        )

        self._sim_pipeline = None
        if not bool(cfg.loop.dry_run):
            self._sim_pipeline = self._build_pipeline()

        self.records: list[IterationRecord] = []

        # Rows at the front of optimizer.X/Y that came from the seed cache.
        # Their Y are ground-truth DAMASK properties of real microstructures,
        # not decoder outputs, so they must not be reported as a found design.
        self._n_seed = 0

        # Adaptive "too_few_grains" rejection penalty. Set later by
        # initialize_from_cache() from the seed cache's valid-Y range so that
        # rejected samples always score *worse* than the worst valid sim
        # (otherwise the optimizer climbs into the rejection plateau).
        # Defaults match the legacy V1-tuned constants.
        self._reject_floor: float = -1.0   # worst score a "near-miss" reject gets
        self._reject_span: float = 2.0     # extra penalty for full-collapse

    def _build_pipeline(self):
        from meridian.simulation import SimulationPipeline
        from meridian.simulation.damask import DAMASKRunner
        from meridian.simulation.extractor import PropertyExtractor
        from meridian.simulation.postproc import StressStrainProcessor

        s = self.cfg.simulation
        parallel_batch = int(getattr(s, "parallel_batch", 1) or 1)
        damask = DAMASKRunner(
            load_yaml=s.load_yaml, phase_yaml=s.phase_yaml,
            target_phase=s.target_phase, target_homog=s.target_homog,
            damask_binary=s.damask_binary, n_threads=int(s.n_threads),
            timeout_seconds=int(s.timeout_seconds),
            bind_cpus=(parallel_batch <= 1),
        )
        post = StressStrainProcessor(loading_direction=s.loading_direction)
        ext = PropertyExtractor(
            youngs_strain_max=float(self.cfg.extraction.youngs_strain_max),
            yield_tolerance=float(self.cfg.extraction.yield_tolerance),
        )
        if self.cfg.codec.class_means_json is None:
            raise ValueError("codec.class_means_json must be set for non-dry-run mode.")

        # Determine template mode (None → segmentation-based, else legacy)
        template = getattr(self.cfg.codec, "template_dream3d", None)
        if template is None:
            template = self._find_template_dream3d_optional()
        else:
            template = Path(template) if template else None

        # Colour tolerance for segmentation. Accepts:
        #   None / null         -> orientation_codec default (tol=1)
        #   integer             -> segment_grains(colour_tol=N)
        #   "smart" or
        #   "smart:NCOL:MED:MIN" -> median+quantize+CC+Voronoi pipeline
        #                          (robust to FM-DiT / diffusion outputs).
        colour_tol_raw = getattr(self.cfg.codec, "colour_tol", None)
        if colour_tol_raw is None:
            colour_tol = None
        elif isinstance(colour_tol_raw, str):
            colour_tol = colour_tol_raw
        else:
            colour_tol = int(colour_tol_raw)

        # Spacing
        spacing_raw = getattr(self.cfg.codec, "spacing", None)
        import numpy as _np
        spacing = _np.array(spacing_raw, dtype=_np.float32) if spacing_raw else None

        return SimulationPipeline(
            damask=damask, postproc=post, extractor=ext,
            class_means_json=self.cfg.codec.class_means_json,
            class_key=self.cfg.codec.class_key,
            template_dream3d=template,
            colour_tol=colour_tol,
            spacing=spacing,
            cleanup_hdf5=bool(s.cleanup_hdf5),
            min_grains=getattr(self.cfg.codec, "min_grains", None),
            target_size=getattr(self.cfg.codec, "target_size", None),
            min_pixel_std=getattr(self.cfg.codec, "min_pixel_std", None),
            calibrate=bool(getattr(self.cfg.codec, "calibrate", False)),
            texture_prior_dir=getattr(self.cfg.codec, "texture_prior_dir", None),
        )

    def _find_template_dream3d_optional(self) -> Path | None:
        """Return a template .dream3d if available, else None (→ segmentation mode)."""
        geom_dir = getattr(self.cfg.simulation, "geometries_dir", None)
        if geom_dir:
            for f in Path(geom_dir).glob("*.dream3d"):
                return f
        return None  # segmentation mode

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _decode_and_save(self, z_batch: np.ndarray, iter_idx: int) -> list[Path]:
        z_t = torch.from_numpy(z_batch).float()
        images = self.decoder.decode(z_t)
        paths = []
        for j, img in enumerate(images):
            p = self.output_dir / "images" / f"iter{iter_idx:04d}_b{j:02d}.png"
            img.save(p)
            paths.append(p)
        return paths

    def _evaluate_one(self, z: np.ndarray, png_path: Path, iter_idx: int, b_idx: int) -> tuple[SimulationOutcome, float]:
        if self.cfg.loop.dry_run:
            from meridian.simulation.mock import mock_simulate
            outcome = mock_simulate(z)
        else:
            sim_dir = self.output_dir / "sims" / f"iter{iter_idx:04d}_b{b_idx:02d}"
            outcome = self._sim_pipeline.evaluate_png(png_path, work_dir=sim_dir, z=z)

        if not outcome.success or outcome.properties is None:
            # Soft-penalize "too_few_grains" rejections so the optimizer gets a
            # finite, monotone gradient ("more grains → better") instead of an
            # uninformative -inf in every direction. Trust-region methods (DANTE,
            # MERIDIAN, BAxUS) cannot recover from a region where every sample
            # scores -inf — they need a usable signal to climb back out.
            err = outcome.error or ""
            if "too_few_grains" in err:
                import re
                m = re.search(r"n_grains=(\d+)\s*<\s*min_grains=(\d+)", err)
                if m:
                    n_g = int(m.group(1))
                    g_min = max(int(m.group(2)), 1)
                    shortfall = max(g_min - n_g, 0) / float(g_min)  # in [0, 1]
                    # Penalty MUST stay strictly below the worst valid sim
                    # (self._reject_floor), with extra deficit (self._reject_span)
                    # for full collapse. Both are tuned by initialize_from_cache()
                    # from the seed-cache Y range. Range: [floor - span, floor].
                    return outcome, float(self._reject_floor - self._reject_span * shortfall)
            if "blank_decode" in err:
                # Watercolor-blob regime: deepest penalty (full span shortfall).
                # Distinct failure label from too_few_grains so the surrogate
                # can model it separately if it carries a feasibility head.
                return outcome, float(self._reject_floor - self._reject_span)
            return outcome, float("-inf")
        # Drop per-sim properties.json for offline inspection.
        if outcome.sim_dir is not None:
            try:
                (outcome.sim_dir / "properties.json").write_text(
                    json.dumps(outcome.properties.to_dict(), indent=2))
            except Exception:
                pass
        score = self.objective(outcome.properties)
        if self.physics is not None:
            from PIL import Image
            score += self.physics(Image.open(png_path))
        return outcome, score

    def _append_log(self, record: IterationRecord, batch_idx: int) -> None:
        """Stream a single evaluation to log.jsonl so partial results survive a kill."""
        try:
            with (self.output_dir / "log.jsonl").open("a") as f:
                f.write(json.dumps({
                    "iter": record.iteration, "batch_idx": batch_idx,
                    "objective": record.objective,
                    "success": record.success, "error": record.error,
                    "elapsed_s": record.elapsed_s, "props": record.properties,
                }) + "\n")
        except Exception:
            pass

    def _evaluate_batch(self, Z: np.ndarray, png_paths: list[Path],
                        iter_idx: int, desc: str) -> np.ndarray:
        """Evaluate a batch of (z, png) pairs, optionally in parallel.

        Each member runs an independent DAMASK subprocess in its own work dir,
        so they are safe to run concurrently. Per-evaluation logging and the
        per-sim properties.json drop are preserved.

        Returns
        -------
        Y : (n,) float32 ndarray of objective values (incl. soft-penalty rejects)
        success : (n,) bool ndarray, True iff the underlying simulation
                  pipeline reported ``outcome.success``.
        """
        n = len(Z)
        Y = np.empty(n, dtype=np.float32)
        success = np.zeros(n, dtype=bool)
        parallel = int(getattr(self.cfg.simulation, "parallel_batch", 1) or 1)
        parallel = max(1, min(parallel, n))

        def _job(i: int) -> tuple[int, IterationRecord]:
            t0 = time.time()
            out, y = self._evaluate_one(Z[i], png_paths[i], iter_idx, i)
            elapsed = time.time() - t0
            rec = IterationRecord(
                iteration=iter_idx, z=Z[i], objective=y,
                properties=(out.properties.to_dict() if out.properties else None),
                success=out.success, error=out.error, elapsed_s=elapsed,
            )
            return i, rec

        if parallel == 1:
            iterator = (_job(i) for i in range(n))
            for i, rec in tqdm(iterator, total=n, desc=desc):
                Y[i] = rec.objective
                success[i] = bool(rec.success)
                self.records.append(rec)
                self._append_log(rec, i)
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(_job, i) for i in range(n)]
                # Iterate in submission order so b00, b01, ... are logged in order.
                for fut in tqdm(futures, total=n, desc=f"{desc}[x{parallel}]"):
                    i, rec = fut.result()
                    Y[i] = rec.objective
                    success[i] = bool(rec.success)
                    self.records.append(rec)
                    self._append_log(rec, i)
        return Y, success

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _surrogate_mask(self, Y: np.ndarray, success: np.ndarray) -> np.ndarray:
        """Indices fed to the surrogate.

        Default behaviour (``loop.mask_failed_from_surrogate=true``): only
        rows whose underlying simulation truly succeeded are exposed to the
        optimizer. Soft-penalty rejects (``-10.37``) are *kept in the log*
        for diagnostics but excluded from the GP/DNN fit so they don't
        distort posterior variance/mean. Set the flag to false to fall back
        to the legacy behaviour (any finite Y is accepted).

        Per-optimizer override (``optimizer.wants_reject_feedback=True``):
        forwards floor-penalised rejects to the optimizer too. BAxUS sets
        this so its low-dim subspace fit doesn't go blind to 30-40% of
        evaluations (B2).
        """
        if getattr(self.optimizer, "wants_reject_feedback", False):
            return np.isfinite(Y)
        if bool(getattr(self.cfg.loop, "mask_failed_from_surrogate", True)):
            return np.asarray(success, dtype=bool) & np.isfinite(Y)
        return np.isfinite(Y)

    def _push_to_optimizer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        valid: np.ndarray,
        properties: list[dict | None] | None = None,
    ) -> None:
        """Dispatch optimizer update.

        If the optimizer advertises ``update_with_properties`` (currently
        only MERIDIAN with ``use_property_acquisition: true``), forward the
        per-row property dicts so its auxiliary heads can be trained. All
        other optimizers fall back to the standard ``update`` signature so
        baseline behaviour is untouched (NeurIPS-fairness).
        """
        if (
            properties is not None
            and hasattr(self.optimizer, "update_with_properties")
            and getattr(self.optimizer, "use_property_acquisition", False)
        ):
            props_valid = [properties[i] for i in np.where(valid)[0]]
            self.optimizer.update_with_properties(X[valid], Y[valid], props_valid)
        else:
            self.optimizer.update(X[valid], Y[valid])

    def initialize(self) -> None:
        # Fresh run: start log.jsonl from scratch (subsequent evals append).
        (self.output_dir / "log.jsonl").write_text("")
        Z0 = build_initial_population(self.cfg)
        if self.box is not None:
            Z0 = self.box.clamp_np(Z0)
        png_paths = self._decode_and_save(Z0, iter_idx=0)
        Y0, S0 = self._evaluate_batch(Z0, png_paths, iter_idx=0, desc="seed")
        # Filter out failed seeds before handing to optimizer
        valid = self._surrogate_mask(Y0, S0)
        if not valid.any():
            raise RuntimeError("All seed simulations failed; cannot start optimization.")
        # Per-row property dicts come from the freshly appended records.
        seed_props = [r.properties for r in self.records[-len(Z0):]]
        self._push_to_optimizer(Z0, Y0, valid, properties=seed_props)
        self._save_state(0)

    @staticmethod
    def _cache_manifest_path(cache_path: Path) -> Path | None:
        """Return the adjacent seed-cache manifest path, if present."""
        candidates = [
            cache_path.with_suffix("").with_suffix(".manifest.json"),
            Path(str(cache_path) + ".manifest.json"),
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    @staticmethod
    def _resolve_cached_file(path_like: str | Path) -> Path:
        """Resolve a path stored in a seed-cache manifest (repository-relative)."""
        p = Path(path_like)
        if p.is_file() or p.is_absolute():
            return p
        return Path(os.environ.get("MERIDIAN_ROOT", ".")) / p

    def _load_seed_properties_from_manifest(
        self, cache_path: Path, n_expected: int
    ) -> list[dict | None]:
        """Load cached mechanical-property dicts from the seed-cache manifest.

        The seed ``.npz`` contains only ``Z/Y/success``; the adjacent manifest
        stores paths to per-simulation ``properties.json`` files. MERIDIAN's
        v2 property heads should see those labels during warm-start, but cached
        ``n_grains`` is often ``-1``/missing, so non-positive grain counts are
        deliberately omitted instead of being learned as physical labels.
        """
        props: list[dict | None] = [None] * int(n_expected)
        sidecar_props: list[dict | None] = [None] * int(n_expected)
        sidecar_path = cache_path.with_suffix(".properties.json")
        if sidecar_path.is_file():
            try:
                raw_sidecar = json.loads(sidecar_path.read_text())
                if isinstance(raw_sidecar, list):
                    for i, row in enumerate(raw_sidecar[:n_expected]):
                        if isinstance(row, dict):
                            sidecar_props[i] = row
            except Exception as exc:
                print(f"[seed-props] WARNING: could not read {sidecar_path}: {exc}")
        manifest_path = self._cache_manifest_path(cache_path)
        if manifest_path is None:
            entries = [{} for _ in range(n_expected)]
        else:
            try:
                entries = json.loads(manifest_path.read_text())
            except Exception as exc:
                print(f"[seed-props] WARNING: could not read {manifest_path}: {exc}")
                entries = [{} for _ in range(n_expected)]
        if not isinstance(entries, list):
            entries = [{} for _ in range(n_expected)]

        loaded = 0
        for i in range(n_expected):
            entry = entries[i] if i < len(entries) and isinstance(entries[i], dict) else {}
            raw: dict = {}
            # Prefer the seed-cache sidecar written by encode/rescore scripts:
            # it contains repaired work terms and authoritative n_grains. The
            # manifest's props path often points back to the original DAMASK
            # properties.json, which may predate those derived fields.
            if isinstance(sidecar_props[i], dict):
                raw.update(sidecar_props[i])
            raw_props = entry.get("properties")
            if isinstance(raw_props, dict):
                raw.update(raw_props)
            else:
                raw_props = entry.get("props")
                if isinstance(raw_props, dict):
                    raw.update(raw_props)
                elif isinstance(raw_props, str) and not raw:
                    p = self._resolve_cached_file(raw_props)
                    if p.is_file():
                        try:
                            old_props = json.loads(p.read_text())
                            if isinstance(old_props, dict):
                                raw.update(old_props)
                        except Exception:
                            pass
            # Manifest records repaired n_grains even when the original
            # properties.json does not. Merge it last so seed property heads
            # see the same grain count used to score seed-cache Y.
            if isinstance(entry, dict) and entry.get("n_grains") is not None:
                raw["n_grains"] = entry.get("n_grains")
            if not raw:
                continue

            clean: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(val):
                    continue
                if key == "n_grains" and val <= 0:
                    continue
                clean[str(key)] = val
            if clean:
                props[i] = clean
                loaded += 1
        if loaded:
            print(
                f"[seed-props] loaded cached property labels for "
                f"{loaded}/{n_expected} seed entries from "
                f"{sidecar_path if sidecar_path.is_file() else manifest_path}"
            )
        return props

    def initialize_from_cache(self, cache_path: str | Path) -> None:
        """Load pre-computed seed evaluations from a .npz file.

        The cache must contain 'Z' (N, dim) and 'Y' (N,) arrays plus a
        'success' (N,) boolean mask.  Only valid entries are fed to the
        optimizer, skipping all DAMASK simulations.
        """
        cache_path = Path(cache_path)
        data = np.load(cache_path, allow_pickle=False)
        Z = data["Z"].astype(np.float32)
        Y = data["Y"].astype(np.float32)
        success = data["success"].astype(bool)
        seed_props = self._load_seed_properties_from_manifest(cache_path, len(Z))

        valid = success & np.isfinite(Y)
        if not valid.any():
            raise RuntimeError(f"Seed cache {cache_path} has no valid entries.")

        # Optional cache pre-filter: drop entries whose decoded RVE has
        # n_grains outside [keep_g_min, keep_g_max]. This removes catastrophic
        # speckle / blob outliers from the warm-start so the surrogate trains
        # on a tight, in-band Y distribution. Conservative defaults: keep
        # everything from 1 grain up to 2x the upper band penalty cutoff.
        # Set loop.cache_filter.enabled=false to disable.
        cf = getattr(self.cfg.loop, "cache_filter", None)
        cf_enabled = bool(getattr(cf, "enabled", False)) if cf is not None else False
        if cf_enabled:
            manifest_path = cache_path.with_suffix("").with_suffix(".manifest.json")
            if not manifest_path.is_file():
                manifest_path = Path(str(cache_path) + ".manifest.json")
            if manifest_path.is_file():
                entries = json.loads(manifest_path.read_text())
                ngs = np.array([int(e.get("n_grains", -1)) for e in entries])
                keep_g_min = int(getattr(cf, "keep_g_min", 1))
                keep_g_max = int(getattr(cf, "keep_g_max", 700))
                in_window = (ngs >= keep_g_min) & (ngs <= keep_g_max)
                n_dropped = int((valid & ~in_window).sum())
                valid = valid & in_window
                print(f"[cache-filter] kept n_grains in [{keep_g_min},{keep_g_max}]: "
                      f"{int(valid.sum())} entries, dropped {n_dropped} outliers")
                if not valid.any():
                    raise RuntimeError("Cache filter dropped all entries.")
            else:
                print(f"[cache-filter] WARNING: enabled but no manifest at {manifest_path}; skipping")

        # Calibrate the rejection penalty so it's always strictly worse than
        # the worst valid seed. Without this, V2-style objectives with valid
        # scores in [-5, -3] are gamed by the optimizer because the legacy
        # constant penalty (-1) is *better* than any real sim → mode-collapse.
        # Use robust statistics (5th percentile) for the floor so a handful
        # of cache outliers (e.g. speckle-regime sims with -100 penalty)
        # don't drag the whole calibration to absurd values that would poison
        # the surrogate's output scale.
        y_valid = Y[valid]
        y_min = float(np.percentile(y_valid, 5))
        y_max = float(y_valid.max())
        y_range = max(y_max - y_min, 1.0)
        self._reject_floor = y_min - 0.1 * y_range          # always < typical-bad
        self._reject_span = 2.0 * y_range                    # full-collapse penalty
        print(f"[reject-penalty] valid Y p5..max=[{y_min:.3f}, {y_max:.3f}]  "
              f"floor={self._reject_floor:.3f}  span={self._reject_span:.3f}  "
              f"(rejected sims will score in [{self._reject_floor - self._reject_span:.3f}, "
              f"{self._reject_floor:.3f}])")

        self._push_to_optimizer(Z, Y, valid, properties=seed_props)
        self._n_seed = len(self.optimizer.Y)

        # Populate records for the log + persist them to log.jsonl up-front
        # (subsequent post-seed evaluations append in _append_log).
        log_path = self.output_dir / "log.jsonl"
        with log_path.open("w") as f:
            for i in range(len(Z)):
                rec = IterationRecord(
                    iteration=0, z=Z[i],
                    objective=float(Y[i]) if valid[i] else float("-inf"),
                    properties=seed_props[i] if valid[i] else None,
                    success=bool(valid[i]),
                    error=None if valid[i] else "cached_failure",
                    elapsed_s=0.0,
                )
                self.records.append(rec)
                f.write(json.dumps({
                    "iter": rec.iteration, "batch_idx": i,
                    "objective": rec.objective,
                    "success": rec.success, "error": rec.error,
                    "elapsed_s": rec.elapsed_s, "props": rec.properties,
                }) + "\n")
        self._save_state(0)
        n_valid = int(valid.sum())
        print(f"Loaded seed cache: {n_valid}/{len(Z)} valid entries from {cache_path}")

        # Optional explicit anchor for the optimizer (DANTE NTE root /
        # MERIDIAN trust-region center). Picks one of:
        #   loop.initial_seed.sim_id    -> match against the cache manifest
        #   loop.initial_seed.index     -> integer index into Z
        #   loop.initial_seed.npz       -> path to a .npz with key 'z' (1-D, dim)
        seed_cfg = getattr(self.cfg.loop, "initial_seed", None)
        if seed_cfg is not None and hasattr(self.optimizer, "set_initial_seed"):
            # Optional ``apply_to`` whitelist: skip the anchor override for
            # optimizers whose internal geometry can't faithfully represent
            # an arbitrary anchor (e.g. BAxUS subspace projection collapses
            # ||z|| when the anchor lies far from the bounds-midpoint in
            # wide latent dims). When omitted, applies to all optimizers.
            apply_to = getattr(seed_cfg, "apply_to", None)
            opt_name = str(getattr(self.optimizer, "name", "")).lower()
            if apply_to is not None and opt_name not in {str(x).lower() for x in apply_to}:
                print(f"[{opt_name}] initial seed override skipped "
                      f"(apply_to={list(apply_to)})")
                return
            z_init = self._resolve_initial_seed(seed_cfg, Z, cache_path)
            if z_init is not None:
                if self.box is not None:
                    z_init = self.box.clamp_np(z_init[None])[0]
                self.optimizer.set_initial_seed(z_init)
                print(f"[{self.optimizer.name}] initial seed overridden via "
                      f"loop.initial_seed (|z|={float(np.linalg.norm(z_init)):.2f})")

    def _resolve_initial_seed(self, seed_cfg, Z: np.ndarray,
                              cache_path: Path) -> np.ndarray | None:
        """Resolve loop.initial_seed config to a concrete latent vector."""
        # 1) explicit .npz path
        npz_path = getattr(seed_cfg, "npz", None)
        if npz_path:
            d = np.load(npz_path, allow_pickle=False)
            return d["z"].astype(np.float32).reshape(-1)
        # 2) explicit integer index into the seed cache
        idx = getattr(seed_cfg, "index", None)
        if idx is not None:
            return Z[int(idx)].astype(np.float32)
        # 3) sim_id lookup via the manifest sidecar of the seed cache
        sim_id = getattr(seed_cfg, "sim_id", None)
        if sim_id is not None:
            manifest_path = cache_path.with_suffix("").with_suffix(".manifest.json")
            if not manifest_path.is_file():
                manifest_path = Path(str(cache_path) + ".manifest.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"loop.initial_seed.sim_id requires manifest at {manifest_path}")
            entries = json.loads(manifest_path.read_text())
            sim_id = str(sim_id)
            for i, e in enumerate(entries):
                if str(e.get("sim_id")) == sim_id or str(e.get("id")) == sim_id \
                        or sim_id in str(e.get("props", "")):
                    return Z[i].astype(np.float32)
            raise ValueError(f"sim_id={sim_id!r} not found in {manifest_path}")
        # 4) data-driven criterion: pick a cache entry by rule
        #    criterion: "top_y"          -> argmax(Y) over valid entries
        #    criterion: "band_top_y"     -> argmax(Y) restricted to entries
        #                                   whose n_grains lies in
        #                                   [band_lo, band_hi]; falls back to
        #                                   global top_y if the band is empty.
        criterion = getattr(seed_cfg, "criterion", None)
        if criterion:
            data = np.load(cache_path)
            Y = data["Y"].astype(np.float32)
            success = data["success"].astype(bool) if "success" in data.files \
                else np.isfinite(Y)
            valid = success & np.isfinite(Y)
            if not valid.any():
                return None
            band_lo = int(getattr(seed_cfg, "band_lo", 80))
            band_hi = int(getattr(seed_cfg, "band_hi", 350))
            manifest_path = cache_path.with_suffix("").with_suffix(".manifest.json")
            if not manifest_path.is_file():
                manifest_path = Path(str(cache_path) + ".manifest.json")
            chosen_idx: int | None = None
            if criterion == "band_top_y" and manifest_path.is_file():
                entries = json.loads(manifest_path.read_text())
                ngs = np.array([int(e.get("n_grains", -1)) for e in entries])
                in_band = valid & (ngs >= band_lo) & (ngs < band_hi)
                if in_band.any():
                    pool = np.where(in_band)[0]
                    chosen_idx = int(pool[np.argmax(Y[pool])])
                    print(f"[initial_seed] criterion=band_top_y "
                          f"[{band_lo},{band_hi}): {int(in_band.sum())} candidates, "
                          f"chose idx={chosen_idx} (sim_id="
                          f"{entries[chosen_idx].get('sim_id')}, "
                          f"n_grains={ngs[chosen_idx]}, Y={Y[chosen_idx]:.3f})")
            if chosen_idx is None:  # top_y, or band fell through
                pool = np.where(valid)[0]
                chosen_idx = int(pool[np.argmax(Y[pool])])
                print(f"[initial_seed] criterion=top_y: chose idx={chosen_idx} "
                      f"(Y={Y[chosen_idx]:.3f})")
            return Z[chosen_idx].astype(np.float32)
        return None

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> int:
        """Restore optimizer state from a checkpoint .npz and return the
        iteration number to resume from.

        The checkpoint must contain 'X' (N, dim) and 'Y' (N,) arrays.
        Returns the iteration number extracted from the filename.
        """
        checkpoint_path = Path(checkpoint_path)
        data = np.load(checkpoint_path, allow_pickle=False)
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.float32)

        if len(X) == 0:
            raise RuntimeError(f"Checkpoint {checkpoint_path} is empty.")

        # Extract iteration number from filename (state_iter0015.npz -> 15)
        stem = checkpoint_path.stem  # e.g. "state_iter0015"
        try:
            resume_iter = int(stem.split("iter")[-1])
        except (ValueError, IndexError):
            resume_iter = 0

        # Reload log if it exists.  For MERIDIAN's property acquisition, also
        # replay the per-row property labels that were originally fed to the
        # optimizer.  Checkpoints only store X/Y, so calling update(X, Y) here
        # would silently resume with all property-head labels missing after a
        # crash.  That makes the target-aware acquisition behave like vanilla
        # qLogNEI for several iterations, which is especially damaging in V2.
        replay_props: list[dict | None] = []
        replay_y: list[float] = []
        n_seed_replay = 0
        log_path = self.output_dir / "log.jsonl"
        if log_path.is_file():
            import json as _json
            for line in log_path.read_text().strip().splitlines():
                entry = _json.loads(line)
                y_val = float(entry["objective"])
                success = bool(entry["success"])
                self.records.append(IterationRecord(
                    iteration=entry["iter"], z=np.zeros(self.cfg.latent.dim),
                    objective=y_val, properties=entry.get("props"),
                    success=success, error=entry.get("error"),
                    elapsed_s=entry.get("elapsed_s", 0.0),
                ))
                if not np.isfinite(y_val):
                    continue
                if getattr(self.optimizer, "wants_reject_feedback", False):
                    keep = True
                elif bool(getattr(self.cfg.loop, "mask_failed_from_surrogate", True)):
                    keep = success
                else:
                    keep = True
                if keep:
                    replay_y.append(y_val)
                    replay_props.append(entry.get("props"))
                    if int(entry["iter"]) == 0:
                        n_seed_replay += 1

        if (
            hasattr(self.optimizer, "update_with_properties")
            and getattr(self.optimizer, "use_property_acquisition", False)
            and replay_props
        ):
            if len(replay_props) != len(X):
                print(
                    f"[resume] WARNING: log replay rows ({len(replay_props)}) "
                    f"!= checkpoint rows ({len(X)}); padding/truncating properties"
                )
                if len(replay_props) < len(X):
                    replay_props = replay_props + [None] * (len(X) - len(replay_props))
                else:
                    replay_props = replay_props[:len(X)]
            elif replay_y and not np.allclose(
                np.asarray(replay_y, dtype=np.float32), Y, rtol=1e-4, atol=1e-5
            ):
                print("[resume] WARNING: log objective sequence differs from checkpoint Y; replaying properties by row order")
            self.optimizer.update_with_properties(X, Y, replay_props)
        else:
            self.optimizer.update(X, Y)

        n = len(X)
        self._n_seed = n_seed_replay
        print(f"Resumed from checkpoint: {n} observations, iteration {resume_iter}")
        return resume_iter

    def step(self, iter_idx: int) -> None:
        Z = self.optimizer.suggest()
        if self.box is not None:
            Z = self.box.clamp_np(Z)
        png_paths = self._decode_and_save(Z, iter_idx=iter_idx)
        Y, S = self._evaluate_batch(Z, png_paths, iter_idx=iter_idx, desc=f"iter{iter_idx}")
        valid = self._surrogate_mask(Y, S)
        if valid.any():
            step_props = [r.properties for r in self.records[-len(Z):]]
            self._push_to_optimizer(Z, Y, valid, properties=step_props)

    def run(self, seed_cache: str | Path | None = None,
            resume_checkpoint: str | Path | None = None) -> dict:
        start_iter = 1
        if resume_checkpoint is not None:
            start_iter = self.resume_from_checkpoint(resume_checkpoint) + 1
        elif seed_cache is not None:
            self.initialize_from_cache(seed_cache)
        else:
            self.initialize()
        n_iter = int(self.cfg.loop.n_iterations)
        ck_every = int(self.cfg.loop.checkpoint_every)
        for it in range(start_iter, n_iter + 1):
            self.step(it)
            if it % ck_every == 0:
                self._save_state(it)
        self._save_state(n_iter)
        z_best, y_best = self._best_proposed()
        z_any, y_any = self.optimizer.best()
        return {
            "best_objective": float(y_best),
            "best_z": z_best.tolist(),
            "best_objective_incl_seeds": float(y_any),
            "n_seed": int(self._n_seed),
        }

    def _best_proposed(self) -> tuple[np.ndarray, float]:
        """Best over decoder-proposed points only; seed-cache rows are excluded."""
        Y = self.optimizer.Y[self._n_seed:]
        if len(Y) == 0:
            return self.optimizer.best()
        i = int(np.argmax(Y)) + self._n_seed
        return self.optimizer.X[i].copy(), float(self.optimizer.Y[i])

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _save_state(self, it: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.output_dir / f"state_iter{it:04d}.npz",
            X=self.optimizer.X, Y=self.optimizer.Y,
        )
        # log.jsonl is streamed per-evaluation in _append_log(); nothing to do here.
