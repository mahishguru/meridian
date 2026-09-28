"""MERIDIAN outer optimizer.

A surrogate-driven trust-region optimiser that operates in a low-dimensional
subspace of the latent manifold, with a feasibility-aware acquisition and a
diversity-promoting batch selection rule.

See ``../../theory.md`` §7 for the full derivation. Single committed
configuration; no internal ablations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from meridian.optimizers import BaseOptimizer
from meridian.optimizers.meridian.batch import acquisition_scores, greedy_dpp_select
from meridian.optimizers.meridian.dkl import DKLSurrogate
from meridian.optimizers.meridian.gradient_seed import (
    GradientSeedConfig,
    gradient_seed_candidate,
)
from meridian.optimizers.meridian.mcts_restart import (
    MCTSRestartConfig,
    mcts_restart_batch,
)
from meridian.optimizers.meridian.property_acquisition import PropertyTargetAcquisition
from meridian.optimizers.meridian.subspace import SubspaceConfig, SubspaceManager
from meridian.optimizers.meridian.trust_region import TRConfig, TrustRegion


class MeridianOptimizer(BaseOptimizer):
    """Manifold-Embedded Robust Inverse Design via Iterative Acquisition Networks."""

    name = "meridian"

    def __init__(
        self,
        dim: int,
        bounds: tuple[float, float],
        batch_size: int,
        meridian_cfg,
        seeds_dir: str | None = None,
        class_key: str | None = None,
        device: str = "cuda",
        objective_cfg=None,
    ) -> None:
        super().__init__(dim, bounds, batch_size)
        self.cfg = meridian_cfg
        self._objective_cfg = objective_cfg
        self.device = device

        # ---- Surrogate (DKL + multi-task feasibility head) ----
        self.surrogate = DKLSurrogate(
            in_dim=dim,
            hidden=list(meridian_cfg.feature_hidden),
            feature_dim=int(meridian_cfg.feature_dim),
            lr=float(meridian_cfg.surrogate_lr),
            epochs=int(meridian_cfg.surrogate_epochs),
            weight_decay=float(meridian_cfg.surrogate_weight_decay),
            device=device,
        )

        # ---- Subspace manager (PCA warm-start → active subspace) ----
        self.subspace = SubspaceManager(
            SubspaceConfig(
                energy=float(meridian_cfg.subspace_energy),
                k_min=int(meridian_cfg.subspace_k_min),
                k_max=int(meridian_cfg.subspace_k_max),
                pca_warmstart_rounds=int(meridian_cfg.pca_warmstart_rounds),
                w_floor=float(getattr(meridian_cfg, "w_floor", 0.1)),
                seeds_dir=seeds_dir,
                class_key=class_key,
            )
        )
        self.subspace.warmstart_from_pool(dim=dim)

        # ---- Trust region in subspace coordinates ----
        self.trust = TrustRegion(
            TRConfig(
                L_init=float(meridian_cfg.L_init),
                L_min=float(meridian_cfg.L_min),
                L_max=float(meridian_cfg.L_max),
                success_tolerance=int(meridian_cfg.success_tolerance),
                failure_tolerance_factor=float(meridian_cfg.failure_tolerance_factor),
                failure_tolerance_max=int(getattr(meridian_cfg, "failure_tolerance_max", 8)),
                max_restarts=int(meridian_cfg.max_restarts),
            )
        )

        # ---- State ----
        self.feasible_mask: np.ndarray = np.empty((0,), dtype=bool)
        self._round = 0
        self._sobol = SobolEngine(dim, scramble=True)
        self.n_candidates = int(meridian_cfg.sobol_cloud)
        self.dpp_pool = int(meridian_cfg.dpp_pool)
        self._perturb_dims_target = int(
            getattr(meridian_cfg, "perturb_dims_target", 20)
        )
        # Number of "diffuse" candidates injected into the Sobol pool: full
        # 512-D isotropic perturbations a la DANTE. Acts as a structural
        # safety net for landscapes (e.g. AZ31_v1 fmdit) where the rare
        # high-Y pocket is reached by full-dim diffuse moves rather than
        # sparse coordinate flips. Set to 0 to disable.
        self._diffuse_explore_frac = float(
            getattr(meridian_cfg, "diffuse_explore_frac", 0.25)
        )
        self._diffuse_sigma_scale = float(
            getattr(meridian_cfg, "diffuse_sigma_scale", 0.5)
        )
        self._needs_refit = True
        self._seeds_dir = seeds_dir
        self._class_key = class_key
        self._restart_z: np.ndarray | None = None  # set on trust-region restart

        # ---- Adaptive shell projection (v4) ----
        # The original implementation pinned every candidate's L2 norm to a
        # static [shell_radius_min, shell_radius_max] window read from yaml.
        # That window was hand-tuned to the seed-cache median for each
        # (decoder, latent-dim) pair, which silently broke whenever:
        #   (a) the decoder/dim changed (e.g. fmdit-768 anchor at 27.55,
        #       static window [22.0, 22.2] -> every candidate rescaled to a
        #       19% off-manifold sphere, no improvement past anchor);
        #   (b) high-Y pockets exist OUTSIDE the seed-cache shell (e.g.
        #       v1_vitdit TuRBO winner at ||z||=24.4 vs MERIDIAN's 22.8 cap).
        # Adaptive shell fixes both: derive [mu - k*sigma, mu + k*sigma]
        # from the empirical norms of feasible X every round (warm-started
        # by the seed cache) and let a small fraction of candidates skip
        # projection entirely so the shell can ever widen.
        self._adaptive_shell = bool(
            getattr(meridian_cfg, "adaptive_shell", True)
        )
        # Half-width of the data shell in standard deviations of feasible
        # ||X||. k=2 is the empirical 95% band; we use 3 to leave headroom
        # for the optimizer to push slightly outside the warm-start cloud.
        self._shell_k_sigma = float(
            getattr(meridian_cfg, "shell_k_sigma", 3.0)
        )
        # Fraction of the Sobol cloud (and the MCTS restart batch) that
        # bypass shell projection entirely. Without this the shell is a
        # one-way ratchet: rescaled candidates can never enlarge the
        # feasible-||X|| support, so the shell cannot widen even if a true
        # off-shell optimum exists. 15% gives ~600 unprojected candidates
        # per round to test that hypothesis without disturbing the bulk.
        self._shell_exempt_frac = float(
            getattr(meridian_cfg, "shell_exempt_frac", 0.15)
        )
        # Robust guardrail for adaptive shells. A few feasible but visibly
        # off-manifold decoded samples can otherwise blow the shell out from
        # e.g. Vit-DiT's real-data band near ||z||≈22.6 to [17, 30], after
        # which the trust-region prior stops meaning "manifold tangent".
        # The cap keeps controlled shell widening while still permitting the
        # useful lower-/upper-norm winners seen in the wide-d sweeps.
        self._shell_max_width = float(
            getattr(meridian_cfg, "shell_max_width", 3.0)
        )

        # ---- A2/A3: property-aware surrogate + acquisition (opt-in) ----
        self.use_property_acquisition = bool(
            getattr(meridian_cfg, "use_property_acquisition", False)
        )
        self.property_keys: tuple[str, ...] = tuple(
            getattr(meridian_cfg, "property_keys",
                    ("sigma_y", "sigma_u", "n", "K", "n_grains"))
        )
        self._property_acq: PropertyTargetAcquisition | None = None
        if self.use_property_acquisition:
            obj_cfg = self._objective_cfg
            targets = (
                getattr(meridian_cfg, "targets", None)
                or (dict(obj_cfg.targets) if obj_cfg is not None and hasattr(obj_cfg, "targets") else None)
            )
            weights = (
                getattr(meridian_cfg, "weights", None)
                or (dict(obj_cfg.weights) if obj_cfg is not None and hasattr(obj_cfg, "weights") else None)
            )
            objective_kind = str(getattr(obj_cfg, "name", "target") if obj_cfg is not None else "target").lower()
            is_v1_objective = objective_kind in {"v1", "toughness", "v1_toughness"}
            if (targets is not None and weights is not None) or is_v1_objective:
                objective_params = {}
                if is_v1_objective and obj_cfg is not None:
                    for key in (
                        "sigma_y_ref", "toughness_ref", "alpha", "beta",
                        "sigma_u_ref", "gamma", "work_key", "n_min", "lambda_n",
                    ):
                        if hasattr(obj_cfg, key):
                            objective_params[key] = getattr(obj_cfg, key)
                self._property_acq = PropertyTargetAcquisition(
                    targets=dict(targets or {}),
                    weights=dict(weights or {}),
                    property_keys=self.property_keys,
                    g_min=int(getattr(meridian_cfg, "property_acq_g_min",
                              getattr(obj_cfg, "g_min", 0) if obj_cfg else 0)),
                    lambda_g=float(getattr(meridian_cfg, "property_acq_lambda_g",
                                   getattr(obj_cfg, "lambda_g", 0.0) if obj_cfg else 0.0)),
                    g_max=int(getattr(meridian_cfg, "property_acq_g_max",
                              getattr(obj_cfg, "g_max", 0) if obj_cfg else 0)),
                    lambda_g_hi=float(getattr(meridian_cfg, "property_acq_lambda_g_hi",
                                      getattr(obj_cfg, "lambda_g_hi", 0.0) if obj_cfg else 0.0)),
                    n_mc_samples=int(getattr(meridian_cfg, "property_acq_mc_samples", 64)),
                    sigma_floor_frac=float(
                        getattr(meridian_cfg, "property_sigma_floor_frac", 0.0)
                    ),
                    reachable_prior=getattr(
                        meridian_cfg, "property_reachable_prior", None
                    ),
                    objective_kind="v1_toughness" if is_v1_objective else "target",
                    objective_params=objective_params,
                )
            else:
                # Misconfigured: heads will train but acquisition can't run.
                # Fall back gracefully (suggest() guards on _property_acq is not None).
                self.use_property_acquisition = False
        # Per-row property matrix tracked alongside (X, Y); None where missing.
        self.P: np.ndarray | None = None  # shape (N, n_props), float32, NaN if missing

        # ---- MERIDIAN-v2 upgrades (A1+A4+A5+A6) ----
        # All knobs are read with defaults so older yamls keep working.
        self.use_meridian_v2: bool = bool(
            getattr(meridian_cfg, "use_meridian_v2", False)
        )
        self.plateau_K: int = int(getattr(meridian_cfg, "plateau_K", 4))
        self.plateau_max_restarts: int = int(
            getattr(
                meridian_cfg,
                "plateau_max_restarts",
                getattr(meridian_cfg, "max_restarts", 3),
            )
        )
        self.restart_strategy: str = str(
            getattr(meridian_cfg, "restart_strategy", "mcts")
        ).lower()
        self.head_reset_every: int = int(
            getattr(meridian_cfg, "head_reset_every", 10)
        )
        self.gradient_seed_enabled: bool = bool(
            getattr(meridian_cfg, "gradient_seed_enabled", self.use_meridian_v2)
        )
        self.dpp_z_kernel_weight: float = float(
            getattr(meridian_cfg, "dpp_z_kernel_weight", 0.0)
        )
        self.dpp_z_kernel_lengthscale: float = float(
            getattr(meridian_cfg, "dpp_z_kernel_lengthscale", 1.0)
        )
        # ---- MERIDIAN-v5 knobs ----
        # Anchor strategy: "current_best" (v4) or "topk_centroid" (v5).
        # The latter averages the top-K feasible points (re-projected to the
        # shell midpoint), removing the gravity-well of a single isolated peak
        # and giving the optimizer room to climb adjacent ridges.
        self._z_center_strategy: str = str(
            getattr(meridian_cfg, "z_center_strategy", "current_best")
        ).lower()
        self._z_center_topk: int = int(
            getattr(meridian_cfg, "z_center_topk", 5)
        )
        # Clean-benchmark guard against seed-cache gravity. The seed rows are
        # still used for surrogate training, but ``fresh_after_seed`` switches
        # center/plateau/incumbent calculations to fresh evaluations once any
        # feasible fresh rows exist.
        self._incumbent_policy: str = str(
            getattr(meridian_cfg, "incumbent_policy", "all")
        ).lower()
        # Acquisition threshold policy.  Keeping ``incumbent_policy=all`` is
        # important for centre/plateau bookkeeping: the optimizer should know
        # whether a fresh candidate truly beats the seed-cache incumbent.
        # For property-space EI, however, using a very strong seed incumbent as
        # ``y_best`` makes EI numerically collapse when the fresh landscape has
        # not yet matched the cache.  ``fresh_after_seed`` lets property-EI rank
        # candidates against the best fresh point once any fresh labels exist,
        # while the global incumbent/centre still remains seed-aware.
        self._property_acq_incumbent_policy: str = str(
            getattr(
                meridian_cfg,
                "property_acq_incumbent_policy",
                self._incumbent_policy,
            )
        ).lower()
        self._seed_rows: int = int(getattr(meridian_cfg, "seed_rows", 100))
        # Late-iter polish mode: after this iter, zero out diffuse/exempt/dpp_z
        # and shrink L to a small fixed cap. Mirrors DANTE's late-iter regime.
        # Set to <=0 to disable.
        self._polish_after_iter: int = int(
            getattr(meridian_cfg, "polish_after_iter", 0)
        )
        # Portfolio batch policy (v7): instead of letting one acquisition/DPP
        # belief choose all q points, reserve slots for complementary roles:
        # exploit, DANTE-style diffuse scout, uncertainty scout, and property
        # gradient seed. This is the main antidote to seed-cache plateaus.
        self._portfolio_batch: bool = bool(
            getattr(meridian_cfg, "portfolio_batch", False)
        )
        self._portfolio_uncertainty_mode: str = str(
            getattr(meridian_cfg, "portfolio_uncertainty_mode", "ucb")
        ).lower()
        self._portfolio_ucb_beta: float = float(
            getattr(meridian_cfg, "portfolio_ucb_beta", 1.0)
        )
        # Plateau bookkeeping
        self._best_y_seen: float = -float("inf")
        self._iters_since_improvement: int = 0
        self._restart_count_v2: int = 0  # A1/A6 restarts (independent of TR)
        self._fits_done: int = 0  # for periodic head reset
        self._last_restart_strategy: str | None = None  # diagnostic

    # ----------------------------------------------------------- explicit anchor
    def set_initial_seed(self, z: np.ndarray) -> None:
        """Pin the trust-region center for the first round to ``z``.

        Used to start MERIDIAN from the same anchor microstructure as DANTE
        (e.g. sim_449) for fair comparison. Implemented via the existing
        restart-z slot, which ``_current_best()`` consumes on its next call.
        """
        self._restart_z = np.asarray(z, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------ data
    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        """Override BaseOptimizer.update to also track feasibility."""
        X_new = np.atleast_2d(X_new).astype(np.float32)
        Y_new = np.atleast_1d(Y_new).astype(np.float32)
        feas_new = np.isfinite(Y_new)
        self.X = np.vstack([self.X, X_new])
        self.Y = np.concatenate([self.Y, np.where(feas_new, Y_new, 0.0)])
        self.feasible_mask = np.concatenate([self.feasible_mask, feas_new])
        # Pad property matrix with NaNs for the new rows so columns stay aligned
        if self.use_property_acquisition:
            n_props = len(self.property_keys)
            new_p = np.full((len(X_new), n_props), np.nan, dtype=np.float32)
            self.P = new_p if self.P is None else np.vstack([self.P, new_p])
        # Trust-region update uses only the feasible subset of the new batch
        feas_y = Y_new[feas_new]
        if len(feas_y):
            self.trust.update(feas_y, k=self.subspace.effective_k, batch_size=self.batch_size)
        self._needs_refit = True

    def update_with_properties(
        self,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        properties: list[dict | None],
    ) -> None:
        """Same as :meth:`update` but also stores a row of measured properties.

        The loop calls this when the optimizer advertises ``use_property_acquisition``.
        Each entry of ``properties`` is the per-evaluation property dict
        (or ``None`` for failed/rejected sims). Missing keys become ``NaN``
        in the property matrix, which is later masked by the surrogate.
        """
        # Append X/Y/feas exactly like update() so the bookkeeping stays
        # in sync, but DO NOT pre-fill P with NaN — we will overwrite it
        # with the freshly supplied dicts below.
        X_new = np.atleast_2d(X_new).astype(np.float32)
        Y_new = np.atleast_1d(Y_new).astype(np.float32)
        feas_new = np.isfinite(Y_new)
        self.X = np.vstack([self.X, X_new])
        self.Y = np.concatenate([self.Y, np.where(feas_new, Y_new, 0.0)])
        self.feasible_mask = np.concatenate([self.feasible_mask, feas_new])

        n_props = len(self.property_keys)
        new_p = np.full((len(X_new), n_props), np.nan, dtype=np.float32)
        for i, pdict in enumerate(properties):
            if not isinstance(pdict, dict):
                continue
            for j, k in enumerate(self.property_keys):
                v = pdict.get(k)
                if v is not None:
                    try:
                        new_p[i, j] = float(v)
                    except (TypeError, ValueError):
                        pass
        self.P = new_p if self.P is None else np.vstack([self.P, new_p])

        feas_y = Y_new[feas_new]
        if len(feas_y):
            self.trust.update(feas_y, k=self.subspace.effective_k, batch_size=self.batch_size)
        self._needs_refit = True

    # ------------------------------------------------------------- internals
    def _shell_bounds(self) -> tuple[float, float]:
        """Return the [r_min, r_max] window for the current round.

        Adaptive mode: ``[mu - k*sigma, mu + k*sigma]`` over feasible
        ||X|| norms (warm-started from the seed cache). When fewer than 5
        feasible observations are available we fall back to the static
        config values, which keeps behaviour identical to v3 during the
        very first iteration before the seed cache lands.

        Static mode (``adaptive_shell=false``): returns the static
        ``shell_radius_min/max`` from yaml.
        """
        cfg_min = float(getattr(self.cfg, "shell_radius_min", 0.0))
        cfg_max = float(getattr(self.cfg, "shell_radius_max", 0.0))
        if not self._adaptive_shell:
            return cfg_min, cfg_max
        feas = self.feasible_mask
        if feas is None or not feas.any() or int(feas.sum()) < 5:
            return cfg_min, cfg_max
        norms = np.linalg.norm(self.X[feas], axis=1)
        mu, sd = float(norms.mean()), float(norms.std() + 1e-6)
        r_min = max(0.0, mu - self._shell_k_sigma * sd)
        r_max = mu + self._shell_k_sigma * sd
        if self._shell_max_width > 0.0 and (r_max - r_min) > self._shell_max_width:
            # Re-center the capped band on the median, not the mean, so a
            # small number of off-shell probes cannot drag the shell center.
            mid = float(np.median(norms))
            half = 0.5 * self._shell_max_width
            r_min = max(0.0, mid - half)
            r_max = mid + half
        return r_min, r_max

    @staticmethod
    def _apply_shell_with_exemption(
        Z: np.ndarray,
        r_min: float,
        r_max: float,
        exempt_frac: float,
        low: float,
        high: float,
    ) -> np.ndarray:
        """Project ``(1 - exempt_frac)`` of rows onto the [r_min, r_max] shell.

        Disabled when ``r_min <= 0`` or ``r_max < r_min``. Exempt rows are
        left at their natural norm (still box-clipped to [low, high]) so
        they can probe regions outside the current shell estimate; if
        evaluations succeed there, the next round's adaptive bounds will
        widen automatically.
        """
        if r_min <= 0.0 or r_max < r_min:
            return Z
        n = len(Z)
        n_exempt = int(round(exempt_frac * n)) if exempt_frac > 0 else 0
        # Always project the bulk; leave the last n_exempt rows alone.
        # The Sobol cloud is contiguous before the diffuse-explore tail,
        # so we project the head and exempt the tail uniformly across
        # both the Sobol and diffuse populations (identical strata).
        head = Z[:n - n_exempt] if n_exempt < n else Z[:0]
        tail = Z[n - n_exempt:] if n_exempt > 0 else Z[:0]
        if len(head):
            r = np.linalg.norm(head, axis=1)
            r = np.where(r > 1e-8, r, 1e-8)
            target_r = np.clip(r, r_min, r_max)
            head = head * (target_r / r)[:, None]
            head = np.clip(head, low, high)
        if len(tail):
            tail = np.clip(tail, low, high)
        return np.vstack([head, tail]) if (len(head) and len(tail)) else (
            head if len(head) else tail
        )

    def _restart_center(self) -> np.ndarray:
        """Cold-start center from the class pool (if available) else random."""
        if self._seeds_dir and self._class_key:
            pool_path = Path(self._seeds_dir) / f"{self._class_key}.npy"
            if pool_path.is_file():
                pool = np.load(pool_path).astype(np.float32)
                idx = np.random.randint(0, len(pool))
                return pool[idx]
        return np.random.uniform(self.low, self.high, size=self.dim).astype(np.float32)

    def _current_best(self) -> np.ndarray:
        # After a restart, use the fresh class-pool sample as center
        if self._restart_z is not None:
            z = self._restart_z
            self._restart_z = None
            return z
        if self.feasible_mask.any():
            feas_idx = self._eligible_incumbent_indices()
            # ---- v5: top-K centroid anchor (re-projected to shell midpoint) ----
            if self._z_center_strategy == "topk_centroid" and len(feas_idx) >= 2:
                k = max(1, min(int(self._z_center_topk), len(feas_idx)))
                order = np.argsort(self.Y[feas_idx])[-k:]
                top_idx = feas_idx[order]
                z = self.X[top_idx].mean(axis=0).astype(np.float32)
                # Re-project to shell midpoint so the centroid stays on-manifold
                # (averaging on a sphere shrinks norm towards origin otherwise).
                r_min, r_max = self._shell_bounds()
                if r_max > r_min > 0.0:
                    target = 0.5 * (r_min + r_max)
                    n = float(np.linalg.norm(z))
                    if n > 1e-8:
                        z = (z * (target / n)).astype(np.float32)
                return z
            return self.X[feas_idx[int(np.argmax(self.Y[feas_idx]))]]
        return self.X[int(np.argmax(self.Y))] if len(self.X) else self._restart_center()

    def _eligible_incumbent_indices(self) -> np.ndarray:
        """Feasible rows allowed to define the incumbent/center."""
        feas_idx = np.where(self.feasible_mask)[0]
        if (
            self._incumbent_policy == "fresh_after_seed"
            and len(self.X) > self._seed_rows
        ):
            fresh = feas_idx[feas_idx >= self._seed_rows]
            if len(fresh):
                return fresh
        return feas_idx

    def _incumbent_best_y(self) -> float:
        if not self.feasible_mask.any():
            return float("-inf")
        idx = self._eligible_incumbent_indices()
        if len(idx) == 0:
            return float("-inf")
        return float(self.Y[idx].max())

    def _property_acq_best_y(self) -> float:
        """Incumbent threshold used only for property-space EI.

        This may intentionally differ from ``_incumbent_best_y``.  The seed
        cache is still part of the optimizer state, centre selection, result
        incumbent, and plateau detection, but property-EI can be thresholded
        against fresh observations to avoid zero-EI collapse under a strong
        cache incumbent.
        """
        if not self.feasible_mask.any():
            return float("-inf")
        if (
            self._property_acq_incumbent_policy == "fresh_after_seed"
            and len(self.X) > self._seed_rows
        ):
            feas_idx = np.where(self.feasible_mask)[0]
            fresh = feas_idx[feas_idx >= self._seed_rows]
            if len(fresh):
                return float(self.Y[fresh].max())
        return self._incumbent_best_y()

    def _refit_surrogate(self) -> None:
        if not self.feasible_mask.any():
            return
        # A5: periodically reset PropertyHeads linear weights so the heads
        # don't lock into a degenerate sigma->floor regime on a saturated
        # cache. Trunk + classifier are kept (warm-start preserves features).
        if (
            self.use_meridian_v2
            and self.head_reset_every > 0
            and self._fits_done > 0
            and self._fits_done % self.head_reset_every == 0
        ):
            self.surrogate.reset_property_heads()
        if self.use_property_acquisition and self.P is not None:
            self.surrogate.fit(
                self.X, self.Y, self.feasible_mask,
                P_all=self.P, property_keys=list(self.property_keys),
            )
        else:
            self.surrogate.fit(self.X, self.Y, self.feasible_mask)
        self._fits_done += 1
        # Mitigation for the slow MERIDIAN-v2 GPU memory leak observed during
        # the 16-cell sweep (single MERIDIAN process grew from ~14 GB to ~83
        # GB over 10 hours -> caused OOM for any co-located job). Each fit
        # rebuilds the GP/MLP and the previous tensors should be GC-able once
        # the new state object replaces the old one. Forcing a cache release
        # here pushes PyTorch to actually return the memory to the GPU pool.
        try:
            import torch as _torch  # local import to avoid altering top-of-file imports
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    def _maybe_refit_subspace(self) -> None:
        if self._round < self.subspace.cfg.pca_warmstart_rounds:
            return
        if not self.feasible_mask.any() or self.surrogate.state is None or self.surrogate.state.gp is None:
            return
        try:
            grads = self.surrogate.grad_mean(self.X[self.feasible_mask])
            self.subspace.maybe_refit(self._round, grads)
        except Exception:
            pass

    # ----------------------------------------------------------------- suggest
    def suggest(self) -> np.ndarray:
        if len(self.X) == 0:
            raise RuntimeError("MERIDIAN requires seed data before suggesting.")
        self._round += 1

        # ---- v5: explore->polish schedule (gated by polish_after_iter) ----
        # In polish mode the optimizer becomes DANTE-like: zero diffuse/exempt/
        # dpp-z, L capped at 0.1, so the batch concentrates on the best basin.
        in_polish = (
            self._polish_after_iter > 0
            and self._round > self._polish_after_iter
        )
        if in_polish:
            diffuse_frac = 0.0
            exempt_frac = 0.0
            dpp_z_w = 0.0
            L_eff = min(float(self.trust.L), 0.1)
        else:
            diffuse_frac = self._diffuse_explore_frac
            exempt_frac = self._shell_exempt_frac
            dpp_z_w = self.dpp_z_kernel_weight
            L_eff = float(self.trust.L)

        # ---- v5: per-iter diagnostic (one line per suggest()) ----
        try:
            _r_min, _r_max = self._shell_bounds()
        except Exception:
            _r_min, _r_max = 0.0, 0.0
        print(
            f"[meridian] iter={self._round} L={float(self.trust.L):.3f} "
            f"L_eff={L_eff:.3f} plateau={self._iters_since_improvement}/{self.plateau_K} "
            f"y*={self._best_y_seen:+.4f} restarts={self._restart_count_v2} "
            f"shell=[{_r_min:.2f},{_r_max:.2f}] polish={int(in_polish)}",
            flush=True,
        )

        # ---- A1: Plateau detection (independent of TR collapse) ----
        if self.use_meridian_v2 and self.feasible_mask.any():
            cur_best = self._incumbent_best_y()
            if not np.isfinite(self._best_y_seen):
                self._best_y_seen = cur_best
                self._iters_since_improvement = 0
            elif cur_best > self._best_y_seen + 1e-3 * max(abs(self._best_y_seen), 1.0):
                self._best_y_seen = cur_best
                self._iters_since_improvement = 0
            else:
                self._iters_since_improvement += 1
        plateau = (
            self.use_meridian_v2
            and self._iters_since_improvement >= self.plateau_K
            and (
                self.plateau_max_restarts < 0
                or self._restart_count_v2 < self.plateau_max_restarts
            )
        )

        # ---- Restart from class pool if trust region collapsed ----
        if self.trust.restart_triggered:
            self._restart_z = self._restart_center()
            self.trust.reset_after_restart()
            # TR-driven restart counts as an A1 reset too
            self._iters_since_improvement = 0
            plateau = False

        # ---- A1+A6: plateau restart (force head reset + MCTS or class pool) ----
        if plateau:
            self._iters_since_improvement = 0
            self._restart_count_v2 += 1
            self.trust.reset_after_restart()  # also resets L
            # Force a head reset on the next refit (independent of cadence).
            if self.use_property_acquisition:
                self.surrogate.reset_property_heads()
            self._needs_refit = True

            if self.restart_strategy == "mcts":
                # Refit before scoring leaves; falls back gracefully if heads
                # aren't ready yet (e.g. first restart with no feasible data).
                self._refit_surrogate()
                self._maybe_refit_subspace()
                self._needs_refit = False
                self._last_restart_strategy = "mcts"
                Y_feas = self.Y[self._eligible_incumbent_indices()] if self.feasible_mask.any() else np.empty(0)
                # Use the adaptive shell for the restart batch too: a static
                # window will silently force MCTS leaves onto the wrong shell
                # whenever decoder/dim differs from the hand-tuned default.
                shell_min, shell_max = self._shell_bounds()
                mcts_cfg = MCTSRestartConfig(
                    n_leaves=int(getattr(self.cfg, "mcts_n_leaves", 4)),
                    n_samples_per_leaf=int(getattr(self.cfg, "mcts_n_samples_per_leaf", 16)),
                    sigma_init=float(getattr(self.cfg, "mcts_sigma_init", 0.05)),
                    sigma_decay=float(getattr(self.cfg, "mcts_sigma_decay", 0.995)),
                    c0=float(getattr(self.cfg, "mcts_c0", 0.1)),
                    seed_pool_size=int(getattr(self.cfg, "mcts_seed_pool_size", 64)),
                    recent_batch_window=int(
                        getattr(self.cfg, "mcts_recent_batch_window", 20)
                    ),
                )
                if self._property_acq is not None:
                    # mcts_restart_batch only uses Y_feasible.max() as the EI
                    # threshold. Keep the same fresh-aware property threshold
                    # used by the normal candidate cloud; otherwise plateau
                    # restarts can still inherit a cache incumbent that is too
                    # high for useful property-EI ranking.
                    Y_restart = np.asarray(
                        [self._property_acq_best_y()], dtype=np.float32
                    )
                else:
                    Y_restart = Y_feas
                Z_batch = mcts_restart_batch(
                    surrogate=self.surrogate,
                    property_acq=self._property_acq,
                    Y_feasible=Y_restart,
                    X_recent=self.X,
                    seeds_dir=self._seeds_dir,
                    class_key=self._class_key,
                    bounds=(self.low, self.high),
                    batch_size=self.batch_size,
                    cfg=mcts_cfg,
                    shell_radius_min=shell_min,
                    shell_radius_max=shell_max,
                )
                return Z_batch.astype(np.float32)
            else:
                # "class_pool" fallback: pull a fresh anchor and continue
                # through the normal suggest pipeline below.
                self._restart_z = self._restart_center()
                self._last_restart_strategy = "class_pool"

        # ---- Refit surrogate + (after warmup) subspace ----
        if self._needs_refit:
            self._refit_surrogate()
            self._maybe_refit_subspace()
            self._needs_refit = False

        # ---- Full-dim anisotropic Sobol cloud with sparse perturbation ----
        z_center = self._current_best()
        weights = self.subspace.w_ambient          # (dim,) per-axis importance

        # Sobol on [0,1]^dim → perturbations scaled by weights × trust-region L
        u = self._sobol.draw(self.n_candidates).numpy().astype(np.float32)
        pert = z_center[None, :] + (u - 0.5) * L_eff * weights[None, :]
        pert = np.clip(pert, self.low, self.high)

        # UNIFORM sparse perturbation mask (TuRBO high-D trick).
        # Every one of the d dims has equal probability of being perturbed.
        # This guarantees no axis is ever ignored, so rare-direction pockets
        # stay reachable -- the same property that lets TuRBO/DANTE find
        # the 0.94+ texture pockets on v1 that the old subspace MERIDIAN
        # missed. Anisotropy is applied to the perturbation AMPLITUDE only
        # (via ``weights`` above), not to the mask probability -- biasing
        # both would double-count importance and reproduce the v1 plateau.
        prob_perturb = float(min(self._perturb_dims_target / float(self.dim), 1.0))
        mask = np.random.rand(self.n_candidates, self.dim) < prob_perturb
        # Guarantee at least 1 dim perturbed per candidate
        empty = mask.sum(axis=1) == 0
        if empty.any():
            mask[
                np.where(empty)[0],
                np.random.randint(0, self.dim, size=int(empty.sum())),
            ] = True

        # Build candidates: center + sparse perturbation
        Z_cand = np.tile(z_center, (self.n_candidates, 1))
        Z_cand[mask] = pert[mask]
        Z_cand = np.clip(Z_cand, self.low, self.high)

        # ---- DANTE-style diffuse exploration injection ----
        # Replace a fraction of the candidate pool with full-512D isotropic
        # perturbations. Sigma is tied to the trust-region length so this
        # tracks exploitation/exploration balance. The DKL surrogate +
        # acquisition will rank both modes uniformly downstream, so DPP
        # naturally picks whichever mode looks promising for this landscape.
        n_diffuse = int(self.n_candidates * diffuse_frac)
        if n_diffuse > 0:
            sigma = self._diffuse_sigma_scale * L_eff * float(
                np.mean(weights)
            )
            diffuse_pert = (
                z_center[None, :]
                + sigma * np.random.randn(n_diffuse, self.dim).astype(np.float32)
            )
            diffuse_pert = np.clip(diffuse_pert, self.low, self.high)
            # Overwrite the *last* n_diffuse rows of Z_cand (Sobol indexing
            # is monotone so this preserves low-discrepancy coverage on the
            # remaining sparse rows).
            Z_cand[-n_diffuse:] = diffuse_pert

        # MERIDIAN-specific: project candidates onto the encoded-real shell
        # ``||z|| in [r_min, r_max]``. The encoder maps real micrographs onto
        # a tight spherical shell in latent space (e.g. 22.07 +/- 0.05 for
        # FM-DiT 512, 27.55 for FM-DiT 768, 31.63 for FM-DiT 1024); decoder
        # fidelity drops sharply off this manifold (blob/speckle regimes).
        # When ``adaptive_shell=true`` (default) the bounds are derived from
        # feasible ||X|| each round so the shell tracks the data manifold
        # automatically across decoders, dimensionalities, and dataset
        # changes -- and a small ``shell_exempt_frac`` of candidates skip
        # projection so the shell can widen if true off-shell optima exist.
        # Setting both static bounds to 0 with adaptive_shell=false disables.
        r_min, r_max = self._shell_bounds()
        Z_cand = self._apply_shell_with_exemption(
            Z_cand, r_min, r_max,
            exempt_frac=exempt_frac,
            low=self.low, high=self.high,
        )

        # ---- Acquisition × feasibility ----
        if self.surrogate.state is None or self.surrogate.state.gp is None:
            # Cold-data fallback: random subset of the cloud
            idx = np.random.choice(len(Z_cand), self.batch_size, replace=False)
            return Z_cand[idx].astype(np.float32)

        scores = acquisition_scores(Z_cand, self.surrogate)

        # ---- A3: optionally augment with property-space EI ----
        if (
            self.use_property_acquisition
            and self._property_acq is not None
            and self.surrogate.state is not None
            and self.surrogate.state.property_heads is not None
            and self.feasible_mask.any()
        ):
            y_best = self._property_acq_best_y()
            prop_scores = self._property_acq.score(Z_cand, self.surrogate, y_best)
            if prop_scores is not None:
                # ---- v5: blend with qLogNEI instead of overwriting ----
                # Both signals carry information (qLogNEI = surrogate value;
                # prop_scores = inverse-design objective EI). Z-standardize so
                # the average is not dominated by whichever has larger scale.
                def _z(x: np.ndarray) -> np.ndarray:
                    s = float(np.std(x))
                    return (x - float(np.mean(x))) / (s if s > 1e-8 else 1.0)
                scores = 0.5 * _z(scores) + 0.5 * _z(prop_scores)

        # ---- Batch selection -------------------------------------------------
        # Legacy: one DPP on one acquisition. v7 portfolio: reserve slots for
        # complementary failure modes so one overconfident surrogate belief
        # cannot spend the whole expensive DAMASK batch.
        def _nearest_score(z: np.ndarray) -> float:
            diff = Z_cand - z.reshape(1, -1)
            idx = int(np.argmin(np.sum(diff * diff, axis=1)))
            return float(scores[idx])

        selected_rows: list[np.ndarray] = []
        slot_scores: list[float] = []
        protected_slots: list[bool] = []

        def _append_candidate(
            z: np.ndarray,
            score: float | None = None,
            protected: bool = False,
        ) -> bool:
            if len(selected_rows) >= self.batch_size:
                return False
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            for old in selected_rows:
                if float(np.linalg.norm(old - z)) < 1e-4:
                    return False
            selected_rows.append(z)
            slot_scores.append(_nearest_score(z) if score is None else float(score))
            protected_slots.append(bool(protected))
            return True

        top_order = np.argsort(scores)[::-1]
        if self._portfolio_batch and self.batch_size >= 4:
            # Slot 1: acquisition exploit. Keep it protected so the later
            # gradient/incumbent guards cannot erase the portfolio's exploit role.
            _append_candidate(Z_cand[top_order[0]], scores[top_order[0]], protected=True)

            # Slot 2: DANTE-style diffuse scout from the injected full-dim tail.
            if n_diffuse > 0:
                start = max(0, len(Z_cand) - n_diffuse)
                diffuse_order = start + np.argsort(scores[start:])[::-1]
                for i in diffuse_order:
                    if _append_candidate(Z_cand[i], scores[i], protected=True):
                        break

            # Slot 3: uncertainty scout (UCB or one-shot Thompson sample) on
            # the same candidate cloud. This captures the "ensemble/TS" role
            # without refitting several expensive DKL models every iteration.
            try:
                mu_u, sigma_u = self.surrogate.predict(Z_cand)
                feas_u = self.surrogate.feasibility(Z_cand)
                if self._portfolio_uncertainty_mode == "ts":
                    u_score = mu_u + sigma_u * np.random.randn(len(mu_u))
                else:
                    u_score = mu_u + self._portfolio_ucb_beta * sigma_u
                u_score = u_score + np.log(np.clip(feas_u, 1e-6, 1.0))
                for i in np.argsort(u_score)[::-1]:
                    if _append_candidate(Z_cand[i], u_score[i], protected=True):
                        break
            except Exception:
                pass

            # Remaining slots: DPP/top-acquisition fill, intentionally
            # replaceable by the gradient seed and incumbent guard below.
            n_fill = max(1, self.batch_size - len(selected_rows))
            Z_fill = greedy_dpp_select(
                Z_cand, scores, self.surrogate,
                q=n_fill, pool_size=self.dpp_pool,
                z_kernel_weight=dpp_z_w,
                z_kernel_lengthscale=self.dpp_z_kernel_lengthscale,
            )
            for z in Z_fill:
                _append_candidate(z, protected=False)
            for i in top_order:
                if len(selected_rows) >= self.batch_size:
                    break
                _append_candidate(Z_cand[i], scores[i], protected=False)
            Z_batch = np.vstack(selected_rows).astype(np.float32)
        else:
            # ---- Diverse batch via greedy-DPP (A4-DPP hybrid kernel optional) ----
            Z_batch = greedy_dpp_select(
                Z_cand, scores, self.surrogate,
                q=self.batch_size, pool_size=self.dpp_pool,
                z_kernel_weight=dpp_z_w,
                z_kernel_lengthscale=self.dpp_z_kernel_lengthscale,
            )
            for z in Z_batch:
                _append_candidate(z, protected=False)
            if not selected_rows:
                _append_candidate(Z_cand[top_order[0]], scores[top_order[0]], protected=False)
            Z_batch = np.vstack(selected_rows).astype(np.float32)
            # Pad with top-acquisition candidates if DPP returned fewer than q
            if len(Z_batch) < self.batch_size:
                for i in top_order:
                    if len(Z_batch) >= self.batch_size:
                        break
                    if _append_candidate(Z_cand[i], scores[i], protected=False):
                        Z_batch = np.vstack([Z_batch, Z_cand[i:i + 1]])

        def _replace_lowest_available(z: np.ndarray, protected: bool = True) -> bool:
            nonlocal Z_batch
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            if len(Z_batch) < self.batch_size:
                if not any(float(np.linalg.norm(Z_batch[i] - z)) < 1e-4 for i in range(len(Z_batch))):
                    # This branch is rarely used (batches are normally full),
                    # but keeps the helper safe for cold/DPP edge cases.
                    ok = _append_candidate(z, protected=protected)
                    if ok:
                        Z_batch = np.vstack(selected_rows).astype(np.float32)
                    return ok
                return False
            candidates = [i for i, is_protected in enumerate(protected_slots) if not is_protected]
            if not candidates:
                return False
            j = min(candidates, key=lambda i: slot_scores[i])
            Z_batch[j] = z
            slot_scores[j] = _nearest_score(z)
            protected_slots[j] = bool(protected)
            return True

        # ---- A4: replace last batch slot with a property-gradient seed ----
        if (
            self.gradient_seed_enabled
            and self.use_property_acquisition
            and self._property_acq is not None
            and self.surrogate.state is not None
            and self.surrogate.state.property_heads is not None
            and self.feasible_mask.any()
            and len(Z_batch) >= 2
        ):
            gs_cfg = GradientSeedConfig(
                n_candidates=int(getattr(self.cfg, "gs_n_candidates", 32)),
                n_steps=int(getattr(self.cfg, "gs_n_steps", 20)),
                lr=float(getattr(self.cfg, "gs_lr", 0.05)),
                grad_clip=float(getattr(self.cfg, "gs_grad_clip", 1.0)),
                feas_min=float(getattr(self.cfg, "gs_feas_min", 0.3)),
                shell_radius_min=r_min,
                shell_radius_max=r_max,
            )
            # Targets / weights / denominators from the property acquisition,
            # in the column order of ``self.property_keys``.
            n_props = len(self.property_keys)
            tv = np.full(n_props, np.nan, dtype=np.float64)
            wv = np.zeros(n_props, dtype=np.float64)
            dv = np.ones(n_props, dtype=np.float64)
            for idx, t, w, d_ in self._property_acq._terms:  # pylint: disable=protected-access
                tv[idx] = t
                wv[idx] = w
                dv[idx] = d_
            obs_mask = getattr(self.surrogate.state, "p_obs_mask", None)
            if obs_mask is not None:
                obs_mask = np.asarray(obs_mask, dtype=bool)
                tv[~obs_mask] = np.nan
                wv[~obs_mask] = 0.0
            try:
                z_seed = gradient_seed_candidate(
                    surrogate=self.surrogate,
                    z_center=z_center,
                    z_pool=Z_cand,
                    targets_vec=tv,
                    weights_vec=wv,
                    denom_vec=dv,
                    bounds=(self.low, self.high),
                    cfg=gs_cfg,
                )
            except Exception:
                z_seed = None
            if z_seed is not None:
                _replace_lowest_available(z_seed, protected=True)

        # ---- v5: incumbent-keep -- guarantee one slot polishes the best seen ----
        # If the running best (current_best, possibly topk_centroid) is not
        # already in the batch, replace the lowest-acquisition slot with it +
        # a tiny jitter (so DAMASK doesn't see an exact duplicate of a prior eval).
        if len(Z_batch) >= 1 and self.feasible_mask.any():
            z_inc = self._current_best().astype(np.float32).reshape(-1)
            already = any(
                float(np.linalg.norm(Z_batch[i].astype(np.float32) - z_inc)) < 1e-3
                for i in range(len(Z_batch))
            )
            if not already:
                jitter = (0.01 * np.random.randn(self.dim)).astype(np.float32)
                z_inc_j = np.clip(z_inc + jitter, self.low, self.high)
                _replace_lowest_available(z_inc_j, protected=True)
        return Z_batch.astype(np.float32)
