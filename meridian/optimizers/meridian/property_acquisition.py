"""Property-space acquisition for MERIDIAN (A3).

Standard Bayesian-optimisation acquisition treats the surrogate as a black
box ``f : R^d -> R``. In inverse design, we *know* the objective is a
deterministic function of a small set of measured properties

    Y = F(p) = -sqrt( sum_i w_i * ((p_i - t_i)/|t_i|)**2 ) - lambda_g * pen(n_grains)

so a property-aware surrogate (see ``DKLSurrogate.predict_properties``) can
push the acquisition into property space:

    EI_prop(z) = E_{p ~ q_phi(p|z)} [ max( F(p) - y*, 0 ) ]

where ``q_phi(p|z) = N(mu_phi(z), diag(sigma_phi(z)^2))`` is the predictive
distribution of the auxiliary property heads. The expectation is computed
by Monte Carlo (typically 64 samples).

This class mirrors the existing :func:`acquisition_scores` interface so it
can be substituted behind a config flag in ``MeridianOptimizer.suggest()``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class PropertyTargetAcquisition:
    """Monte-Carlo EI in property space, multiplied by feasibility prob.

    Parameters
    ----------
    targets, weights
        Target values and weights as in :class:`TargetDrivenObjective`. Only
        keys present in both ``property_keys`` and ``targets`` contribute.
    property_keys
        Ordered tuple of property names matching the columns produced by
        :meth:`DKLSurrogate.predict_properties`.
    g_min, lambda_g, g_max, lambda_g_hi
        Grain-count penalty knobs (mirror the v2 objective). Applied in
        property space when ``"n_grains"`` is in ``property_keys``.
    n_mc_samples
        Number of property samples drawn per candidate.
    """

    def __init__(
        self,
        targets: Mapping[str, float] | None,
        weights: Mapping[str, float] | None,
        property_keys: Sequence[str],
        g_min: int = 0,
        lambda_g: float = 0.0,
        g_max: int = 0,
        lambda_g_hi: float = 0.0,
        n_mc_samples: int = 64,
        sigma_floor_frac: float = 0.0,
        reachable_prior: Mapping[str, object] | None = None,
        objective_kind: str = "target",
        objective_params: Mapping[str, float | str] | None = None,
    ) -> None:
        self.property_keys = tuple(property_keys)
        self.objective_kind = str(objective_kind).lower()
        self.objective_params = dict(objective_params or {})
        # Pre-compute (idx, target, weight, |target|) for each property that
        # contributes to the distance term.
        self._terms: list[tuple[int, float, float, float]] = []
        targets = targets or {}
        weights = weights or {}
        for k, t in targets.items():
            if k in self.property_keys:
                idx = self.property_keys.index(k)
                w = float(weights.get(k, 1.0))
                denom = abs(float(t)) if abs(float(t)) > 1e-9 else 1.0
                self._terms.append((idx, float(t), w, denom))
        # A2: per-column target vector for the raw-unit sigma floor. Columns
        # without a target keep NaN so the floor is skipped for them.
        import numpy as _np
        n_props = len(self.property_keys)
        self._target_vec = _np.full(n_props, _np.nan, dtype=_np.float64)
        for idx, t, _w, _d in self._terms:
            self._target_vec[idx] = float(t)
        self.sigma_floor_frac = float(sigma_floor_frac)
        # n_grains penalty uses its own column lookup
        self._g_idx = (
            self.property_keys.index("n_grains")
            if "n_grains" in self.property_keys
            else -1
        )
        self.g_min = int(g_min)
        self.lambda_g = float(lambda_g)
        self.g_max = int(g_max)
        self.lambda_g_hi = float(lambda_g_hi)
        self.n_mc_samples = int(n_mc_samples)

        # Optional acquisition-only prior over the *empirically reachable*
        # high-score manifold. The published objective remains unchanged; this
        # only nudges property-EI toward the property bands repeatedly observed
        # in cache-beating v2 runs (e.g. n≈0.30, K≈600 MPa, grains≈60--80).
        self._reachable_prior_weight: float = 0.0
        self._reachable_band_terms: list[tuple[int, float, float, float]] = []
        self._reachable_target_terms: list[tuple[int, float, float]] = []
        if reachable_prior and bool(reachable_prior.get("enabled", False)):
            self._reachable_prior_weight = float(
                reachable_prior.get("weight", 0.25)
            )
            bands = reachable_prior.get("bands", {}) or {}
            if isinstance(bands, Mapping):
                for key, band in bands.items():
                    if key not in self.property_keys:
                        continue
                    try:
                        lo, hi = float(band[0]), float(band[1])  # type: ignore[index]
                    except (TypeError, ValueError, IndexError):
                        continue
                    if hi < lo:
                        lo, hi = hi, lo
                    idx = self.property_keys.index(str(key))
                    scale = max(0.5 * (hi - lo), 1e-6)
                    self._reachable_band_terms.append((idx, lo, hi, scale))
            targets_prior = reachable_prior.get("targets", {}) or {}
            scales_prior = reachable_prior.get("scales", {}) or {}
            if isinstance(targets_prior, Mapping):
                for key, target in targets_prior.items():
                    if key not in self.property_keys:
                        continue
                    idx = self.property_keys.index(str(key))
                    try:
                        t = float(target)
                        s = float(scales_prior.get(key, max(abs(t), 1.0)))  # type: ignore[union-attr]
                    except (TypeError, ValueError):
                        continue
                    self._reachable_target_terms.append((idx, t, max(s, 1e-6)))

    # ----------------------------------------------------- reachable prior
    def _reachable_prior_log_bonus(
        self, mu_p: np.ndarray, active_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Return an acquisition-space bonus for reachable property bands.

        The bonus is non-positive and equals zero inside every configured
        soft band/target. It therefore cannot create artificial optima far
        from high-EI candidates; it only breaks ties toward the empirically
        successful property manifold.
        """
        if self._reachable_prior_weight <= 0.0:
            return np.zeros(mu_p.shape[0], dtype=np.float64)
        penalty = np.zeros(mu_p.shape[0], dtype=np.float64)
        for idx, lo, hi, scale in self._reachable_band_terms:
            if active_mask is not None and not bool(active_mask[idx]):
                continue
            below = np.clip(lo - mu_p[:, idx], 0.0, None)
            above = np.clip(mu_p[:, idx] - hi, 0.0, None)
            penalty += ((below + above) / scale) ** 2
        for idx, target, scale in self._reachable_target_terms:
            if active_mask is not None and not bool(active_mask[idx]):
                continue
            penalty += ((mu_p[:, idx] - target) / scale) ** 2
        return -float(self._reachable_prior_weight) * penalty

    # -------------------------------------------------------- _objective_mc
    def _objective_from_property_samples(
        self, P: np.ndarray, active_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply ``F(p)`` to a batch of property samples ``P`` of shape
        ``(K, N, n_props)``. Returns ``(K, N)`` of objective values."""
        if self.objective_kind in {"v1", "toughness", "v1_toughness"}:
            return self._v1_objective_from_property_samples(P, active_mask)
        ssq = np.zeros(P.shape[:2], dtype=np.float64)
        for idx, target, weight, denom in self._terms:
            if active_mask is not None and not bool(active_mask[idx]):
                continue
            ssq += weight * ((P[..., idx] - target) / denom) ** 2
        score = -np.sqrt(np.maximum(ssq, 0.0))
        if self._g_idx >= 0 and (active_mask is None or bool(active_mask[self._g_idx])):
            ng = P[..., self._g_idx]
            if self.lambda_g > 0.0 and self.g_min > 0:
                shortfall = np.clip((self.g_min - ng) / float(self.g_min), 0.0, 1.0)
                score = score - self.lambda_g * shortfall
            if self.lambda_g_hi > 0.0 and self.g_max > 0:
                excess = np.clip(
                    (ng - self.g_max) / float(self.g_max), 0.0, 2.0
                )
                score = score - self.lambda_g_hi * excess
        return score

    def _v1_objective_from_property_samples(
        self, P: np.ndarray, active_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Apply the clean V1 work-weighted objective in property space."""
        params = self.objective_params
        keys = self.property_keys

        def _idx(name: str) -> int:
            return keys.index(name) if name in keys else -1

        i_sy = _idx("sigma_y")
        i_su = _idx("sigma_u")
        i_n = _idx("n")
        i_eu = _idx("epsilon_uniform")
        work_key = str(params.get("work_key", ""))
        i_work = _idx(work_key) if work_key else -1
        if i_work < 0:
            i_work = _idx("work_uniform")
        # If no explicit work column found, fall back to sigma_u * epsilon_uniform
        use_su_x_eu = False
        if i_work < 0:
            if i_su >= 0 and i_eu >= 0:
                use_su_x_eu = True
            else:
                # Try plastic_work_uniform as last resort
                i_work = _idx("plastic_work_uniform")

        if use_su_x_eu:
            required = [i_sy, i_su, i_eu]
        else:
            required = [i_sy, i_su, i_work]
        if any(i < 0 for i in required):
            return np.full(P.shape[:2], -np.inf, dtype=np.float64)
        if active_mask is not None and any(not bool(active_mask[i]) for i in required):
            return np.full(P.shape[:2], -np.inf, dtype=np.float64)

        sigma_y_ref = float(params.get("sigma_y_ref", 150.0))
        toughness_ref = float(params.get("toughness_ref", 62.0))
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 1.0))
        sigma_u_ref = float(params.get("sigma_u_ref", 315.0))
        gamma = float(params.get("gamma", 0.25))
        n_min = float(params.get("n_min", 0.10))
        lambda_n = float(params.get("lambda_n", 5.0))

        strength = np.clip(P[..., i_sy], 0.0, None) / max(sigma_y_ref, 1e-8)
        if use_su_x_eu:
            work = np.clip(P[..., i_su], 0.0, None) * np.clip(P[..., i_eu], 0.0, None)
        else:
            work = np.clip(P[..., i_work], 0.0, None)
        toughness = work / max(toughness_ref, 1e-8)
        score = np.power(strength, alpha) * np.power(toughness, beta)
        if gamma != 0.0:
            uts = np.clip(P[..., i_su], 0.0, None) / max(sigma_u_ref, 1e-8)
            score = score * np.power(uts, gamma)
        if i_n >= 0 and (active_mask is None or bool(active_mask[i_n])):
            score = score - lambda_n * np.clip(n_min - P[..., i_n], 0.0, None)
        if self._g_idx >= 0 and (active_mask is None or bool(active_mask[self._g_idx])):
            ng = P[..., self._g_idx]
            if self.lambda_g > 0.0 and self.g_min > 0:
                shortfall = np.clip((self.g_min - ng) / float(self.g_min), 0.0, 1.0)
                score = score - self.lambda_g * shortfall
            if self.lambda_g_hi > 0.0 and self.g_max > 0:
                excess = np.clip((ng - self.g_max) / float(self.g_max), 0.0, 2.0)
                score = score - self.lambda_g_hi * excess
        return score

    # ---------------------------------------------------------------- score
    def score(
        self,
        Z_cand: np.ndarray,
        surrogate,
        y_best: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Property-space EI on the candidate cloud, multiplied by g(z).

        Returns a 1-D array of acquisition scores (in log-space, monotone
        with raw EI) for use by the existing greedy-DPP batch selector.
        """
        out = surrogate.predict_properties(Z_cand)
        if out is None:
            # Heads not fitted yet: signal failure to caller (it will fall
            # back to qLogNEI).
            return None  # type: ignore[return-value]
        mu_p, sigma_p = out  # (N, n_props)
        active_mask = getattr(getattr(surrogate, "state", None), "p_obs_mask", None)
        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool)

        # A2: raw-unit sigma floor. After many feasible observations the
        # property heads' aleatoric variance saturates to ``log_var.clamp_min``
        # which corresponds to physical sigma ~ 0.001 * p_std. The MC
        # integral then becomes a delta at mu_p and EI degenerates to
        # argmax-mu, killing exploration. We enforce
        #     sigma_p_i >= sigma_floor_frac * max(|mu_p_i - target_i|,
        #                                         abs_floor_frac * |target_i|)
        # so the optimizer always has a non-zero exploration radius even at
        # convergence (when mu_p approaches target and the gap-based floor
        # would otherwise collapse to zero, also killing EI). Skipped on
        # columns without a target (NaN target_vec entry).
        if self.sigma_floor_frac > 0.0:
            tv = self._target_vec[None, :]  # (1, n_props)
            mask = np.isfinite(tv)
            if active_mask is not None:
                mask = mask & active_mask[None, :]
            if mask.any():
                gap = np.abs(mu_p - np.where(mask, tv, 0.0))
                # Absolute backstop: 5% of |target| (or 5% of |mu| if target=0)
                abs_floor_frac = 0.05
                backstop = abs_floor_frac * np.maximum(
                    np.abs(np.where(mask, tv, mu_p)), 1e-8
                )
                floor = self.sigma_floor_frac * np.maximum(gap, backstop)
                sigma_p = np.where(mask, np.maximum(sigma_p, floor), sigma_p)

        if rng is None:
            rng = np.random.default_rng()
        K = self.n_mc_samples
        N, P = mu_p.shape
        # Reparameterised Gaussian samples: (K, N, P)
        eps = rng.standard_normal(size=(K, N, P)).astype(np.float64)
        P_samples = mu_p[None, :, :] + sigma_p[None, :, :] * eps
        Y_samp = self._objective_from_property_samples(P_samples, active_mask)  # (K, N)
        improvement = np.maximum(Y_samp - float(y_best), 0.0)
        ei = improvement.mean(axis=0)  # (N,)
        # Feasibility multiplier — match acquisition_scores: log-space sum
        feas = surrogate.feasibility(Z_cand)
        log_ei = np.log(np.clip(ei, 1e-30, None))
        score = log_ei + np.log(np.clip(feas, 1e-6, 1.0))
        if self._reachable_prior_weight > 0.0:
            score = score + self._reachable_prior_log_bonus(mu_p, active_mask)
        return score
