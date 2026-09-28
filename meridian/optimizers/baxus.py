"""BAxUS: Bayesian Optimization with Adaptively Expanding Subspaces.

Implements the random-Hadamard-style sparse projection from low target_dim
into the full input space, with periodic dimension doubling on stagnation.
Adapted (and simplified) from the BoTorch BAxUS tutorial.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from meridian.optimizers import BaseOptimizer


@dataclass
class BaxusState:
    dim: int
    target_dim: int
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5e-2
    length_max: float = 1.6
    success_counter: int = 0
    success_tolerance: int = 3
    failure_counter: int = 0
    failure_tolerance: int = 0
    best_value: float = -float("inf")


def _update_state(state: BaxusState, Y_new: np.ndarray) -> BaxusState:
    if max(Y_new) > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, float(np.max(Y_new)))
    return state


def _embedding_matrix(target_dim: int, full_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse +/-1 random projection: each input column has exactly 1 nonzero."""
    S = np.zeros((full_dim, target_dim), dtype=np.float64)
    cols = rng.integers(0, target_dim, size=full_dim)
    signs = rng.choice([-1.0, 1.0], size=full_dim)
    for r, (c, s) in enumerate(zip(cols, signs)):
        S[r, c] = s
    return S


def _pca_embedding_from_pool(Z: np.ndarray, target_dim: int) -> tuple[np.ndarray, int]:
    """PCA-of-seeds embedding for BAxUS.

    Returns a (full_dim, k) matrix ``S`` whose columns are the leading PCA
    directions of the seed pool, replacing the standard random-sparse BAxUS
    embedding. This is the fix for the FM-DiT manifold problem: a random
    embedding sends most target-space points off the trained-decoder manifold
    (-> noise images), whereas a PCA-aligned embedding keeps lifted candidates
    inside the seed cluster.

    ``k = min(target_dim, rank(Z_centered))``; with N=100 seeds the centered
    matrix has rank <= 99 so larger target_dim values are silently capped.
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _sing, Vt = np.linalg.svd(Zc, full_matrices=False)
    k_eff = int(min(target_dim, Vt.shape[0]))
    # Vt is (rank, full_dim); take first k rows -> (k, full_dim) -> transpose.
    Q = Vt[:k_eff].T.astype(np.float64)  # (full_dim, k_eff), orthonormal columns
    return Q, k_eff


class BAxUSOptimizer(BaseOptimizer):
    name = "baxus"
    # B2: BAxUS works in a low-dim subspace whose orientation is critical.
    # Codec rejects (blank_decode, too_few_grains) carry the SIGNAL "this
    # subspace direction is bad" -- masking them out leaves the GP blind to
    # 30-40% of its evaluations and the subspace drifts off-manifold. Opt
    # in to reject feedback so the loop forwards floor-penalised samples
    # alongside successful ones.
    wants_reject_feedback: bool = True

    def __init__(self, dim, bounds, batch_size, baxus_cfg, device: str = "cuda") -> None:
        super().__init__(dim, bounds, batch_size)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.cfg = baxus_cfg
        target_dim = int(baxus_cfg.initial_target_dim)
        tau_fail_raw = max(1, int(math.ceil(target_dim / batch_size * float(baxus_cfg.failure_tolerance_factor))))
        # See note in turbo.py: cap tau_fail so the published TR restart
        # mechanism is reachable within typical NeurIPS-scale budgets.
        # Original ``failure_tolerance_factor`` left at default.
        tau_fail_max = int(getattr(baxus_cfg, "failure_tolerance_max", 8))
        tau_fail = min(tau_fail_raw, max(1, tau_fail_max))
        self.state = BaxusState(
            dim=dim, target_dim=target_dim,
            length=float(baxus_cfg.length_init), length_init=float(baxus_cfg.length_init),
            length_min=float(baxus_cfg.length_min), length_max=float(baxus_cfg.length_max),
            failure_tolerance=tau_fail,
            success_tolerance=int(baxus_cfg.success_tolerance),
        )
        self.acquisition = str(baxus_cfg.acquisition).lower()
        self._rng = np.random.default_rng()
        # ``S`` starts as random-sparse (BAxUS default). Rebuilt as a
        # PCA-of-seeds basis on the first ``_on_update`` call so that the
        # target-space search lives on the FM-DiT seed manifold rather than
        # in random axis-aligned directions (which produced noise images).
        self.S = _embedding_matrix(self.state.target_dim, self.dim, self._rng)
        self._S_is_pca: bool = False
        # Cache target-space coordinates for points we evaluated (after expansion)
        self._U_target: np.ndarray = np.empty((0, self.state.target_dim), dtype=np.float64)
        # Optional anchor: pin the trust-region center at this latent for the
        # first suggestion (so DANTE / MERIDIAN / BAxUS all start from the same
        # known-good microstructure when ``loop.initial_seed`` is configured).
        self._initial_seed_z: np.ndarray | None = None
        self._initial_seed_used: bool = False
        # Persistent translation that shifts the BAxUS subspace to pass
        # through the anchor (set by ``set_initial_seed``). Zero by default.
        self._anchor_residual: np.ndarray = np.zeros(self.dim, dtype=np.float32)

    # ----------------------------------------------------------- explicit anchor
    def set_initial_seed(self, z: np.ndarray) -> None:
        """Pin the trust-region center for the first ``suggest()`` to ``z``."""
        self._initial_seed_z = np.asarray(z, dtype=np.float32).reshape(-1)
        self._initial_seed_used = False
        # Anchor-offset: BAxUS searches in the rank-target_dim PCA subspace;
        # the anchor (and seed pool mean) generally lie OUTSIDE this subspace,
        # so the naive round-trip collapses ||z||. Compensate by translating
        # the entire searchable subspace so it passes through the anchor:
        # candidate full-d vectors get ``_anchor_residual`` added, where
        # the residual is ``anchor - project(unproject(anchor))``.
        z_unit = self._to_unit_full(self._initial_seed_z[None, :])
        rhs = 2.0 * z_unit - 1.0
        S_pinv = np.linalg.pinv(self.S)
        u_t_anchor = (rhs @ S_pinv.T).clip(-1.0, 1.0)
        anchor_proj_unit = self._project_target_to_full(u_t_anchor)
        anchor_proj_full = self._from_unit_full(anchor_proj_unit)[0]
        self._anchor_residual = (self._initial_seed_z - anchor_proj_full).astype(np.float32)

    def _to_unit_full(self, X: np.ndarray) -> np.ndarray:
        return (X - self.low) / (self.high - self.low)

    def _from_unit_full(self, U: np.ndarray) -> np.ndarray:
        return self.low + U * (self.high - self.low)

    def _project_target_to_full(self, U_target: np.ndarray) -> np.ndarray:
        # U_target in [-1, 1]^target_dim. Embedded into [0, 1]^full_dim via S then shifted.
        full = U_target @ self.S.T  # (N, full_dim) in [-1, 1] roughly
        return np.clip(0.5 * (full + 1.0), 0.0, 1.0)

    def _expand_subspace(self) -> None:
        """Double target dimensionality and re-project existing data."""
        new_target = min(self.dim, self.state.target_dim * 2)
        if new_target == self.state.target_dim:
            return
        # If we have a PCA basis, rebuild from data (more useful directions);
        # otherwise extend the random-sparse stitch as before.
        if self._S_is_pca and len(self.X) >= 2:
            new_S, k_eff = _pca_embedding_from_pool(self.X, new_target)
            self.S = new_S
            new_target = k_eff
        else:
            old_S = self.S
            new_S = _embedding_matrix(new_target, self.dim, self._rng)
            # Stitch: existing target columns embed into the first old_S.shape[1] cols of new_S
            new_S[:, : old_S.shape[1]] = old_S
            self.S = new_S
        # Re-derive target-space points by least-squares pseudo-inverse
        U_full = self._to_unit_full(self.X)  # in [0,1]^full
        # Map [0,1] -> [-1, 1] then solve S^T u_target ≈ (2*U_full - 1)
        rhs = 2.0 * U_full - 1.0
        # Pseudo-inverse projection (target_dim << full_dim)
        S_pinv = np.linalg.pinv(self.S)  # (target_dim, full_dim)
        self._U_target = (rhs @ S_pinv.T).clip(-1.0, 1.0)
        self.state.target_dim = new_target
        self.state.length = self.state.length_init
        self.state.failure_counter = 0
        self.state.success_counter = 0

    def _on_update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        # First-call PCA fit: replace random S with the leading PCA directions
        # of whatever pool we just got (typically the 100-entry seed cache).
        # Caps target_dim to rank(centered pool) so SVD is well-defined.
        if not self._S_is_pca and len(self.X) >= 2:
            S_pca, k_eff = _pca_embedding_from_pool(self.X, self.state.target_dim)
            self.S = S_pca
            self.state.target_dim = k_eff
            self._S_is_pca = True
            # Re-project all existing points into the new target space.
            U_full_all = self._to_unit_full(self.X)
            rhs_all = 2.0 * U_full_all - 1.0
            S_pinv = np.linalg.pinv(self.S)
            self._U_target = (rhs_all @ S_pinv.T).clip(-1.0, 1.0)
            # And recompute Sobol engine dim to match.
            # (lazily re-created in suggest() if needed; we just reset here.)
        else:
            # Append target-space coords for new points (subtract anchor
            # residual so we round-trip in the same frame as suggest()).
            U_full_new = self._to_unit_full(X_new - self._anchor_residual[None, :])
            rhs = 2.0 * U_full_new - 1.0
            S_pinv = np.linalg.pinv(self.S)
            U_t = (rhs @ S_pinv.T).clip(-1.0, 1.0)
            self._U_target = np.vstack([self._U_target, U_t])
        prev_failure = self.state.failure_counter
        _update_state(self.state, Y_new)
        # Trigger subspace expansion when length collapses below floor
        if self.state.length < self.state.length_min and self.state.target_dim < self.dim:
            self._expand_subspace()

    def _fit_gp(self, U: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=U.shape[-1], lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(U, Y.unsqueeze(-1), likelihood=likelihood, covar_module=covar).to(
            device=self.device, dtype=self.dtype
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return model

    def suggest(self) -> np.ndarray:
        if len(self._U_target) == 0:
            # Cold start in target space
            sobol = SobolEngine(self.state.target_dim, scramble=True)
            U_t = (sobol.draw(self.batch_size).numpy() * 2.0 - 1.0)
            U_full = self._project_target_to_full(U_t)
            X = self._from_unit_full(U_full) + self._anchor_residual[None, :]
            return np.clip(X, self.low, self.high).astype(np.float32)

        U_t = torch.from_numpy(self._U_target).to(device=self.device, dtype=self.dtype)
        Y = torch.from_numpy(self.Y).to(device=self.device, dtype=self.dtype)
        Yn = (Y - Y.mean()) / (Y.std() + 1e-6)
        model = self._fit_gp(U_t, Yn)

        # Default: center the trust region on the current best point. When
        # an explicit anchor was supplied via set_initial_seed (e.g. sim_449),
        # use it for the very first suggestion so all optimizers start from
        # the same microstructure.
        if self._initial_seed_z is not None and not self._initial_seed_used:
            U_full_seed = self._to_unit_full(self._initial_seed_z[None, :])
            rhs = 2.0 * U_full_seed - 1.0
            S_pinv = np.linalg.pinv(self.S)
            U_t_seed = (rhs @ S_pinv.T).clip(-1.0, 1.0)[0]
            x_center = torch.from_numpy(U_t_seed).to(device=self.device, dtype=self.dtype)
            self._initial_seed_used = True
        else:
            x_center = U_t[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / self.state.target_dim))
        tr_lb = torch.clamp(x_center - weights * self.state.length, -1.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length, -1.0, 1.0)

        sobol = SobolEngine(self.state.target_dim, scramble=True)
        n_cand = min(5000, max(2000, 200 * self.state.target_dim))
        pert = sobol.draw(n_cand).to(device=self.device, dtype=self.dtype)
        cand_t = tr_lb + (tr_ub - tr_lb) * pert
        ts = MaxPosteriorSampling(model=model, replacement=False)
        X_next_t = ts(cand_t, num_samples=self.batch_size).detach().cpu().numpy()
        U_full = self._project_target_to_full(X_next_t)
        X = self._from_unit_full(U_full) + self._anchor_residual[None, :]
        return np.clip(X, self.low, self.high).astype(np.float32)
