"""TuRBO-1 with Thompson sampling (or qLogEI) — adapted from BoTorch tutorial."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from meridian.optimizers import BaseOptimizer


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5e-2
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False
    # B1: plateau detection independent of TR shrinkage cadence.
    iters_since_improvement: int = 0
    plateau_K: int = 0  # 0 disables plateau-restart


def _update_state(state: TurboState, Y_new: np.ndarray) -> TurboState:
    if max(Y_new) > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.iters_since_improvement = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
        state.iters_since_improvement += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, float(np.max(Y_new)))
    if state.length < state.length_min:
        state.restart_triggered = True
    # B1: force a Sobol cold-restart on plateau even if L hasn't collapsed.
    # Without this, TuRBO's published TR-shrink cadence requires ~7 halvings
    # from L_init=0.8 to reach L_min=0.005 — many more iters than typical
    # NeurIPS budgets, and the optimizer flatlines at iter 6-8.
    if state.plateau_K > 0 and state.iters_since_improvement >= state.plateau_K:
        state.restart_triggered = True
        state.iters_since_improvement = 0
    return state


class TuRBOOptimizer(BaseOptimizer):
    name = "turbo"

    def __init__(self, dim, bounds, batch_size, turbo_cfg, device: str = "cuda") -> None:
        super().__init__(dim, bounds, batch_size)
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.cfg = turbo_cfg
        tau_fail_raw = max(1, int(math.ceil(dim / batch_size * float(turbo_cfg.failure_tolerance_factor))))
        # In high-dim, low-batch regimes (e.g. d=512, q=4 -> tau=192) the
        # canonical formula exceeds typical evaluation budgets, preventing
        # the documented TR restart mechanism from ever triggering. We cap
        # tau_fail at ``failure_tolerance_max`` (default 8) so the
        # algorithm's published restart behaviour is reachable within
        # budget. The original `failure_tolerance_factor` is held at
        # default; only this guardrail is added. Set the cap to a large
        # value (e.g. 10000) to recover the un-capped behaviour.
        tau_fail_max = int(getattr(turbo_cfg, "failure_tolerance_max", 8))
        tau_fail = min(tau_fail_raw, max(1, tau_fail_max))
        self.state = TurboState(
            dim=dim,
            batch_size=batch_size,
            length=float(turbo_cfg.length_init),
            length_min=float(turbo_cfg.length_min),
            length_max=float(turbo_cfg.length_max),
            failure_tolerance=tau_fail,
            success_tolerance=int(turbo_cfg.success_tolerance),
            plateau_K=int(getattr(turbo_cfg, "plateau_K", 0)),
        )
        self.n_candidates = int(turbo_cfg.n_candidates)
        self.acquisition = str(turbo_cfg.acquisition).lower()
        # Optional anchor: pin the trust-region center at this latent for the
        # first suggestion (so DANTE / MERIDIAN / BAxUS / TuRBO all start from
        # the same known-good microstructure when ``loop.initial_seed`` is set).
        self._initial_seed_z: np.ndarray | None = None
        self._initial_seed_used: bool = False

    # ------------------------------------------------------------ explicit anchor
    def set_initial_seed(self, z: np.ndarray) -> None:
        """Pin the trust-region center for the first ``suggest()`` to ``z``."""
        self._initial_seed_z = np.asarray(z, dtype=np.float32).reshape(-1)
        self._initial_seed_used = False

    # Map z in [low, high] <-> u in [0, 1]
    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        return (X - self.low) / (self.high - self.low)

    def _from_unit(self, U: np.ndarray) -> np.ndarray:
        return self.low + U * (self.high - self.low)

    def _on_update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        _update_state(self.state, Y_new)

    def _fit_gp(self, U: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=self.dim, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(U, Y.unsqueeze(-1), likelihood=likelihood, covar_module=covar_module).to(
            device=self.device, dtype=self.dtype
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return model

    def suggest(self) -> np.ndarray:
        if self.state.restart_triggered:
            # Cold restart from Sobol within the unit box
            sobol = SobolEngine(self.dim, scramble=True)
            U = sobol.draw(self.batch_size).numpy()
            self.state = TurboState(
                dim=self.dim, batch_size=self.batch_size,
                length=float(self.cfg.length_init), length_min=float(self.cfg.length_min),
                length_max=float(self.cfg.length_max),
                failure_tolerance=self.state.failure_tolerance,
                success_tolerance=self.state.success_tolerance,
                best_value=float(np.max(self.Y)) if len(self.Y) else -float("inf"),
                plateau_K=self.state.plateau_K,
            )
            return self._from_unit(U).astype(np.float32)

        U = torch.from_numpy(self._to_unit(self.X)).to(device=self.device, dtype=self.dtype)
        Y = torch.from_numpy(self.Y).to(device=self.device, dtype=self.dtype)
        # Standardize Y for GP stability
        Yn = (Y - Y.mean()) / (Y.std() + 1e-6)
        model = self._fit_gp(U, Yn)

        # Trust region centered at current best, scaled by lengthscales.
        # Override with explicit anchor on the first suggest() if provided.
        if self._initial_seed_z is not None and not self._initial_seed_used:
            U_seed = self._to_unit(self._initial_seed_z[None, :])[0]
            x_center = torch.from_numpy(U_seed).to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)
            self._initial_seed_used = True
        else:
            x_center = U[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / self.dim))
        tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)

        if self.acquisition == "ts":
            sobol = SobolEngine(self.dim, scramble=True)
            pert = sobol.draw(self.n_candidates).to(device=self.device, dtype=self.dtype)
            pert = tr_lb + (tr_ub - tr_lb) * pert
            # Probabilistic perturbation mask (TuRBO trick for high dim)
            prob_perturb = min(20.0 / self.dim, 1.0)
            mask = torch.rand(self.n_candidates, self.dim, device=self.device) < prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind):
                mask[ind, torch.randint(0, self.dim, size=(len(ind),), device=self.device)] = True
            X_cand = x_center.expand(self.n_candidates, self.dim).clone()
            X_cand[mask] = pert[mask]
            ts = MaxPosteriorSampling(model=model, replacement=False)
            X_next = ts(X_cand, num_samples=self.batch_size)
        else:  # EI
            acq = qLogExpectedImprovement(model=model, best_f=Yn.max())
            X_next, _ = optimize_acqf(
                acq, bounds=torch.stack([tr_lb, tr_ub]),
                q=self.batch_size, num_restarts=10, raw_samples=512,
            )

        return self._from_unit(X_next.detach().cpu().numpy()).astype(np.float32)
