"""A4: property-gradient seeding for MERIDIAN.

Picks one batch slot by **inverse design**: starts from a Sobol cloud of
candidates inside the trust region, runs K Adam steps minimising

    L(z) = sum_i w_i * ((mu_p_i(z) - target_i) / |target_i|)^2

through the (frozen) DKL trunk + property heads, projects back to the
feasibility region (classifier > 0.5) and (optionally) the encoded-real
shell, and returns the candidate with the best property-EI.

This is the first MERIDIAN mechanism that uses the *differentiability* of
the auxiliary property surrogate to do inverse design directly, rather
than scoring a random Sobol cloud.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GradientSeedConfig:
    n_candidates: int = 32         # initial cloud (subset of Sobol cloud)
    n_steps: int = 20              # Adam iters per candidate
    lr: float = 0.05               # Adam lr in raw z units
    grad_clip: float = 1.0         # per-step ||grad||_2 cap
    feas_min: float = 0.3          # reject candidates with feasibility < this
    shell_radius_min: float = 0.0  # 0 disables shell projection
    shell_radius_max: float = 0.0


def gradient_seed_candidate(
    surrogate,
    z_center: np.ndarray,
    z_pool: np.ndarray,
    targets_vec: np.ndarray,         # (n_props,) NaN where no target
    weights_vec: np.ndarray,         # (n_props,) 0 where no target
    denom_vec: np.ndarray,           # (n_props,) |target|; 1 where no target
    bounds: tuple[float, float],
    cfg: GradientSeedConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Return a single (d,) candidate or ``None`` if the surrogate isn't ready.

    Parameters
    ----------
    surrogate
        :class:`DKLSurrogate` that exposes ``predict_properties_diff`` and
        ``feasibility_diff``. Returns ``None`` if heads aren't fitted.
    z_pool
        ``(M, d)`` candidate cloud (e.g. the Sobol cloud already produced
        by ``MeridianOptimizer.suggest()``); ``cfg.n_candidates`` rows are
        sub-sampled from it as Adam starting points.
    """
    if surrogate.predict_properties_diff(torch.zeros(1, len(z_center))) is None:
        return None  # heads not ready

    rng = rng or np.random.default_rng()
    M = len(z_pool)
    n = min(cfg.n_candidates, M)
    idx = rng.choice(M, size=n, replace=False)
    z0 = torch.from_numpy(np.ascontiguousarray(z_pool[idx])).to(
        surrogate.device, dtype=surrogate.dtype
    )

    target_mask = np.isfinite(targets_vec)
    if not target_mask.any():
        # No targets to optimise toward; nothing to do
        return None
    tv = torch.from_numpy(np.where(target_mask, targets_vec, 0.0)).to(
        surrogate.device, dtype=surrogate.dtype
    )
    wv = torch.from_numpy(np.where(target_mask, weights_vec, 0.0)).to(
        surrogate.device, dtype=surrogate.dtype
    )
    dv = torch.from_numpy(np.where(target_mask, denom_vec, 1.0)).to(
        surrogate.device, dtype=surrogate.dtype
    )

    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.lr)
    lo, hi = float(bounds[0]), float(bounds[1])

    for _ in range(int(cfg.n_steps)):
        opt.zero_grad()
        out = surrogate.predict_properties_diff(z)
        if out is None:
            return None
        mu_p, _sigma_p = out  # (n, n_props)
        diff = (mu_p - tv) / dv
        loss = (wv * diff.pow(2)).sum(dim=-1).mean()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], cfg.grad_clip)
        opt.step()
        with torch.no_grad():
            z.clamp_(lo, hi)

    # ---- Project to feasibility region ----
    with torch.no_grad():
        feas = surrogate.feasibility_diff(z).cpu().numpy()  # (n,)
    z_np = z.detach().cpu().numpy().astype(np.float32)

    # ---- Optional shell projection (matches MeridianOptimizer.suggest) ----
    if cfg.shell_radius_min > 0.0 and cfg.shell_radius_max >= cfg.shell_radius_min:
        r = np.linalg.norm(z_np, axis=1)
        r = np.where(r > 1e-8, r, 1e-8)
        target_r = np.clip(r, cfg.shell_radius_min, cfg.shell_radius_max)
        z_np = z_np * (target_r / r)[:, None]
        z_np = np.clip(z_np, lo, hi)

    # ---- Score: per-candidate ||p_hat - target|| (lower is better) ----
    with torch.no_grad():
        out = surrogate.predict_properties_diff(
            torch.from_numpy(z_np).to(surrogate.device, dtype=surrogate.dtype)
        )
    if out is None:
        return None
    mu_p_np = out[0].cpu().numpy()
    diff_np = np.where(
        target_mask[None, :], (mu_p_np - targets_vec[None, :]) / denom_vec[None, :], 0.0
    )
    err = (np.where(target_mask[None, :], weights_vec[None, :], 0.0) * diff_np ** 2).sum(axis=-1)

    # Mask infeasible candidates by inflating their error
    err = np.where(feas >= cfg.feas_min, err, err + 1e6)
    best = int(np.argmin(err))
    return z_np[best].astype(np.float32)
