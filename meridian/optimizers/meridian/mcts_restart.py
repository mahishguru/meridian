"""A6: DANTE-style MCTS plateau-restart for MERIDIAN.

When MeridianOptimizer's plateau detector fires (no improvement for K
iters), we run a one-iteration mini Neural Tree Explorer rollout instead
of the standard class-pool restart. The mini-NTE:

  1. Builds leaves by KMeans-partitioning {class-pool centers ∪ recent
     batch members} into ``n_leaves`` clusters.
  2. DUCB-selects the best leaf using **property-EI** (the same
     A2/A3-aware acquisition score MERIDIAN already uses), so the
     exploration is biased toward target-relevant regions.
  3. σ-decay samples ``n_samples_per_leaf`` candidates around each leaf.
  4. Returns the top-``q`` candidates by acquisition score as the full
     batch.

The next iteration resumes normal MERIDIAN suggest().
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meridian.optimizers.meridian.batch import acquisition_scores


@dataclass
class MCTSRestartConfig:
    n_leaves: int = 4
    n_samples_per_leaf: int = 16
    sigma_init: float = 0.05
    sigma_decay: float = 0.995
    c0: float = 0.1
    seed_pool_size: int = 64        # rows sampled from class pool
    recent_batch_window: int = 20   # how many recent batch members to include


def _kmeans(Z: np.ndarray, k: int, n_iters: int = 20,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Tiny KMeans returning cluster centers (k, d)."""
    rng = rng or np.random.default_rng()
    N = len(Z)
    k = max(1, min(k, N))
    centers = Z[rng.choice(N, size=k, replace=False)].copy()
    for _ in range(n_iters):
        d2 = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(axis=1)
        new_centers = np.array([
            Z[assign == j].mean(axis=0) if (assign == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return centers


def mcts_restart_batch(
    surrogate,
    property_acq,
    Y_feasible: np.ndarray,
    X_recent: np.ndarray,
    seeds_dir: str | None,
    class_key: str | None,
    bounds: tuple[float, float],
    batch_size: int,
    cfg: MCTSRestartConfig,
    shell_radius_min: float = 0.0,
    shell_radius_max: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a (batch_size, d) batch from a one-iter NTE rollout.

    Falls back to a small random cloud if the surrogate isn't ready yet.
    """
    rng = rng or np.random.default_rng()
    lo, hi = float(bounds[0]), float(bounds[1])
    d = X_recent.shape[1] if len(X_recent) else 1

    # ---- 1. Assemble seed pool: class-pool sample + recent batch tail ----
    pool_rows: list[np.ndarray] = []
    if seeds_dir and class_key:
        pool_path = Path(seeds_dir) / f"{class_key}.npy"
        if pool_path.is_file():
            cp = np.load(pool_path).astype(np.float32)
            n_take = min(cfg.seed_pool_size, len(cp))
            idx = rng.choice(len(cp), size=n_take, replace=False)
            pool_rows.append(cp[idx])
    if len(X_recent):
        n_take = min(cfg.recent_batch_window, len(X_recent))
        pool_rows.append(X_recent[-n_take:].astype(np.float32))
    if not pool_rows:
        # Cold fallback: random in box
        Z = rng.uniform(lo, hi, size=(batch_size, d)).astype(np.float32)
        return Z
    Z_pool = np.vstack(pool_rows).astype(np.float32)

    # ---- 2. KMeans -> leaves ----
    centers = _kmeans(Z_pool, k=cfg.n_leaves, rng=rng).astype(np.float32)

    # ---- 3. Score each leaf via property-EI; build DUCB ----
    if surrogate.state is None or surrogate.state.gp is None:
        # Surrogate not ready: just sample around random leaves
        z_picks = []
        for c in centers:
            z = c[None, :] + cfg.sigma_init * rng.standard_normal(
                size=(cfg.n_samples_per_leaf, d)
            ).astype(np.float32)
            z_picks.append(np.clip(z, lo, hi))
        Z_cand = np.vstack(z_picks)
    else:
        # Score leaves themselves
        if property_acq is not None and len(Y_feasible):
            y_best = float(Y_feasible.max())
            v_leaf = property_acq.score(centers, surrogate, y_best)
            if v_leaf is None:
                v_leaf = acquisition_scores(centers, surrogate)
        else:
            v_leaf = acquisition_scores(centers, surrogate)
        v_leaf = np.asarray(v_leaf, dtype=np.float64)

        # DUCB: leaf score + c0 * sqrt(2*log(N+1)/(visits+1))
        # On a one-shot rollout every leaf has visits=0, so DUCB simplifies
        # to v_leaf + c0 * sqrt(2*log(N+1)) -- a uniform exploration bonus
        # that gets dwarfed by the score gap. We keep the formula for
        # consistency and weight all leaves equally with sample budget.
        N = len(centers)
        bonus = cfg.c0 * math.sqrt(2.0 * math.log(max(N, 1) + 1))
        ducb = v_leaf + bonus

        # Allocate samples per leaf proportionally to softmax(ducb)
        e = np.exp(ducb - ducb.max())
        p = e / e.sum()
        total_samples = cfg.n_leaves * cfg.n_samples_per_leaf
        alloc = np.maximum(1, np.round(p * total_samples).astype(int))

        # ---- 4. sigma-decay sample around each leaf ----
        z_picks = []
        sigma = cfg.sigma_init
        for j, c in enumerate(centers):
            n_s = int(alloc[j])
            z = c[None, :] + sigma * rng.standard_normal(
                size=(n_s, d)
            ).astype(np.float32)
            z = np.clip(z, lo, hi)
            z_picks.append(z)
            sigma *= cfg.sigma_decay
        Z_cand = np.vstack(z_picks)

    # ---- Optional shell projection (consistent with MERIDIAN.suggest) ----
    if shell_radius_min > 0.0 and shell_radius_max >= shell_radius_min:
        r = np.linalg.norm(Z_cand, axis=1)
        r = np.where(r > 1e-8, r, 1e-8)
        target_r = np.clip(r, shell_radius_min, shell_radius_max)
        Z_cand = Z_cand * (target_r / r)[:, None]
        Z_cand = np.clip(Z_cand, lo, hi)

    # ---- 5. Final score + top-q ----
    if surrogate.state is not None and surrogate.state.gp is not None:
        if property_acq is not None and len(Y_feasible):
            y_best = float(Y_feasible.max())
            scores = property_acq.score(Z_cand, surrogate, y_best)
            if scores is None:
                scores = acquisition_scores(Z_cand, surrogate)
        else:
            scores = acquisition_scores(Z_cand, surrogate)
        order = np.argsort(np.asarray(scores))[::-1]
        return Z_cand[order[:batch_size]].astype(np.float32)
    # Cold fallback
    idx = rng.choice(len(Z_cand), size=min(batch_size, len(Z_cand)), replace=False)
    return Z_cand[idx].astype(np.float32)
