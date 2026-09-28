"""Neural-surrogate-guided Tree Exploration (NTE) for DANTE.

Implements the four innovations from Wei et al. 2025:

  1. DUCB: Data-driven UCB.
        DUCB(node) = v_ML(node) + c0 * c(rho) * sqrt(2*log(N)/(n+1))
  2. Conditional selection: a leaf only replaces the root if its DUCB
     strictly exceeds the root's DUCB.
  3. Local backpropagation: only the root and the chosen leaf get their
     visit counts incremented (no path-update).
  4. Adaptive exploration: c(rho) = max of the running observed values,
     EMA-smoothed.

The tree is conceptually a single-root expansion with stochastic leaves;
we interleave conditional selection and local backprop to scale to high
dimensions without storing the full tree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    z: np.ndarray
    visits: int = 0
    v_ml: float = 0.0


@dataclass
class NTEConfig:
    n_rollouts: int = 200
    n_leaves_per_expand: int = 16
    sigma_init: float = 0.5
    sigma_decay: float = 0.99
    c0: float = 0.1
    rho_smoothing: float = 0.5
    bounds: tuple[float, float] = (-3.0, 3.0)


class NeuralTreeExplorer:
    def __init__(self, cfg: NTEConfig) -> None:
        self.cfg = cfg
        self._rho_ema: float = 1.0  # c(rho) starts neutral

    @staticmethod
    def _ducb(v_ml: float, n_root: int, n_leaf: int, c0: float, c_rho: float) -> float:
        # Add 1 to denominator to avoid division-by-zero on unvisited leaves
        return v_ml + c0 * c_rho * math.sqrt(2.0 * math.log(max(n_root, 1) + 1) / (n_leaf + 1))

    def update_rho(self, observed_values: np.ndarray) -> float:
        if len(observed_values) == 0:
            return self._rho_ema
        c_rho_new = float(np.max(observed_values))
        a = self.cfg.rho_smoothing
        self._rho_ema = a * c_rho_new + (1 - a) * self._rho_ema
        return self._rho_ema

    def _expand(self, root: Node, sigma: float, rng: np.random.Generator) -> list[Node]:
        leaves = []
        for _ in range(self.cfg.n_leaves_per_expand):
            z_new = root.z + sigma * rng.standard_normal(size=root.z.shape).astype(np.float32)
            z_new = np.clip(z_new, self.cfg.bounds[0], self.cfg.bounds[1])
            leaves.append(Node(z=z_new))
        return leaves

    def search(
        self,
        seed_z: np.ndarray,
        surrogate,
        n_candidates: int,
        observed_values: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Run NTE rollouts; return top-`n_candidates` distinct latents.

        `seed_z` is the current best observed latent (root initialization).
        """
        rng = rng or np.random.default_rng()
        c_rho = self.update_rho(observed_values)

        root = Node(z=seed_z.astype(np.float32).copy())
        root.v_ml = float(surrogate.predict(root.z[None])[0])
        visited: list[tuple[float, np.ndarray]] = [(root.v_ml, root.z.copy())]

        sigma = self.cfg.sigma_init
        for _ in range(self.cfg.n_rollouts):
            leaves = self._expand(root, sigma, rng)
            Z = np.stack([n.z for n in leaves], axis=0)
            preds = surrogate.predict(Z)
            for n_, p in zip(leaves, preds):
                n_.v_ml = float(p)

            # Conditional selection
            ducb_root = self._ducb(root.v_ml, root.visits, root.visits, self.cfg.c0, c_rho)
            best_leaf = max(
                leaves,
                key=lambda nd: self._ducb(nd.v_ml, root.visits, nd.visits, self.cfg.c0, c_rho),
            )
            ducb_leaf = self._ducb(best_leaf.v_ml, root.visits, best_leaf.visits, self.cfg.c0, c_rho)

            # Local backpropagation: bump root and selected leaf only.
            root.visits += 1
            best_leaf.visits += 1

            if ducb_leaf > ducb_root:
                root = best_leaf  # promote
            # Track every explored leaf for the candidate pool
            for n_ in leaves:
                visited.append((n_.v_ml, n_.z.copy()))

            sigma *= self.cfg.sigma_decay

        # Rank by predicted value, deduplicate by quantizing z to 4 decimals
        visited.sort(key=lambda t: t[0], reverse=True)
        out: list[np.ndarray] = []
        seen = set()
        for v, z in visited:
            key = tuple(np.round(z, 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append(z)
            if len(out) >= n_candidates:
                break
        return np.stack(out, axis=0)
