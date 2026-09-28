"""Optimizer interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseOptimizer(ABC):
    name: str = "base"

    def __init__(self, dim: int, bounds: tuple[float, float], batch_size: int) -> None:
        self.dim = int(dim)
        self.low, self.high = float(bounds[0]), float(bounds[1])
        self.batch_size = int(batch_size)
        self.X: np.ndarray = np.empty((0, self.dim), dtype=np.float32)
        self.Y: np.ndarray = np.empty((0,), dtype=np.float32)

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        X_new = np.atleast_2d(X_new).astype(np.float32)
        Y_new = np.atleast_1d(Y_new).astype(np.float32)
        self.X = np.vstack([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])
        self._on_update(X_new, Y_new)

    def _on_update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        """Override for trust-region / surrogate retraining hooks."""

    def best(self) -> tuple[np.ndarray, float]:
        if len(self.Y) == 0:
            raise RuntimeError("No data; call update() first.")
        i = int(np.argmax(self.Y))
        return self.X[i].copy(), float(self.Y[i])

    @abstractmethod
    def suggest(self) -> np.ndarray:
        """Return a (batch_size, dim) array of next candidates."""


def get_optimizer(cfg) -> BaseOptimizer:
    name = cfg.optimizer.name.lower()
    dim = int(cfg.latent.dim)
    bounds = tuple(cfg.latent.bounds)
    batch_size = int(cfg.optimizer.batch_size)

    if name == "dante":
        from meridian.optimizers.dante.optimizer import DANTEOptimizer
        return DANTEOptimizer(dim=dim, bounds=bounds, batch_size=batch_size, dante_cfg=cfg.optimizer.dante,
                              device=cfg.experiment.device)
    if name == "turbo":
        from meridian.optimizers.turbo import TuRBOOptimizer
        return TuRBOOptimizer(dim=dim, bounds=bounds, batch_size=batch_size, turbo_cfg=cfg.optimizer.turbo,
                              device=cfg.experiment.device)
    if name == "baxus":
        from meridian.optimizers.baxus import BAxUSOptimizer
        return BAxUSOptimizer(dim=dim, bounds=bounds, batch_size=batch_size, baxus_cfg=cfg.optimizer.baxus,
                              device=cfg.experiment.device)
    if name == "meridian":
        from meridian.optimizers.meridian import MeridianOptimizer
        # Pull class-pool wiring from the existing latent.init block so MERIDIAN
        # can warm-start its subspace and seed its restarts from the same source.
        seeds_dir = getattr(cfg.latent.init, "seeds_dir", None)
        class_key = getattr(cfg.latent.init, "class_key", None) or getattr(cfg.codec, "class_key", None)
        # Resolve the active-objective sub-config (targets, weights, grain
        # penalty knobs) so MERIDIAN's property-space acquisition can use
        # the *exact* same definition of F(p) the loop is scoring against.
        obj_name = getattr(getattr(cfg, "objective", None), "name", None)
        objective_cfg = (
            getattr(cfg.objective, obj_name, None) if obj_name else None
        )
        return MeridianOptimizer(
            dim=dim, bounds=bounds, batch_size=batch_size,
            meridian_cfg=cfg.optimizer.meridian,
            seeds_dir=seeds_dir, class_key=class_key,
            device=cfg.experiment.device,
            objective_cfg=objective_cfg,
        )
    raise ValueError(f"Unknown optimizer: {name}")
