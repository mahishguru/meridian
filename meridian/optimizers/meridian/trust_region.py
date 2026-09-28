"""TuRBO-style trust region for MERIDIAN, operating in subspace coordinates.

The trust region is a box ``{|xi| <= L_t/2}`` in the k-D subspace defined by
:class:`SubspaceManager`. The success/failure cadence is the standard TuRBO
recipe:

  * 3 consecutive batch improvements → grow ``L *= 2``
  * ``ceil(k/q)`` consecutive non-improvements → shrink ``L /= 2``
  * ``L < L_min`` → restart from a fresh class-pool sample, increment
    ``restart_count``.

The class is intentionally a thin state container so :class:`MeridianOptimizer`
can drive it without coupling to the surrogate or the subspace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TRConfig:
    L_init: float = 0.8
    L_min: float = 0.5e-2
    L_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance_factor: float = 1.0  # tau_fail = ceil(k / q * factor)
    failure_tolerance_max: int = 8         # absolute cap on tau_fail (high-d safety)
    max_restarts: int = 3


class TrustRegion:
    def __init__(self, cfg: TRConfig) -> None:
        self.cfg = cfg
        self.L: float = float(cfg.L_init)
        self.success_count: int = 0
        self.failure_count: int = 0
        self.best_value: float = -float("inf")
        self.restart_triggered: bool = False
        self.restart_count: int = 0

    def update(self, y_batch: np.ndarray, k: int, batch_size: int) -> None:
        """Apply success/failure cadence after one evaluated batch."""
        if len(y_batch) == 0:
            return
        improved = float(np.max(y_batch)) > self.best_value + 1e-3 * abs(self.best_value)
        if improved:
            self.success_count += 1
            self.failure_count = 0
        else:
            self.failure_count += 1
            self.success_count = 0

        tau_fail_raw = max(1, int(math.ceil(k / max(batch_size, 1) * self.cfg.failure_tolerance_factor)))
        tau_fail = min(tau_fail_raw, max(1, int(self.cfg.failure_tolerance_max)))
        if self.success_count >= self.cfg.success_tolerance:
            self.L = min(2.0 * self.L, self.cfg.L_max)
            self.success_count = 0
        elif self.failure_count >= tau_fail:
            self.L /= 2.0
            self.failure_count = 0

        self.best_value = max(self.best_value, float(np.max(y_batch)))
        self.restart_triggered = self.L < self.cfg.L_min

    def reset_after_restart(self) -> None:
        """Reset trust-region state after a cold-start from the class pool."""
        self.L = float(self.cfg.L_init)
        self.success_count = 0
        self.failure_count = 0
        self.restart_triggered = False
        self.restart_count += 1
        self.best_value = -float("inf")  # reset so the new basin can "succeed"

    @property
    def converged(self) -> bool:
        return self.restart_count >= self.cfg.max_restarts
