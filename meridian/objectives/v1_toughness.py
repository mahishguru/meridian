"""Objective V1: dimensionless work-weighted strength.

    F1 = (sigma_y / sigma_y_ref)^alpha * (work / work_ref)^beta
         * (sigma_u / sigma_u_ref)^gamma
         - lambda_n * relu(n_min - n)
         - lambda_g * relu((g_min - n_grains) / g_min)

Rewards a microstructure that is simultaneously *strong* (high yield) and
*tough* (large area under the stress-strain curve, preferably plastic work
up to uniform strain). This replaces the legacy ``sigma_u*epsilon_uniform``
term, which was weakly discriminative because ``epsilon_uniform`` is nearly
constant in the AZ31 runs. Older cached properties that lack the work fields
fall back to ``sigma_u*epsilon_uniform`` so legacy logs remain readable.
"""
from __future__ import annotations

import math

from meridian.objectives import BaseObjective
from meridian.simulation.extractor import MechanicalProperties


class ToughnessObjective(BaseObjective):
    name = "v1"

    def __init__(
        self,
        sigma_y_ref: float,
        toughness_ref: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        sigma_u_ref: float = 315.0,
        gamma: float = 0.25,
        work_key: str = "",
        n_min: float = 0.10,
        lambda_n: float = 5.0,
        g_min: int = 0,
        lambda_g: float = 0.0,
        g_max: int = 0,
        lambda_g_hi: float = 0.0,
    ) -> None:
        self.sigma_y_ref = sigma_y_ref
        self.toughness_ref = toughness_ref
        self.alpha = alpha
        self.beta = beta
        self.sigma_u_ref = sigma_u_ref
        self.gamma = gamma
        self.work_key = str(work_key)
        self.n_min = n_min
        self.lambda_n = lambda_n
        self.g_min = int(g_min)
        self.lambda_g = float(lambda_g)
        # Upper-band penalty (see TargetDrivenObjective for rationale).
        self.g_max = int(g_max)
        self.lambda_g_hi = float(lambda_g_hi)

    def __call__(self, p: MechanicalProperties) -> float:
        if not (math.isfinite(p.sigma_y) and math.isfinite(p.sigma_u) and math.isfinite(p.epsilon_uniform)):
            return float("-inf")
        strength = max(p.sigma_y, 0.0) / self.sigma_y_ref
        work = float(getattr(p, self.work_key, float("nan")))
        if not math.isfinite(work) or work <= 0.0:
            # Backward-compatible fallback for old seed-cache properties.
            work = max(p.sigma_u * p.epsilon_uniform, 0.0)
        toughness = max(work, 0.0) / self.toughness_ref
        uts = max(p.sigma_u, 0.0) / self.sigma_u_ref
        score = (strength ** self.alpha) * (toughness ** self.beta)
        if self.gamma != 0.0:
            score *= uts ** self.gamma
        if math.isfinite(p.n):
            score -= self.lambda_n * max(self.n_min - p.n, 0.0)
        if self.lambda_g > 0.0 and self.g_min > 0 and p.n_grains > 0:
            shortfall = max(self.g_min - int(p.n_grains), 0) / float(self.g_min)
            score -= self.lambda_g * shortfall
        if self.lambda_g_hi > 0.0 and self.g_max > 0 and p.n_grains > 0:
            # Saturating excess: cap at 2x g_max (see v2_target.py for rationale).
            excess = min(max(int(p.n_grains) - self.g_max, 0) / float(self.g_max), 2.0)
            score -= self.lambda_g_hi * excess
        return float(score)
