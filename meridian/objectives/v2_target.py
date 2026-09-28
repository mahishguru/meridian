"""Objective V2: weighted target-driven inverse design (negative distance).

    F2 = - sqrt( sum_i w_i * (p_i - target_i)^2 / |target_i| )
         - lambda_g * relu((g_min - n_grains) / g_min)

The "C" of Co-PiLOT: drives the latent toward a microstructure whose
mechanical signature matches a user-supplied target tuple
(sigma_y, n, K, sigma_u). Any property absent from the target is ignored.
The 1/|target_i| normalisation makes the weighting roughly unit-free.

The optional grain-count penalty mirrors V1's ``g_min``/``lambda_g``: the
DAMASK constitutive law is grain-size insensitive so the optimizer can
otherwise hit the targets via a few favorably-oriented large grains
(texture flukes), which is not what we want.
"""
from __future__ import annotations

import math
from typing import Mapping

from meridian.objectives import BaseObjective
from meridian.simulation.extractor import MechanicalProperties


class TargetDrivenObjective(BaseObjective):
    name = "v2"

    def __init__(
        self,
        targets: Mapping[str, float],
        weights: Mapping[str, float],
        g_min: int = 0,
        lambda_g: float = 0.0,
        g_max: int = 0,
        lambda_g_hi: float = 0.0,
        scales: "Mapping[str, float] | None" = None,
        name: str = "v2",
    ) -> None:
        self.targets = {k: float(v) for k, v in targets.items()}
        self.weights = {k: float(weights.get(k, 1.0)) for k in self.targets}
        # Residual denominator. Defaults to |target| (v2). v3 passes the seed
        # population std so that properties with different relative spreads
        # contribute comparably instead of |target| deciding the balance.
        scales = scales or {}
        self.scales = {
            k: float(scales.get(k) or (abs(v) if abs(v) > 1e-9 else 1.0))
            for k, v in self.targets.items()
        }
        self.name = name
        self.g_min = int(g_min)
        self.lambda_g = float(lambda_g)
        # Upper-band penalty: discourages "speckle decodes" whose RGB noise
        # segments into hundreds of spurious grains. Phenopowerlaw is
        # calibrated against ~150-300-grain RVEs at 300^2 @ 2um; outside
        # that band the constitutive law gives physically meaningless
        # (sigma_y, n) that the optimizer otherwise climbs into.
        self.g_max = int(g_max)
        self.lambda_g_hi = float(lambda_g_hi)

    def __call__(self, p: MechanicalProperties) -> float:
        d = p.to_dict()
        ssq = 0.0
        for key, target in self.targets.items():
            val = d.get(key, float("nan"))
            if not math.isfinite(val):
                return float("-inf")
            ssq += self.weights[key] * ((val - target) / self.scales[key]) ** 2
        score = -math.sqrt(ssq)
        if self.lambda_g > 0.0 and self.g_min > 0 and p.n_grains > 0:
            shortfall = max(self.g_min - int(p.n_grains), 0) / float(self.g_min)
            score -= self.lambda_g * shortfall
        if self.lambda_g_hi > 0.0 and self.g_max > 0 and p.n_grains > 0:
            # Saturating excess: cap at 2x g_max so a 34975-grain speckle
            # outlier doesn't blow Y to -100. Without the cap the surrogate's
            # output scale is ruled by outliers, not by the in-band region
            # we want to optimize over.
            excess = min(max(int(p.n_grains) - self.g_max, 0) / float(self.g_max), 2.0)
            score -= self.lambda_g_hi * excess
        return float(score)
