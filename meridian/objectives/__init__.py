"""Objective function interface and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

from meridian.simulation.extractor import MechanicalProperties


class BaseObjective(ABC):
    """All objectives are MAXIMIZED. Higher = better."""

    name: str = "base"

    @abstractmethod
    def __call__(self, props: MechanicalProperties) -> float: ...


def get_objective(cfg) -> BaseObjective:
    name = cfg.objective.name.lower()
    if name == "v1":
        from meridian.objectives.v1_toughness import ToughnessObjective
        c = cfg.objective.v1
        return ToughnessObjective(
            sigma_y_ref=float(c.sigma_y_ref),
            toughness_ref=float(c.toughness_ref),
            alpha=float(c.alpha),
            beta=float(c.beta),
            sigma_u_ref=float(getattr(c, "sigma_u_ref", 315.0)),
            gamma=float(getattr(c, "gamma", 0.25)),
            work_key=str(getattr(c, "work_key", "")),
            n_min=float(c.n_min),
            lambda_n=float(c.lambda_n),
            g_min=int(getattr(c, "g_min", 0)),
            lambda_g=float(getattr(c, "lambda_g", 0.0)),
            g_max=int(getattr(c, "g_max", 0)),
            lambda_g_hi=float(getattr(c, "lambda_g_hi", 0.0)),
        )
    if name == "v2":
        from meridian.objectives.v2_target import TargetDrivenObjective
        c = cfg.objective.v2
        return TargetDrivenObjective(
            targets=dict(c.targets),
            weights=dict(c.weights),
            g_min=int(getattr(c, "g_min", 0)),
            lambda_g=float(getattr(c, "lambda_g", 0.0)),
            g_max=int(getattr(c, "g_max", 0)),
            lambda_g_hi=float(getattr(c, "lambda_g_hi", 0.0)),
        )
    if name == "v3":
        from meridian.objectives.v2_target import TargetDrivenObjective
        c = cfg.objective.v3
        scales = dict(getattr(c, "scales", {}) or {})
        if not scales:
            raise ValueError("objective.v3 requires explicit 'scales'")
        return TargetDrivenObjective(
            targets=dict(c.targets),
            weights=dict(c.weights),
            scales=scales,
            g_min=int(getattr(c, "g_min", 0)),
            lambda_g=float(getattr(c, "lambda_g", 0.0)),
            g_max=int(getattr(c, "g_max", 0)),
            lambda_g_hi=float(getattr(c, "lambda_g_hi", 0.0)),
            name="v3",
        )
    raise ValueError(f"Unknown objective: {name}")
