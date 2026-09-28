"""MERIDIAN — Manifold-Embedded Robust Inverse Design via Iterative Acquisition Networks.

A successor to DANTE (Wei et al. 2025) tailored for high-dimensional latent
inverse design. Single committed configuration — see ``optimizer.py`` for
algorithm details and ``../../theory.md`` §7 for theoretical justification.
"""
from meridian.optimizers.meridian.optimizer import MeridianOptimizer

__all__ = ["MeridianOptimizer"]
