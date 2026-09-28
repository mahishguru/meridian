"""Dry-run mock simulator: synthesizes a stress-strain response from z.

Used when `loop.dry_run: true` to debug the full pipeline without DAMASK.
The mock is deterministic in z and exposes a smooth nonlinear landscape
so that the optimizers have something meaningful to climb.
"""
from __future__ import annotations

import math

import numpy as np

from meridian.simulation import SimulationOutcome
from meridian.simulation.extractor import MechanicalProperties


def mock_simulate(z: np.ndarray) -> SimulationOutcome:
    """Map z directly to plausible mechanical properties (no DAMASK call)."""
    z = np.asarray(z, dtype=np.float32).ravel()
    # 4 latent "directions" drive the 4 properties; everything else is decorative noise.
    a = float(np.tanh(z[:32].mean()))
    b = float(np.tanh(z[32:64].mean()))
    c = float(np.tanh(z[64:96].mean()))
    d = float(np.tanh(z[96:128].mean()))

    sigma_y = 220.0 + 60.0 * a + 5.0 * math.sin(b * 4)
    sigma_u = sigma_y + 70.0 + 30.0 * c
    n = 0.18 + 0.05 * d + 0.02 * a
    K = sigma_u * (1.0 + 0.5 * n)
    eps_uniform = max(0.05, 0.18 + 0.05 * b - 0.03 * a)
    E = 45000.0
    eps_y = sigma_y / E
    eps_f = eps_uniform + 0.02
    sigma_f = sigma_u * 0.85
    work_uniform = 0.5 * (sigma_y + sigma_u) * eps_uniform
    work_fracture = work_uniform + 0.5 * (sigma_u + sigma_f) * max(eps_f - eps_uniform, 0.0)
    plastic_work_uniform = max(work_uniform - 0.5 * sigma_y * eps_y, 0.0)
    plastic_work_fracture = max(work_fracture - 0.5 * sigma_y * eps_y, 0.0)

    props = MechanicalProperties(
        E=E, sigma_y=sigma_y, epsilon_y=eps_y,
        sigma_u=sigma_u, epsilon_uniform=eps_uniform,
        K=K, n=n, epsilon_fracture=eps_f, sigma_fracture=sigma_f,
        work_uniform=work_uniform, work_fracture=work_fracture,
        plastic_work_uniform=plastic_work_uniform,
        plastic_work_fracture=plastic_work_fracture,
    )
    return SimulationOutcome(success=True, properties=props, sim_dir=None)
