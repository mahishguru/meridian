#!/usr/bin/env python3
"""MERIDIAN on a generic black-box function (no decoder, no DAMASK).

Maximises a negated 20-D Ackley function on the box [-3, 3]^20 with batches of
4 evaluations. A region of the box is declared infeasible (the function returns
NaN there, like a crashed simulation); MERIDIAN keeps those points and learns
where evaluations fail through its feasibility head.

    python examples/standalone_blackbox.py            # CPU is fine
"""
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from meridian.config import ConfigNode, _to_node
from meridian.optimizers.meridian import MeridianOptimizer

DIM, BOX, BATCH, ROUNDS, N_INIT = 20, (-3.0, 3.0), 4, 20, 32
rng = np.random.default_rng(0)
torch.manual_seed(0)


def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2 * math.pi
    return float(-a * np.exp(-b * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(c * x))) + a + math.e)


def objective(x: np.ndarray) -> float:
    """Black box to MAXIMISE. NaN marks an infeasible evaluation."""
    if x[0] > 2.0 and x[1] > 2.0:          # e.g. a region where the simulator fails
        return float("nan")
    return -ackley(x)


# MERIDIAN hyper-parameters from the reference config; the latent-shell
# projection is specific to encoder latents and is switched off here.
cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/base.yaml").read_text())
mcfg: ConfigNode = _to_node(cfg["optimizer"]["meridian"])
mcfg.update(adaptive_shell=False, shell_radius_min=0.0, shell_radius_max=0.0,
            surrogate_epochs=100, sobol_cloud=2048, dpp_pool=128)

opt = MeridianOptimizer(dim=DIM, bounds=BOX, batch_size=BATCH, meridian_cfg=mcfg,
                        device="cuda" if torch.cuda.is_available() else "cpu")

X0 = rng.uniform(*BOX, size=(N_INIT, DIM)).astype(np.float32)
Y0 = np.array([objective(x) for x in X0], dtype=np.float32)
opt.update(X0, Y0)
print(f"initial design: best f = {np.nanmax(Y0):.3f}  ({np.isnan(Y0).sum()} infeasible)")

for r in range(1, ROUNDS + 1):
    X = opt.suggest()                                   # (BATCH, DIM)
    Y = np.array([objective(x) for x in X], dtype=np.float32)
    opt.update(X, Y)
    feas = opt.Y[opt.feasible_mask]
    print(f"round {r:2d}: batch best {np.nanmax(Y) if np.isfinite(Y).any() else float('nan'):8.3f} | "
          f"overall best {feas.max():8.3f} | trust-region L = {opt.trust.L:.3f}")

best = int(np.argmax(np.where(opt.feasible_mask, opt.Y, -np.inf)))
print(f"best feasible f = {opt.Y[best]:.3f} at |x| = {np.linalg.norm(opt.X[best]):.2f}")
