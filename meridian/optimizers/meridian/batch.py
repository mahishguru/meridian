"""Acquisition + diverse batch selection for MERIDIAN.

  * **Acquisition** — qLogNoisyExpectedImprovement on the DKL posterior over
    a Sobol cloud of candidates inside the trust region. Numerically stable
    in regions of high posterior, batch-aware, noise-tolerant.
  * **Feasibility multiplier** — acquisition is multiplied by the classifier
    probability ``g_psi(z)`` so the optimiser avoids regions where the
    decode/codec/DAMASK pipeline is likely to fail.
  * **Greedy-DPP batch** — picks ``q`` candidates that are simultaneously
    high-acquisition and mutually diverse (under the GP feature kernel). This
    avoids spending two near-identical DAMASK runs in the same batch.
"""
from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement


def acquisition_scores(
    Z_cand: np.ndarray,
    surrogate,
) -> np.ndarray:
    """qLogNEI scores on the DKL posterior, multiplied by feasibility prob."""
    feats = surrogate.features(Z_cand)
    # Posterior best baseline = best feasible feature embedding
    if surrogate.state.gp is None or feats.shape[0] == 0:
        return np.zeros(len(Z_cand), dtype=np.float32)

    # X_baseline must be on the same device/dtype as the model
    train_inputs = surrogate.state.gp.train_inputs[0]
    acq = qLogNoisyExpectedImprovement(
        model=surrogate.state.gp,
        X_baseline=train_inputs,
        prune_baseline=False,
    )
    # qLogNEI expects a (q=1, d) batch; we score one candidate at a time via vmap-style call
    with torch.no_grad():
        # Vectorise over candidates by treating each as q=1
        scores = acq(feats.unsqueeze(1)).cpu().numpy()
    feas = surrogate.feasibility(Z_cand)
    # Combine in log-space to remain monotone with qLogNEI
    return scores + np.log(np.clip(feas, 1e-6, 1.0))


def greedy_dpp_select(
    Z_cand: np.ndarray,
    scores: np.ndarray,
    surrogate,
    q: int,
    pool_size: int = 256,
    z_kernel_weight: float = 0.0,
    z_kernel_lengthscale: float = 1.0,
) -> np.ndarray:
    """Greedy DPP MAP selection on the top-``pool_size`` candidates.

    Quality-weighted DPP kernel: ``L_ij = q_i * q_j * k(z_i, z_j)`` where
    ``q_i = exp(scores_i / temperature)`` and ``k`` is the GP feature kernel.

    A4-DPP hybrid kernel
    --------------------
    When ``z_kernel_weight > 0`` the kernel becomes
        K = (1 - w) * k_phi(phi_i, phi_j) + w * k_z(z_i, z_j)
    where ``k_z`` is an RBF on raw latents with bandwidth
    ``z_kernel_lengthscale``. This protects diversity when the feature
    kernel collapses (saturated heads, identical phi) by adding a spatial
    diversity term in z-space.
    """
    n = len(Z_cand)
    if n == 0:
        return np.empty((0, Z_cand.shape[1]), dtype=Z_cand.dtype)
    if q >= n:
        return Z_cand.copy()

    # 1. Restrict to top-pool by acquisition score
    order = np.argsort(scores)[::-1][:min(pool_size, n)]
    Z_top = Z_cand[order]
    s_top = scores[order]

    # 2. Compute pairwise kernel matrix on phi(z) features
    feats = surrogate.features(Z_top)
    kernel = surrogate.state.gp.covar_module
    with torch.no_grad():
        K = kernel(feats).evaluate().cpu().numpy()
    K = 0.5 * (K + K.T)

    # A4-DPP: blend in an RBF on raw latents to protect spatial diversity
    # when phi-space collapses (saturated heads -> near-identical features).
    if z_kernel_weight > 0.0:
        diff = Z_top[:, None, :] - Z_top[None, :, :]
        sqd = np.sum(diff * diff, axis=-1)
        ls = max(float(z_kernel_lengthscale), 1e-6)
        Kz = np.exp(-0.5 * sqd / (ls * ls))
        K = (1.0 - z_kernel_weight) * K + z_kernel_weight * Kz

    # 3. Quality weights — soft-temper to keep numerics sane
    s_norm = (s_top - s_top.max()) / (s_top.std() + 1e-6)
    quality = np.exp(s_norm)
    L = K * np.outer(quality, quality)

    # 4. Greedy MAP (k-DPP): pick argmax marginal-gain, repeat q times
    selected: list[int] = []
    remaining = list(range(len(Z_top)))
    log_det = -np.inf
    L_inv_sub = None  # placeholder; we recompute det incrementally below

    for _ in range(q):
        best_gain = -np.inf
        best_idx = -1
        for j in remaining:
            cand = selected + [j]
            sub = L[np.ix_(cand, cand)]
            sign, logdet = np.linalg.slogdet(sub + 1e-9 * np.eye(len(cand)))
            if sign <= 0:
                continue
            gain = logdet - log_det if log_det != -np.inf else logdet
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        sub = L[np.ix_(selected, selected)]
        _, log_det = np.linalg.slogdet(sub + 1e-9 * np.eye(len(selected)))

    return Z_top[selected]
