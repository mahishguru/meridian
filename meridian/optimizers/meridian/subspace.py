"""Subspace / directional-weight estimation for MERIDIAN.

Provides two modes behind a uniform interface:

  1. **Legacy subspace mode** (``full_dim_search=False``): candidates are
     generated in a k-D subspace and lifted to d-D via ``project`` / ``lift``.
  2. **Full-dim mode** (``full_dim_search=True``, default): the eigenspectrum
     is converted to per-ambient-axis importance weights ``w_ambient`` that
     shape a TuRBO-style anisotropic trust region in all d dimensions.

Both modes share the same eigendecomposition back-end:

  * **PCA** on the per-class pre-encoded latent pool — used as a warm-start
    for the first few rounds while the surrogate gradient is unreliable.
  * **Active subspace** from the surrogate's gradient covariance,
    ``C_hat = (1/N) sum_i ∇mu(z_i) ∇mu(z_i)^T`` — used once enough data is
    available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _pick_k(eig: np.ndarray, energy: float, k_min: int, k_max: int) -> int:
    """Smallest k whose cumulative spectral energy >= ``energy``, clipped."""
    eig = np.maximum(eig, 0.0)
    if eig.sum() <= 0:
        return k_min
    cum = np.cumsum(eig) / eig.sum()
    k = int(np.searchsorted(cum, energy) + 1)
    return int(np.clip(k, k_min, k_max))


def _eigenweights_to_ambient(
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    w_floor: float,
    energy: float,
    k_min: int,
    k_max: int,
) -> tuple[np.ndarray, int]:
    """Convert an eigendecomposition to per-axis ambient weights.

    Parameters
    ----------
    eigvecs : (d, rank) orthonormal columns (PCA right-singular vectors or
              eigh eigenvectors, ordered descending by eigenvalue).
    eigvals : (rank,) eigenvalues in descending order.
    w_floor : minimum relative weight for any direction (prevents zero
              exploration along "unimportant" axes).

    Returns
    -------
    w_ambient : (d,) per-axis trust-region weights, geometric-mean normalised.
    k_eff     : effective dimensionality (spectral energy threshold).
    """
    k_eff = _pick_k(eigvals, energy, k_min, k_max)

    # Eigenweights: sqrt(λ)/max, floored so no axis is fully dead
    w_eig = np.sqrt(np.maximum(eigvals, 0.0))
    w_eig = w_eig / (w_eig.max() + 1e-12)
    w_eig = np.maximum(w_eig, w_floor)

    # Map to ambient axes: w_j = || V[j, :] ⊙ w_eig ||_2
    d = eigvecs.shape[0]
    rank = eigvecs.shape[1]
    w_ambient = np.sqrt((eigvecs ** 2) @ (w_eig[:rank] ** 2))

    # Geometric-mean normalise (TuRBO convention) so that product of weights
    # is invariant to uniform rescaling of eigenvalues.
    log_w = np.log(np.maximum(w_ambient, 1e-12))
    w_ambient = w_ambient / np.exp(np.mean(log_w))

    return w_ambient.astype(np.float32), k_eff


def pca_basis(
    Z: np.ndarray,
    energy: float = 0.95,
    k_min: int = 8,
    k_max: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA on a (N, d) latent pool. Returns (Q, s) with shapes (d, k), (k,)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # Use SVD for numerical stability; Zc = U S V^T with V columns = eigenvectors
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    eig = (S ** 2) / max(len(Z) - 1, 1)
    k = _pick_k(eig, energy, k_min, k_max)
    Q = Vt[:k].T.astype(np.float32)              # (d, k)
    s = np.sqrt(eig[:k]).astype(np.float32)      # per-axis std-dev
    # Normalise so that the largest s is 1 — consistent with active_subspace()
    s = s / (s.max() + 1e-12)
    return Q, s


def active_subspace(
    grads: np.ndarray,
    energy: float = 0.95,
    k_min: int = 8,
    k_max: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Active subspace from a (N, d) array of gradients.

    Eigendecomposition of C = (1/N) G^T G yields directions where the
    surrogate posterior mean changes most rapidly.
    """
    N, d = grads.shape
    if N == 0:
        raise ValueError("Need at least one gradient sample.")
    # Symmetric eigendecomposition (descending order)
    C = (grads.T @ grads) / N
    eig, V = np.linalg.eigh(C)
    eig, V = eig[::-1], V[:, ::-1]
    k = _pick_k(eig, energy, k_min, k_max)
    Q = V[:, :k].astype(np.float32)              # (d, k)
    s = np.sqrt(np.maximum(eig[:k], 1e-8)).astype(np.float32)
    # Normalise so that the largest s is 1 — keeps L_t in a sensible range
    s = s / (s.max() + 1e-12)
    return Q, s


@dataclass
class SubspaceConfig:
    energy: float = 0.95
    k_min: int = 8
    k_max: int = 24
    pca_warmstart_rounds: int = 3
    w_floor: float = 0.1
    seeds_dir: str | None = None
    class_key: str | None = None


class SubspaceManager:
    """Holds eigendecomposition state and provides ambient weights."""

    def __init__(self, cfg: SubspaceConfig) -> None:
        self.cfg = cfg
        self.Q: np.ndarray | None = None        # (d, k)
        self.s: np.ndarray | None = None        # (k,)
        self._init_pool: np.ndarray | None = None
        # Full-dim mode: per-axis ambient weights for anisotropic TR
        self.w_ambient: np.ndarray | None = None  # (d,)
        self._effective_k: int = 0

    def warmstart_from_pool(self, dim: int) -> None:
        """Initialise ``(Q, s)`` from the class-conditional latent pool."""
        if self.cfg.seeds_dir is None or self.cfg.class_key is None:
            k = self.cfg.k_min
            Q = np.zeros((dim, k), dtype=np.float32)
            np.fill_diagonal(Q, 1.0)
            self.Q, self.s = Q, np.ones(k, dtype=np.float32)
            self.w_ambient = np.ones(dim, dtype=np.float32)
            self._effective_k = k
            return
        pool_path = Path(self.cfg.seeds_dir) / f"{self.cfg.class_key}.npy"
        if not pool_path.is_file():
            k = self.cfg.k_min
            Q = np.zeros((dim, k), dtype=np.float32)
            np.fill_diagonal(Q, 1.0)
            self.Q, self.s = Q, np.ones(k, dtype=np.float32)
            self.w_ambient = np.ones(dim, dtype=np.float32)
            self._effective_k = k
            return
        pool = np.load(pool_path).astype(np.float32)
        self._init_pool = pool
        self.Q, self.s = pca_basis(
            pool,
            energy=self.cfg.energy,
            k_min=self.cfg.k_min,
            k_max=self.cfg.k_max,
        )
        self._compute_ambient_weights_pca(pool)

    def _compute_ambient_weights_pca(self, pool: np.ndarray) -> None:
        """Full-rank PCA → per-axis ambient weights."""
        Zc = pool - pool.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        eig = (S ** 2) / max(len(pool) - 1, 1)
        V = Vt.T  # (d, rank)
        self.w_ambient, self._effective_k = _eigenweights_to_ambient(
            V, eig, self.cfg.w_floor,
            self.cfg.energy, self.cfg.k_min, self.cfg.k_max,
        )

    def _compute_ambient_weights_active(self, grads: np.ndarray) -> None:
        """Active-subspace gradients → per-axis ambient weights."""
        N, d = grads.shape
        C = (grads.T @ grads) / N
        eig, V = np.linalg.eigh(C)
        eig, V = eig[::-1], V[:, ::-1]
        self.w_ambient, self._effective_k = _eigenweights_to_ambient(
            V, eig, self.cfg.w_floor,
            self.cfg.energy, self.cfg.k_min, self.cfg.k_max,
        )

    def maybe_refit(self, round_idx: int, grads: np.ndarray | None) -> None:
        """Switch to active-subspace once we have a trained surrogate."""
        if round_idx < self.cfg.pca_warmstart_rounds or grads is None or len(grads) < 4:
            return
        try:
            Q_new, s_new = active_subspace(
                grads,
                energy=self.cfg.energy,
                k_min=self.cfg.k_min,
                k_max=self.cfg.k_max,
            )
            self.Q, self.s = Q_new, s_new
            self._compute_ambient_weights_active(grads)
        except np.linalg.LinAlgError:
            pass

    @property
    def k(self) -> int:
        return 0 if self.Q is None else int(self.Q.shape[1])

    @property
    def effective_k(self) -> int:
        return self._effective_k if self._effective_k > 0 else self.k

    def project(self, z: np.ndarray) -> np.ndarray:
        """z (N, d) -> xi (N, k)."""
        return z @ self.Q

    def lift(self, xi: np.ndarray) -> np.ndarray:
        """xi (N, k) -> z (N, d) via Q."""
        return xi @ self.Q.T
