"""DANTE outer optimizer: DNN surrogate + NTE + closed-loop driver."""
from __future__ import annotations

import numpy as np

from meridian.optimizers import BaseOptimizer
from meridian.optimizers.dante.nte import NeuralTreeExplorer, NTEConfig
from meridian.optimizers.dante.surrogate import DNNSurrogate


class DANTEOptimizer(BaseOptimizer):
    name = "dante"

    def __init__(self, dim, bounds, batch_size, dante_cfg, device: str = "cuda") -> None:
        super().__init__(dim, bounds, batch_size)
        self.surrogate = DNNSurrogate(
            in_dim=dim,
            hidden=list(dante_cfg.surrogate_hidden),
            lr=float(dante_cfg.surrogate_lr),
            weight_decay=float(dante_cfg.surrogate_weight_decay),
            epochs=int(dante_cfg.surrogate_epochs),
            batch_size=int(dante_cfg.surrogate_batch_size),
            dropout=float(dante_cfg.surrogate_dropout),
            val_split=float(dante_cfg.val_split),
            early_stop_patience=int(dante_cfg.early_stop_patience),
            device=device,
        )
        self.nte = NeuralTreeExplorer(
            NTEConfig(
                n_rollouts=int(dante_cfg.n_rollouts),
                n_leaves_per_expand=int(dante_cfg.n_leaves_per_expand),
                sigma_init=float(dante_cfg.sigma_init),
                sigma_decay=float(dante_cfg.sigma_decay),
                c0=float(dante_cfg.c0),
                rho_smoothing=float(dante_cfg.rho_smoothing),
                bounds=(self.low, self.high),
            )
        )
        self._needs_refit = True
        # Optional explicit anchor for the NTE root. When set, replaces the
        # default argmax(Y) seed for the FIRST suggest() call. After at least
        # one update() with new observations, we revert to argmax(Y) so the
        # tree follows the optimization frontier.
        self._initial_seed: np.ndarray | None = None
        self._initial_seed_used: bool = False

    def set_initial_seed(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if z.shape[0] != self.dim:
            raise ValueError(f"initial seed dim {z.shape[0]} != optimizer dim {self.dim}")
        self._initial_seed = z
        self._initial_seed_used = False

    def _on_update(self, X_new, Y_new) -> None:
        self._needs_refit = True
        # Once new observations come in, hand control back to argmax(Y).
        self._initial_seed_used = True

    def suggest(self) -> np.ndarray:
        if len(self.X) == 0:
            raise RuntimeError("DANTE requires seed data before suggesting.")
        if self._needs_refit:
            self.surrogate.fit(self.X, self.Y)
            self._needs_refit = False
        if self._initial_seed is not None and not self._initial_seed_used:
            seed = self._initial_seed
            self._initial_seed_used = True
        else:
            seed = self.X[int(np.argmax(self.Y))]
        return self.nte.search(
            seed_z=seed,
            surrogate=self.surrogate,
            n_candidates=self.batch_size,
            observed_values=self.Y,
        )
