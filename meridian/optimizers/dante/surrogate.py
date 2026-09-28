"""DNN surrogate for DANTE.

Lightweight MLP with LayerNorm + GELU + dropout. Trained from scratch
each round on the cumulative dataset (paper-faithful: small data, full
re-fit, no online updates).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPSurrogate(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DNNSurrogate:
    """Wraps an MLP with simple fit/predict semantics."""

    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 32,
        dropout: float = 0.1,
        val_split: float = 0.1,
        early_stop_patience: int = 20,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = MLPSurrogate(in_dim, hidden, dropout=dropout).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = early_stop_patience
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        self.model = MLPSurrogate(
            X.shape[1],
            [m.out_features for m in self.model.net if isinstance(m, nn.Linear)][:-1],
            dropout=next((m.p for m in self.model.net if isinstance(m, nn.Dropout)), 0.1),
        ).to(self.device)

        self._y_mean = float(np.mean(Y))
        self._y_std = float(np.std(Y) + 1e-6)
        Yn = (Y - self._y_mean) / self._y_std

        n = len(X)
        idx = np.random.permutation(n)
        n_val = max(1, int(self.val_split * n))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr = torch.from_numpy(X[tr_idx]).float().to(self.device)
        Ytr = torch.from_numpy(Yn[tr_idx]).float().to(self.device)
        Xv = torch.from_numpy(X[val_idx]).float().to(self.device)
        Yv = torch.from_numpy(Yn[val_idx]).float().to(self.device)

        loader = DataLoader(
            TensorDataset(Xtr, Ytr), batch_size=min(self.batch_size, len(Xtr)), shuffle=True
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        bad = 0
        history = []
        for ep in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                v = float(loss_fn(self.model(Xv), Yv).item())
            history.append(v)
            if v < best_val - 1e-5:
                best_val = v
                best_state = {k: v_.detach().clone() for k, v_ in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        return {"val_loss": best_val, "epochs": len(history)}

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.from_numpy(np.atleast_2d(X)).float().to(self.device)
        y = self.model(x).cpu().numpy()
        return y * self._y_std + self._y_mean
