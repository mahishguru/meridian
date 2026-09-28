"""Per-variant texture-prior head: z -> predicted ODF histogram.

Loads the frozen ``head.pt`` produced by
``microstructure_ed/eval/texture_prior.py`` (microstructure-encoder-decoder)
(keys: ``state`` / ``K`` / ``in_dim``) together with the shared S-space
k-means ``codebook.npy``. The predicted histogram drives the mandatory
whole-grain Sinkhorn repaint (calibPW) in ``codec.recon_to_dream3d``.

The MLP below replicates ``texture_prior.HistHead`` exactly (hidden=1024,
dropout inert at eval); the head is tiny so CPU inference is used to keep
the GPU free for the decoder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class _HistHead(torch.nn.Module):
    def __init__(self, in_dim: int, K: int, hidden: int = 1024, p: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(p),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, K),
        )

    def forward(self, z):
        return self.net(z)


class TexturePrior:
    """z -> softmax histogram over the S-space codebook (CPU, lazy-loaded)."""

    def __init__(self, prior_dir: str | Path) -> None:
        self.prior_dir = Path(prior_dir)
        head_pt = self.prior_dir / "head.pt"
        codebook = self.prior_dir / "codebook.npy"
        if not head_pt.is_file():
            raise FileNotFoundError(f"texture prior head not found: {head_pt}")
        if not codebook.is_file():
            raise FileNotFoundError(f"texture prior codebook not found: {codebook}")
        blob = torch.load(head_pt, map_location="cpu", weights_only=False)
        self.in_dim = int(blob["in_dim"])
        self.K = int(blob["K"])
        self.head = _HistHead(self.in_dim, self.K)
        self.head.load_state_dict(blob["state"])
        self.head.eval()
        self.codebook_npy = codebook

    @torch.no_grad()
    def predict_hist(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(1, -1)
        if z.shape[1] != self.in_dim:
            raise ValueError(
                f"latent dim {z.shape[1]} != texture head in_dim {self.in_dim}")
        logits = self.head(torch.from_numpy(z))
        return torch.softmax(logits.float(), -1)[0].numpy().astype(np.float32)
