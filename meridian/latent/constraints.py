"""Latent constraints: hard box clamp + optional soft physics penalty."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class BoxConstraint:
    """Symmetric box clamp on latent vectors."""

    def __init__(self, low: float = -3.0, high: float = 3.0) -> None:
        if low >= high:
            raise ValueError("low must be < high")
        self.low = float(low)
        self.high = float(high)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low, self.high

    def clamp_np(self, z: np.ndarray) -> np.ndarray:
        return np.clip(z, self.low, self.high)

    def clamp(self, z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z, self.low, self.high)


class PhysicsPenalty:
    """Soft penalty on degenerate decoded images (e.g. uniform gray)."""

    def __init__(self, min_pixel_std: float = 5.0, weight: float = 10.0) -> None:
        self.min_pixel_std = float(min_pixel_std)
        self.weight = float(weight)

    def __call__(self, image: Image.Image) -> float:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
        std = float(arr.std())
        if std >= self.min_pixel_std:
            return 0.0
        return -self.weight * (self.min_pixel_std - std) / self.min_pixel_std
