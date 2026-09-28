"""MockDecoder: deterministic placeholder used until real decoders ship.

Generates smoothly-varying RGB images conditioned on z so the full
optimization loop is end-to-end testable without any pretrained weights.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from meridian.decoders import BaseDecoder


class MockDecoder(BaseDecoder):
    name = "mock"

    def __init__(self, latent_dim: int = 512, image_size: int = 512, seed_offset: int = 0) -> None:
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.seed_offset = seed_offset

    def decode(self, z: torch.Tensor) -> list[Image.Image]:
        z_np = z.detach().cpu().numpy().astype(np.float32)
        H = W = self.image_size
        images: list[Image.Image] = []
        for i, zi in enumerate(z_np):
            seed = int(abs(np.tanh(zi).sum() * 1e6)) + self.seed_offset + i
            rng = np.random.default_rng(seed)

            # 3 low-frequency Gaussian fields blended with z-driven sinusoids.
            base = rng.normal(size=(8, 8, 3)).astype(np.float32)
            base_norm = (base - base.min()) / (np.ptp(base) + 1e-9)
            img = np.array(
                Image.fromarray((base_norm * 255).astype(np.uint8)).resize(
                    (W, H), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            )

            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / max(H, W)
            for k in range(3):
                freq = 2.0 + 8.0 * float(np.tanh(zi[k % len(zi)]))
                phase = float(zi[(k + 1) % len(zi)])
                img[..., k] += 40.0 * np.sin(2 * np.pi * freq * xx + phase) * np.cos(
                    2 * np.pi * freq * yy + phase
                )

            img = np.clip(img, 0, 255).astype(np.uint8)
            images.append(Image.fromarray(img, mode="RGB"))
        return images
