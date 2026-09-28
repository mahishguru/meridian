"""Decoder interface and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseDecoder(ABC):
    """All decoders take z in R^{B x dim} and return a list of PIL images."""

    name: str = "base"
    latent_dim: int = 512
    image_size: int = 512

    @abstractmethod
    def decode(self, z: torch.Tensor) -> list[Image.Image]: ...


def get_decoder(cfg) -> BaseDecoder:
    name = cfg.decoder.name.lower()
    if name == "mock":
        from meridian.decoders.mock import MockDecoder
        return MockDecoder(
            latent_dim=int(cfg.latent.dim),
            image_size=int(cfg.decoder.image_size),
            seed_offset=int(cfg.decoder.mock.seed_offset),
        )
    if name == "fmdit":
        from meridian.decoders.fmdit import FMDiTDecoder
        return FMDiTDecoder(cfg)
    if name in ("fmdit_768", "fmdit_1024", "fmdit_1280"):
        # Wider-bottleneck FM-DiT variants: same module, bottleneck width taken
        # from `decoder.fmdit.target_dim` (defaults to the width in the name).
        if getattr(cfg.decoder.fmdit, "target_dim", None) is None:
            cfg.decoder.fmdit.target_dim = int(name.split("_")[1])
        from meridian.decoders.fmdit import FMDiTDecoder
        return FMDiTDecoder(cfg)
    if name == "vitdit":
        from meridian.decoders.vitdit import ViTDiTDecoder
        return ViTDiTDecoder(cfg)
    raise ValueError(f"Unknown decoder: {name}")
