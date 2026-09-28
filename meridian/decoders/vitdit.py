"""ViT-conditioned DiT-XL decoder wrapper (DDPM).

Loads the trained `ViTDiTDecoder` (DiT-XL/2 backbone + Spatial Cross-Attention
hijack + 4-channel SD VAE) of the `microstructure-encoder-decoder` package
(baseline `microstructure_ed.baselines.vitdit`).

Mirrors the FMDiT wrapper layout: checkpoint is a dict with keys
    {"epoch", "encoder", "decoder", "optimizer", "lr_scheduler", ...}
where the `decoder` entry holds the ViTDiTDecoder state_dict.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from meridian.decoders import BaseDecoder


class ViTDiTDecoder(BaseDecoder):
    name = "vitdit"

    def __init__(self, cfg) -> None:
        ckpt = cfg.decoder.vitdit.checkpoint
        if ckpt is None:
            raise NotImplementedError(
                "ViTDiTDecoder requires `decoder.vitdit.checkpoint` to point at a trained\n"
                "ViTDiTDecoder checkpoint .pth (with 'decoder' state_dict)."
            )
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"ViTDiT checkpoint not found: {ckpt}")

        from microstructure_ed.baselines.vitdit.decoder_arch_pretrained import (
            ViTDiTDecoder as _ViTDiTDecoder, SDVAE,
        )

        device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_steps = int(cfg.decoder.num_inference_steps)
        self.image_size = int(cfg.decoder.image_size)
        self.batch_size = int(getattr(cfg.decoder.vitdit, "decode_batch_size", 4))

        dit_pretrained = getattr(
            cfg.decoder.vitdit, "dit_pretrained", "facebook/DiT-XL-2-256"
        )
        sd_vae_model_id = getattr(
            cfg.decoder.vitdit, "sd_vae_model_id", "stabilityai/sd-vae-ft-mse"
        )

        print(f"[vitdit] loading DiT-XL/2 backbone ({dit_pretrained})")
        self.model = _ViTDiTDecoder(dit_checkpoint=dit_pretrained)
        self.vae = SDVAE(model_id=sd_vae_model_id)

        print(f"[vitdit] loading decoder weights from {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        dec_state = state.get("decoder", state)
        # `trainer.py` may save a DDP-wrapped state_dict whose keys are prefixed
        # with "module."; strip it so non-DDP loads cleanly.
        if any(k.startswith("module.") for k in dec_state):
            dec_state = {k.removeprefix("module."): v for k, v in dec_state.items()}

        missing, unexpected = self.model.load_state_dict(dec_state, strict=False)
        n_loaded = max(0, len(dec_state) - len(unexpected))
        print(f"[vitdit] loaded {n_loaded} tensors "
              f"({len(unexpected)} unexpected, {len(missing)} missing)")

        self.model.to(self.device).eval()
        self.vae.to(self.device)

    @torch.no_grad()
    def decode(self, z):
        """z: (B, 512) -> list of PIL.Image (image_size x image_size, RGB)."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device, dtype=torch.float32)

        images = []
        for start in range(0, z.shape[0], self.batch_size):
            chunk = z[start:start + self.batch_size]
            latents = self.model.sample(chunk, num_steps=self.num_steps)
            imgs = self.vae.decode(latents)
            imgs = ((imgs.clamp(-1, 1) + 1.0) * 127.5).round()
            arr = imgs.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for a in arr:
                im = Image.fromarray(a, mode="RGB")
                if im.size != (self.image_size, self.image_size):
                    im = im.resize((self.image_size, self.image_size), Image.NEAREST)
                images.append(im)
        return images
