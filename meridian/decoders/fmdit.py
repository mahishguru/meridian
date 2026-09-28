"""FM-DiT decoder wrapper (SD3.5 hijack + flow matching).

Loads the trained `FlowMatchingDiTDecoder` adapter (~15 M trainable params on a
frozen 2.5 B-param SD3.5 MMDiT backbone) plus the frozen 16-channel SD3.5 VAE
from a checkpoint of the `microstructure-encoder-decoder` package
(training checkpoint or the slim release checkpoint from the Hugging Face Hub).

The checkpoint .pth is a dict with keys
    {"epoch", "encoder", "decoder", "optimizer", "lr_scheduler", ...}
and the `decoder` entry holds the FlowMatchingDiTDecoder state_dict.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
from PIL import Image

from meridian.decoders import BaseDecoder


class FMDiTDecoder(BaseDecoder):
    name = "fmdit"

    def __init__(self, cfg) -> None:
        ckpt = cfg.decoder.fmdit.checkpoint
        if ckpt is None:
            raise NotImplementedError(
                "FMDiTDecoder requires `decoder.fmdit.checkpoint` to point at a trained\n"
                "FlowMatchingDiTDecoder checkpoint .pth (with 'decoder' state_dict)."
            )
        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"FMDiT checkpoint not found: {ckpt}")

        # ---- v4 checkpoint compatibility -------------------------------
        # v4scratch checkpoints carry a texture head + rank-64 LoRA (incl.
        # FFN targets). FlowMatchingDiTDecoder reads these env vars in its
        # __init__, so they must be set before construction or the LoRA
        # state_dict shapes mismatch (rank 16 vs 64) and tex_head keys are
        # unexpected. Controlled per-config via decoder.fmdit.load_env.
        import os as _os
        load_env = getattr(cfg.decoder.fmdit, "load_env", None)
        if load_env:
            for k, v in dict(load_env).items():
                _os.environ[str(k)] = str(v)
                print(f"[fmdit] load_env {k}={v}")

        # One FM-DiT module serves all bottleneck widths (512/768/1024/1280).
        target_dim = int(getattr(cfg.decoder.fmdit, "target_dim", None) or 512)
        arch_mod = importlib.import_module("microstructure_ed.fmdit.decoder_arch_pretrained")
        FlowMatchingDiTDecoder = arch_mod.FlowMatchingDiTDecoder
        SD35VAE = arch_mod.SD35VAE

        device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_steps = int(cfg.decoder.num_inference_steps)
        self.image_size = int(cfg.decoder.image_size)
        # Validated v4 sampling recipe (defaults = legacy behaviour).
        self.noise_temp = float(getattr(cfg.decoder.fmdit, "noise_temp", 1.0))
        _shift = getattr(cfg.decoder.fmdit, "shift", None)
        self.shift = float(_shift) if _shift is not None else None
        self.guidance_scale = float(getattr(cfg.decoder.fmdit, "guidance_scale", 1.0))
        print(f"[fmdit] sampler: steps={self.num_steps} noise_temp={self.noise_temp} "
              f"shift={self.shift} guidance={self.guidance_scale}")
        self.batch_size = int(getattr(cfg.decoder.fmdit, "decode_batch_size", 4))

        model_id = cfg.decoder.fmdit.sd35_model_id
        hf_token = getattr(cfg.decoder.fmdit, "hf_token", None)

        print(f"[fmdit] loading SD3.5 backbone from {model_id}")
        self.model = FlowMatchingDiTDecoder(model_id=model_id, token=hf_token,
                                            target_dim=target_dim)
        self.vae = SD35VAE(model_id=model_id, token=hf_token)

        print(f"[fmdit] loading adapter weights from {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        dec_state = state.get("decoder", state)
        # `trainer.py` may save a DDP-wrapped state_dict whose keys are prefixed
        # with "module."; strip it so non-DDP loads cleanly.
        if any(k.startswith("module.") for k in dec_state):
            dec_state = {k.removeprefix("module."): v for k, v in dec_state.items()}

        missing, unexpected = self.model.load_state_dict(dec_state, strict=False)
        n_loaded = max(0, len(dec_state) - len(unexpected))
        print(f"[fmdit] loaded {n_loaded} tensors "
              f"({len(unexpected)} unexpected, {len(missing)} missing - frozen "
              f"backbone reuses pretrained weights)")

        # IMPORTANT — keep the trainable adapter (token_generator + pooled_proj)
        # in float32 to match the trainer (`trainer.py` runs them under
        # `torch.amp.autocast("cuda", dtype=torch.bfloat16)` while parameters
        # remain fp32).  Casting the weights to bf16 here destroys the trained
        # signal and produces noise output.
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latents = self.model.sample(
                    chunk, num_steps=self.num_steps,
                    noise_temp=self.noise_temp, shift=self.shift,
                    guidance_scale=self.guidance_scale)
            imgs = self.vae.decode(latents)
            imgs = ((imgs.clamp(-1, 1) + 1.0) * 127.5).round()
            arr = imgs.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for a in arr:
                im = Image.fromarray(a, mode="RGB")
                if im.size != (self.image_size, self.image_size):
                    im = im.resize((self.image_size, self.image_size), Image.NEAREST)
                images.append(im)
        return images
