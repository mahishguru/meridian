#!/usr/bin/env python
"""Pre-encode microstructure images into latent vectors for seed initialization.

Encodes all PNG/TIFF images in a per-class directory tree through the ViT encoder,
saving one .npy file per class under assets/latent_seeds/.

Expected input directory structure (one subfolder per alloy class):

    /path/to/encoded_images/
    ├── AZ31_extruded/
    │   ├── sample_001.png
    │   ├── sample_002.png
    │   └── ...
    ├── ME21_extruded/
    │   ├── sample_001.png
    │   └── ...
    └── Mg-10Gd_extruded/
        └── ...

Output:

    assets/latent_seeds/
    ├── AZ31_extruded.npy          # (N, 512) float32
    ├── ME21_extruded.npy
    ├── Mg-10Gd_extruded.npy
    └── ...

Usage:
    python scripts/encode_latent_seeds.py \
        --images-dir /path/to/encoded_images \
        --encoder-ckpt /path/to/encoder.pth \
        --output-dir assets/latent_seeds \
        --device cuda
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def encode_class(
    class_dir: Path,
    encoder,
    transform,
    device: str,
    *,
    recursive: bool = True,
    limit: int | None = None,
    seed: int = 42,
    batch_size: int = 8,
) -> np.ndarray:
    """Encode all images in a single class directory."""
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    iterator = class_dir.rglob("*") if recursive else class_dir.iterdir()
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)
    if not files:
        target_dim = int(getattr(encoder, "target_dim", 512))
        return np.empty((0, target_dim), dtype=np.float32)

    if limit is not None and limit > 0 and len(files) > limit:
        rng = random.Random(seed)
        files = sorted(rng.sample(files, limit))

    latents = []
    with torch.no_grad():
        for start in range(0, len(files), int(batch_size)):
            batch_files = files[start:start + int(batch_size)]
            imgs = []
            for f in batch_files:
                img = Image.open(f).convert("RGB")
                imgs.append(transform(img))
            t = torch.stack(imgs, dim=0).to(device)
            z = encoder(t).cpu().numpy()
            latents.append(z)

    return np.concatenate(latents, axis=0).astype(np.float32)


def _load_config(path: str | None):
    if not path:
        return None
    from meridian.config import load_config
    return load_config(path)


def _infer_from_config(cfg):
    target_dim = int(cfg.latent.dim)
    decoder_name = str(cfg.decoder.name)
    decoder_cfg = cfg.decoder.get("vitdit") if decoder_name == "vitdit" else cfg.decoder.get("fmdit")
    if decoder_cfg is None:
        raise ValueError(f"Could not infer decoder config for decoder.name={decoder_name!r}")
    repo_path = Path(decoder_cfg.repo_path)
    ckpt = Path(decoder_cfg.checkpoint)
    return target_dim, repo_path, ckpt


def _strip_module_prefix(state: dict) -> dict:
    if not state:
        return state
    if all(str(k).startswith("module.") for k in state.keys()):
        return {str(k)[7:]: v for k, v in state.items()}
    return state


def _load_encoder(encoder_ckpt: Path, repo_path: Path, target_dim: int, device: str,
                  spatial_tokens: int = 0):
    from microstructure_ed.encoder_arch_pretrained import Compressor  # repo_path unused

    encoder = Compressor(
        use_gradient_checkpointing=False,
        trainable_blocks=0,
        target_dim=int(target_dim),
        spatial_tokens=int(spatial_tokens),
    ).to(device).eval()
    ckpt = torch.load(str(encoder_ckpt), map_location=device)
    if isinstance(ckpt, dict):
        if "encoder" in ckpt:
            state = ckpt["encoder"]
        elif "compressor_state_dict" in ckpt:
            state = ckpt["compressor_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    encoder.load_state_dict(_strip_module_prefix(state), strict=True)
    return encoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-encode microstructure images to latent vectors.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional Co-PiLOT config; infers encoder checkpoint, repo path, and target dim.")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Root dir with one subfolder per alloy class, each containing PNGs.")
    parser.add_argument("--encoder-ckpt", type=str, default=None,
                        help="Path to the ViT encoder checkpoint (.pth).")
    parser.add_argument("--output-dir", type=str, default="assets/latent_seeds",
                        help="Output directory for per-class .npy files.")
    parser.add_argument("--repo-path", type=str, default=None,
                        help="Unused; kept for compatibility (encoder imported from microstructure_ed).")
    parser.add_argument("--target-dim", type=int, default=None,
                        help="Encoder bottleneck dimension; inferred from --config when provided.")
    parser.add_argument("--spatial-tokens", type=int, default=None,
                        help="Compressor spatial-token layout (v4 encoders use 16). "
                             "Default: cfg latent.spatial_tokens or 0.")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Optional class names to encode; defaults to every subdirectory in --images-dir.")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Optional deterministic sample count per class before encoding.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-recursive", action="store_true",
                        help="Only read images directly inside each class directory.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    cfg_target_dim = cfg_repo_path = cfg_encoder_ckpt = None
    if cfg is not None:
        cfg_target_dim, cfg_repo_path, cfg_encoder_ckpt = _infer_from_config(cfg)

    target_dim = int(args.target_dim or cfg_target_dim or 512)
    repo_path = Path(args.repo_path or cfg_repo_path or ".")
    encoder_ckpt = Path(args.encoder_ckpt or cfg_encoder_ckpt) if (args.encoder_ckpt or cfg_encoder_ckpt) else None
    if encoder_ckpt is None:
        print("ERROR: provide --encoder-ckpt or --config")
        return 1

    images_root = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_root.is_dir():
        print(f"ERROR: --images-dir not found: {images_root}")
        return 1

    # Discover class subdirectories
    if args.classes:
        class_dirs = [images_root / c for c in args.classes]
    else:
        class_dirs = sorted(d for d in images_root.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"ERROR: no subdirectories found in {images_root}")
        return 1
    missing = [str(d) for d in class_dirs if not d.is_dir()]
    if missing:
        print(f"ERROR: class directories not found: {missing}")
        return 1

    print(f"Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}")
    print(f"Encoder: {encoder_ckpt}")
    print(f"Repo:    {repo_path}")
    print(f"Dim:     {target_dim}")

    # Load encoder
    import torchvision.transforms as T

    spatial_tokens = int(args.spatial_tokens if args.spatial_tokens is not None
                         else (getattr(cfg.latent, "spatial_tokens", 0) or 0) if cfg is not None else 0)
    encoder = _load_encoder(encoder_ckpt, repo_path, target_dim, args.device,
                            spatial_tokens=spatial_tokens)

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # Encode each class
    total = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"  Encoding {class_name} ...", end="", flush=True)
        latents = encode_class(
            class_dir,
            encoder,
            transform,
            args.device,
            recursive=not args.no_recursive,
            limit=args.limit_per_class,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        if latents.shape[0] == 0:
            print(f" SKIP (no images)")
            continue
        out_path = output_dir / f"{class_name}.npy"
        np.save(out_path, latents)
        print(f" {latents.shape[0]} images → {out_path.name}  (shape: {latents.shape})")
        total += latents.shape[0]

    print(f"\nDone. {total} latent vectors saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
