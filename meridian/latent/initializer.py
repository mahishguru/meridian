"""Initial latent population builders.

Four modes (config-selectable):
  - sobol         : Sobol quasi-random samples in the box [low, high].
  - from_npy      : load a pre-computed (N, dim) array.
  - encode_images : run the ViT encoder on a directory of PNG images
                    (delegated to the upstream encoder package).
  - from_class    : randomly sample n_seed vectors from a pre-encoded
                    .npy file for a specific material class (recommended
                    over sobol for production runs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.quasirandom import SobolEngine


def init_sobol(n_seed: int, dim: int, low: float, high: float, seed: int = 42) -> np.ndarray:
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    u = sobol.draw(n_seed).numpy()
    return (low + (high - low) * u).astype(np.float32)


def init_from_npy(path: str | Path, n_seed: int, dim: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected (N, {dim}) latent .npy, got {arr.shape}")
    if len(arr) < n_seed:
        raise ValueError(f"Requested n_seed={n_seed} but file has {len(arr)} rows")
    return arr[:n_seed].copy()


def init_from_images(
    images_dir: str | Path,
    encoder_ckpt: str | Path,
    dim: int,
    device: str = "cuda",
) -> np.ndarray:
    """Encode a directory of PNGs through the ViT encoder.

    Imports the upstream encoder lazily; only required when this mode is used.
    """
    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    try:
        from microstructure_ed.encoder_arch_pretrained import MicrostructureEncoder  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "encode_images mode needs the encoder-decoder repo on PYTHONPATH "
            "(install microstructure-encoder-decoder)"
        ) from exc

    enc = MicrostructureEncoder().to(device).eval()
    state = torch.load(str(encoder_ckpt), map_location=device)
    enc.load_state_dict(state if isinstance(state, dict) and "model" not in state else state["model"], strict=False)

    from PIL import Image
    import torchvision.transforms as T

    tfm = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    out = []
    with torch.no_grad():
        for f in files:
            img = tfm(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            z = enc(img).squeeze(0).cpu().numpy()
            out.append(z)
    arr = np.stack(out).astype(np.float32)
    if arr.shape[1] != dim:
        raise ValueError(f"Encoder produced dim={arr.shape[1]}, expected {dim}")
    return arr


def init_from_class(
    seeds_dir: str | Path,
    class_key: str,
    n_seed: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Randomly sample *n_seed* latent vectors from pre-encoded class file.

    Looks for ``{seeds_dir}/{class_key}.npy`` with shape ``(N, dim)``.
    Samples without replacement when ``n_seed <= N``, with replacement otherwise.
    """
    seeds_dir = Path(seeds_dir)
    npy_path = seeds_dir / f"{class_key}.npy"
    if not npy_path.is_file():
        available = sorted(p.stem for p in seeds_dir.glob("*.npy"))
        raise FileNotFoundError(
            f"No latent seed file for class '{class_key}'. "
            f"Expected {npy_path}. Available: {available}"
        )
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"Expected (N, {dim}) in {npy_path}, got {arr.shape}")

    rng = np.random.default_rng(seed)
    replace = n_seed > len(arr)
    idx = rng.choice(len(arr), size=n_seed, replace=replace)
    return arr[idx].copy()


def build_initial_population(cfg) -> np.ndarray:
    init = cfg.latent.init
    mode = init.mode.lower()
    dim = int(cfg.latent.dim)
    low, high = cfg.latent.bounds
    seed = int(cfg.experiment.seed)
    n_seed = int(init.n_seed)

    if mode == "sobol":
        return init_sobol(n_seed, dim, float(low), float(high), seed=seed)
    if mode == "from_npy":
        if init.npy_path is None:
            raise ValueError("latent.init.npy_path required for mode=from_npy")
        return init_from_npy(init.npy_path, n_seed, dim)
    if mode == "encode_images":
        if init.images_dir is None or init.encoder_ckpt is None:
            raise ValueError("latent.init.images_dir and encoder_ckpt required for mode=encode_images")
        return init_from_images(
            init.images_dir, init.encoder_ckpt, dim, device=cfg.experiment.device
        )
    if mode == "from_class":
        seeds_dir = getattr(init, "seeds_dir", None) or "assets/latent_seeds"
        class_key = getattr(init, "class_key", None) or cfg.codec.class_key
        return init_from_class(seeds_dir, class_key, n_seed, dim, seed=seed)
    raise ValueError(f"Unknown latent.init.mode: {mode}")
