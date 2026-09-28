#!/usr/bin/env bash
# Download the released encoder-decoder checkpoints, texture priors and the SAM
# ViT-B checkpoint into $MSED_ROOT (default: weights/).
#   Hugging Face: https://huggingface.co/mahishguru/microstructure-encoder-decoder
#   usage: scripts/download_weights.sh [variant ...]   (default: all FM-DiT widths + ViT-DiT)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -gt 0 ]] || VARIANTS=(vitfmdit vitfmdit_768 vitfmdit_1024 vitfmdit_1280 vitdit)
mkdir -p "$MSED_ROOT/checkpoints" "$(dirname "$SAM_CHECKPOINT")"
"$PY" - "$MSED_ROOT" "${VARIANTS[@]}" <<'PY'
import os, sys
from pathlib import Path
from microstructure_ed.checkpoints import TEXTURE_PRIOR_VARIANTS, download, download_texture_prior
root = Path(sys.argv[1])
for v in sys.argv[2:]:
    link = root / "checkpoints" / f"{v}.pth"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(download(v))
    print(f"{v:14s} -> {link}")
    if v in TEXTURE_PRIOR_VARIANTS:
        print(f"{'':14s}    texture prior -> {download_texture_prior(v, root)}")
print(f"{'shared prior':14s} -> {download_texture_prior(None, root)}")
PY
if [[ ! -f "$SAM_CHECKPOINT" ]]; then
  info "downloading SAM ViT-B -> $SAM_CHECKPOINT"
  curl -L -o "$SAM_CHECKPOINT" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi
