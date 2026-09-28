# Shared environment for the launchers (sourced, not executed).
#   PYTHON         python with meridian, microstructure-encoder-decoder, orientation-codec
#   DAMASK_ENV     optional conda env prefix providing DAMASK_grid (added to PATH / LD_LIBRARY_PATH)
#   MSED_ROOT      encoder-decoder weights + texture priors (default: weights/, see download_weights.sh)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MERIDIAN_ROOT="$PWD"
export MSED_ROOT="${MSED_ROOT:-$MERIDIAN_ROOT/weights}"
export SAM_CHECKPOINT="${SAM_CHECKPOINT:-$MSED_ROOT/checkpoints_sam/sam_vit_b_01ec64.pth}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
if [[ -n "${DAMASK_ENV:-}" ]]; then
  export PATH="$DAMASK_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$DAMASK_ENV/lib:${LD_LIBRARY_PATH:-}"
fi
command -v DAMASK_grid >/dev/null 2>&1 || echo "[warn] DAMASK_grid not on PATH (set DAMASK_ENV or activate the damask env)" >&2
N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"; [[ "$N_GPUS" -ge 1 ]] || N_GPUS=1
info() { echo "[$(date +%F_%T)] $*"; }
