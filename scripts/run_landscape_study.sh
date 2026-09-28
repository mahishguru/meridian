#!/usr/bin/env bash
# Where is the design objective sensitive? Paired DAMASK simulations of
# controlled perturbations (FM-DiT-768, AZ31 and Mg-5Gd):
#   latent  z -> z + eps*u, eps in {0.25, 1.0}          (probes J o f o D)
#   direct  per-grain reorientation of 5/15 deg RMS, or boundary migration
#           of 1/5 % of pixels, applied to the seed RVE  (probes J o f)
# The direct arm needs the seed RVEs (.dream3d) referenced by the seed-cache
# manifests under simulations/runs/.
#
# usage: scripts/run_landscape_study.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
WORK=${WORK:-$MERIDIAN_ROOT/_landscape_work}; mkdir -p logs/d1 results/d1 "$WORK"
AZ_CFG=configs/acta2026/az31_fmdit_768_meridian_v2.yaml
MG_CFG=configs/acta2026/mg5gd_fmdit_768_meridian_v2.yaml
AZ_CACHE=data/seed_caches/seed_cache_AZ31_v2_fmdit_768.npz
MG_CACHE=data/seed_caches/seed_cache_Mg5Gd_v2_fmdit_768.npz

launch() {  # gpu arm alloy config cache shard nshards concurrency
  local gpu=$1 arm=$2 alloy=$3 cfg=$4 cache=$5 shard=$6 nsh=$7 conc=$8
  local tag="${alloy}_${arm}_s${shard}" nbase=5
  [[ "$arm" == latent ]] && nbase=10
  CUDA_VISIBLE_DEVICES=$(( gpu % N_GPUS )) "$PY" scripts/d1_lipschitz.py \
      --config "$cfg" --arm "$arm" --alloy "$alloy" --seed-cache "$cache" \
      --out "results/d1/d1_${tag}.jsonl" --work-root "$WORK/$tag" \
      --n-base "$nbase" --n-dir 4 --shard "$shard" --n-shards "$nsh" \
      --concurrency "$conc" --resume \
      --override "simulation.n_threads=4" --override "decoder.fmdit.decode_batch_size=1" \
      >> "logs/d1/${tag}.log" 2>&1 &
  info "launched $tag"
}
launch 0 latent AZ31  "$AZ_CFG" "$AZ_CACHE" 0 2 7
launch 1 latent AZ31  "$AZ_CFG" "$AZ_CACHE" 1 2 7
launch 2 latent Mg5Gd "$MG_CFG" "$MG_CACHE" 0 2 7
launch 3 latent Mg5Gd "$MG_CFG" "$MG_CACHE" 1 2 7
launch 0 direct AZ31  "$AZ_CFG" "$AZ_CACHE" 0 1 13
launch 1 direct Mg5Gd "$MG_CFG" "$MG_CACHE" 0 1 13
wait
info "done; endpoints in results/d1/*.jsonl"
