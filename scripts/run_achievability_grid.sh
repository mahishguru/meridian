#!/usr/bin/env bash
# AZ31 target-achievability map: MERIDIAN on FM-DiT-512 against a 4x4 grid of
# (sigma_y, n) targets (16 cells x 160 evaluations), objective v3 (miss distance
# in seed standard deviations). Configs and rescored seed caches are in
# configs/acta2026/fig15/ and data/seed_caches/fig15/ (regenerate with
# scripts/fig15_grid_design.py).
#
# usage: scripts/run_achievability_grid.sh        env: CONCURRENT (default 2 x GPUs)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
CONCURRENT=${CONCURRENT:-$(( N_GPUS * 2 ))}
GROUP=AZ31_512_fig15; LOG=logs/fig15; mkdir -p "$LOG"
slot=0
for i in $(seq -w 0 15); do
  CFG="configs/acta2026/fig15/az31_512_fig15_c${i}.yaml"
  gpu=$(( slot % N_GPUS ))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_ablation.py \
      --config "$CFG" --group "$GROUP" \
      --optimizers meridian --decoders fmdit --objectives v3 \
      --rep-start "$((10#$i))" --repeats 1 \
      --skip-completed --resume --resume-min-iter 5 \
      --override "simulation.n_threads=4" "decoder.fmdit.decode_batch_size=1" \
      >> "$LOG/${GROUP}_c${i}.log" 2>&1 &
  info "cell $i -> GPU $gpu"
  slot=$((slot + 1))
  if (( slot % CONCURRENT == 0 )); then wait; fi
done
wait
info "done; results in results/$GROUP/"
