#!/usr/bin/env bash
# Optimiser comparison under a fixed budget (40 rounds x batch 4 = 160 DAMASK
# evaluations, warm-started from the released 100-seed cache).
#
# usage: scripts/run_campaign.sh <AZ31|Mg5Gd> <fmdit|fmdit_768|fmdit_1024|fmdit_1280> \
#            [optimizers=dante,turbo,baxus,meridian] [rep_start=0] [repeats=1]
# env:   MAX_PAR (parallel optimizer runs, default 1), SEED (default 42), N_THREADS (DAMASK threads)
#
# AZ31 uses one config per optimizer (configs/acta2026/az31_<decoder>_<opt>_v2.yaml);
# Mg-5Gd runs every optimizer from configs/acta2026/mg5gd_<decoder>_meridian_v2.yaml.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ALLOY=${1:?alloy}; DEC=${2:?decoder}; OPTS=${3:-dante,turbo,baxus,meridian}
REP_START=${4:-0}; REPEATS=${5:-1}; MAX_PAR=${MAX_PAR:-1}; SEED=${SEED:-42}
lower=$(echo "$ALLOY" | tr '[:upper:]' '[:lower:]')
CACHE="data/seed_caches/seed_cache_${ALLOY}_v2_${DEC}.npz"
[[ -f "$CACHE" ]] || { echo "missing seed cache $CACHE" >&2; exit 1; }
LOG=logs/campaign_${ALLOY}_${DEC}_$(date +%Y%m%d_%H%M%S); mkdir -p "$LOG"
EXTRA=(); [[ -n "${N_THREADS:-}" ]] && EXTRA+=("simulation.n_threads=$N_THREADS")

run_one() {  # idx optimizer
  local idx=$1 opt=$2 cfg
  if [[ "$ALLOY" == "AZ31" ]]; then cfg="configs/acta2026/az31_${DEC}_${opt}_v2.yaml"
  else cfg="configs/acta2026/${lower}_${DEC}_meridian_v2.yaml"; fi
  local gpu=$(( idx % N_GPUS ))
  info "start $opt ($cfg) on GPU $gpu -> $LOG/$opt.log"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_ablation.py \
      --config "$cfg" --seed-cache "$CACHE" --group "${ALLOY}_v2_${DEC}" \
      --only "$opt" --decoders "$DEC" --objectives v2 \
      --rep-start "$REP_START" --repeats "$REPEATS" \
      --skip-completed --resume --resume-min-iter 5 \
      --override "experiment.seed=$SEED" "${EXTRA[@]}" > "$LOG/$opt.log" 2>&1
  info "done $opt (exit $?)"
}

idx=0; running=0
for opt in ${OPTS//,/ }; do
  run_one $idx "$opt" & idx=$((idx + 1)); running=$((running + 1))
  if (( running >= MAX_PAR )); then wait -n; running=$((running - 1)); fi
done
wait
info "results in results/${ALLOY}_v2_${DEC}/"
