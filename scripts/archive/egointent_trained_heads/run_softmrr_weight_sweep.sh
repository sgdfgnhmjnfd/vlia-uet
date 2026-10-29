#!/usr/bin/env bash
set -euo pipefail

cd ~/vlia-uet

SCRIPT="scripts/train_egointent_softmrr.py"
ROOT="/media/dhqg/d1/vlia_outputs/egointent_softmrr_weight_sweep"

WEIGHTS=(0.03 0.05 0.10 0.20)

for W in "${WEIGHTS[@]}"; do
  TAG=$(echo "$W" | tr '.' 'p')
  OUT="${ROOT}/w_${TAG}"

  echo "================================================================================================"
  echo "SOFT-MRR WEIGHT = ${W}"
  echo "OUTPUT = ${OUT}"
  echo "================================================================================================"

  python "$SCRIPT" \
    --conditions what_hardneg_guided softmrr_guided \
    --seeds 42 123 456 \
    --what-hard-weight 0.30 \
    --what-hard-margin 0.10 \
    --softmrr-weight "$W" \
    --rank-temperature 0.05 \
    --rank-positive-temperature 0.05 \
    --output-dir "$OUT" \
    2>&1 | tee "${OUT}_console.log"
done
