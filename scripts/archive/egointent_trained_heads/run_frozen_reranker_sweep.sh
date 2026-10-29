#!/usr/bin/env bash
set -euo pipefail

cd ~/vlia-uet

SCRIPT="scripts/train_egointent_frozen_reranker.py"
BASE_ROOT="/media/dhqg/d1/vlia_outputs/egointent_softmrr_v1"
OUT_ROOT="/media/dhqg/d1/vlia_outputs/egointent_frozen_reranker_sweep"

# Keep the objective fixed; only vary how much the frozen baseline
# can be corrected. 0.20 is the current best.
SCALES=(0.10 0.15 0.20 0.25)

for SCALE in "${SCALES[@]}"; do
  TAG=$(echo "$SCALE" | tr '.' 'p')
  OUT="${OUT_ROOT}/scale_${TAG}"

  echo "================================================================================================"
  echo "FROZEN RERANKER | residual_scale=${SCALE}"
  echo "================================================================================================"

  python "$SCRIPT" \
    --base-root "$BASE_ROOT" \
    --seeds 42 123 456 \
    --residual-scale "$SCALE" \
    --softmrr-weight 1.0 \
    --cosine-weight 0.05 \
    --residual-reg-weight 0.001 \
    --rank-temperature 0.05 \
    --rank-positive-temperature 0.05 \
    --output-dir "$OUT" \
    2>&1 | tee "${OUT}_console.log"
done
