#!/usr/bin/env bash
set -euo pipefail

cd ~/vlia-uet

SCRIPT="scripts/train_egointent_what_hardneg.py"
BASE_OUT="/media/dhqg/d1/vlia_outputs/egointent_what_hardneg_sweep"
SEEDS="42 123 456"

mkdir -p "$BASE_OUT"

for W in 0.05 0.10 0.20 0.30; do
  TAG="${W/./p}"
  OUT_DIR="${BASE_OUT}/weight_${TAG}"

  echo
  echo "================================================================"
  echo "RUNNING what_hard_weight=${W}"
  echo "OUTPUT: ${OUT_DIR}"
  echo "================================================================"

  python "$SCRIPT" \
    --conditions what_hardneg_guided \
    --seeds $SEEDS \
    --what-hard-weight "$W" \
    --what-hard-margin 0.10 \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "${BASE_OUT}/weight_${TAG}.log"
done

echo
echo "================================================================"
echo "SWEEP COMPLETE"
echo "Logs:"
ls -1 "${BASE_OUT}"/*.log
echo "================================================================"
