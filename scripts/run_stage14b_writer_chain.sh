#!/usr/bin/env bash
# Stage 14b Part B chain (resumable). Waits for the Part C chain to leave the GPU. Status: /media/dhqg/d3/logs/stage14b_writer_status.txt
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=/media/dhqg/d3/logs
STATUS=$LOG/stage14b_writer_status.txt
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
free_root() { [ "$(df --output=avail -m / | tail -1)" -gt 500 ] || { say "STOP: less than 500 MB free on /"; exit 1; }; }
WR() { (cd $REPO && $PY scripts/run_stage14b_writer.py "$@"); }
say "chain start ($(git -C $REPO rev-parse --short HEAD)); waiting for the GPU"
while pgrep -f "run_stage14b_variants.py|train_intention_only|libero_goal_ablation.py --out" > /dev/null; do sleep 60; done
(cd $REPO && $PY scripts/test_stage14b_writer.py) > $LOG/stage14b_writer_test.log 2>&1 || { say "tests FAILED"; exit 1; }
for step in train-ego write-ego train-libero write-libero unseen; do
  free_root
  WR $step >> $LOG/stage14b_writer.log 2>&1 || { say "$step FAILED"; exit 1; }
  say "$step done"
done
