#!/usr/bin/env bash
# Stage 14b chain (resumable): calibration G0/G1/G2/O1 at p = 25, 50 -> choose -> chosen G and O1 on the 299 units
#   at all p -> blank controls -> analyze. No training, no robot. Status: /media/dhqg/d3/logs/stage14b_status.txt
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=/media/dhqg/d3/logs
STATUS=$LOG/stage14b_status.txt
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
free_root() { [ "$(df --output=avail -m / | tail -1)" -gt 500 ] || { say "STOP: less than 500 MB free on /"; exit 1; }; }
V() { (cd $REPO && $PY scripts/run_stage14b_variants.py "$@"); }
say "chain start ($(git -C $REPO rev-parse --short HEAD))"
(cd $REPO && $PY scripts/test_stage14b.py) > $LOG/stage14b_test.log 2>&1 || { say "tests FAILED"; exit 1; }
for v in G0 G1 G2 O1; do
  free_root
  V gen --variant $v --split cal --p 25,50 >> $LOG/stage14b_gen.log 2>&1 || { say "gen $v cal FAILED"; exit 1; }
  say "calibration $v done"
done
V choose > $LOG/stage14b_choose.log 2>&1 || { say "choose FAILED"; exit 1; }
CG=$($PY -c "import json;print(json.load(open('$REPO/results/stage14b_generate_variants/chosen.json'))['G'])")
say "chosen: G = $CG, G+obj = O1 ($(tail -1 $LOG/stage14b_choose.log))"
for v in $CG O1; do
  free_root
  V gen --variant $v --split test --p 0,10,25,50,75,100 >> $LOG/stage14b_gen.log 2>&1 || { say "gen $v test FAILED"; exit 1; }
  V blank --variant $v >> $LOG/stage14b_gen.log 2>&1 || { say "blank $v FAILED"; exit 1; }
  say "test $v done"
done
V blank --variant G0 >> $LOG/stage14b_gen.log 2>&1
V analyze > $LOG/stage14b_analyze.log 2>&1 || { say "analyze FAILED"; exit 1; }
say "STAGE14B DONE"
