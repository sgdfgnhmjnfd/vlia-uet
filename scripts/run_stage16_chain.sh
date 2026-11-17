#!/usr/bin/env bash
# Stage 16 chain: wait until VLIA 100k is paused (OUT/PAUSE, after a complete checkpoint), extract embeddings on the GPU
# (one job), resume VLIA (rm PAUSE), then analyze on CPU. Resumable: extract skips finished clips; re-run = continue.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
L=/media/dhqg/d3/logs; ST=$L/stage16_status.txt; OUT=/media/dhqg/d3/vlia_outputs/train_100k/vlia
say() { echo "$(date '+%F %T') $*" | tee -a $ST; }
if [ ! -f /media/dhqg/d3/vlia_outputs/stage16/emb/extract_done.json ]; then
  [ -f $OUT/PAUSE ] || touch $OUT/PAUSE
  say "waiting for VLIA to pause (next complete checkpoint) and the GPU to be free"
  until [ -f $OUT/PAUSE ] && ! { [ -f $OUT/train.pid ] && kill -0 $(cat $OUT/train.pid) 2>/dev/null; } && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
  say "GPU free; extract start"
  (cd $REPO && $PY scripts/run_stage16_scene_motion.py extract) >> $L/stage16_extract.log 2>&1 || { say "extract FAILED"; exit 1; }
  say "extract done"
fi
rm -f $OUT/PAUSE && say "VLIA resumed (PAUSE removed)"
(cd $REPO && CUDA_VISIBLE_DEVICES="" S16_PROCS=8 nice -n 5 $PY scripts/run_stage16_scene_motion.py analyze) >> $L/stage16_analyze.log 2>&1 || { say "analyze FAILED"; exit 1; }
say "STAGE16 ANALYZE DONE"
