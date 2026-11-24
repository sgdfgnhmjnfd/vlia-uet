#!/usr/bin/env bash
# Bosung 20/11 step 2: C rerun (extract -> cv -> predict). Same command again after a power cut: finished feature files
# are kept (unreadable ones moved aside), the CV is redone (about 10 min), predict is redone.
cd /media/dhqg/d3/vlia-100k || exit 1
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
L=/media/dhqg/d3/logs
grep -q "^EXIT 0" $L/bosung_c_predict.log 2>/dev/null && { echo "C rerun already done"; exit 0; }
$PY scripts/bosung_c_rerun.py extract >> $L/bosung_c_extract.log 2>&1 &&
  $PY scripts/bosung_c_rerun.py cv > $L/bosung_c_cv.log 2>&1 &&
  $PY scripts/bosung_c_rerun.py predict > $L/bosung_c_predict.log 2>&1
echo "EXIT $?" >> $L/bosung_c_predict.log
