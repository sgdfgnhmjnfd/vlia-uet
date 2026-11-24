#!/usr/bin/env bash
# Trang 21/11: stop VLIA at 70k, run C rerun + bosung blocks 1-3 (G0 first), then VLIA continues; block 4 (final
# evaluation) waits for VLIA 100k. Re-runnable: each step checks what is already done.
O=/media/dhqg/d3/vlia_outputs/train_100k/vlia
CK=$O/intention_only/seed_0
CL=/media/dhqg/d3/logs/train100k_vlia_chain.log
BL=/media/dhqg/d3/logs/bosung_2026-11-20.log
S=/media/dhqg/d3/vlia-100k/scripts
STATE=/media/dhqg/d3/vlia_outputs/ke_hoach_cuoi/STATE.md
st() { echo "$(date '+%F %T') | bosung | $* | - | -" >> $STATE; echo "$(date '+%F %T') $*"; }
M=/media/dhqg/d3/vlia_outputs/bosung_2026-11-20/BLOCKS_123_DONE_VLIA_RESUMED
# 1. pause so that the chain stops right after checkpoint 70,000 (skipped once blocks 1-3 are done)
if [ ! -f $M ] && [ ! -f $O/STOPPED_BY_TRANG ]; then
  until grep -q "checkpoint checkpoint_067500 complete" $CL; do sleep 60; done
  [ -f $O/PAUSE ] || { touch $O/PAUSE; st "PAUSE set after 67,500 -> VLIA stops after 70,000"; }
  until grep -q "PAUSED after checkpoint_070000" $CL && ! pgrep -f train_intention_only_LIBERO7D.py > /dev/null; do sleep 60; done
  touch $O/STOPPED_BY_TRANG; st "VLIA paused at checkpoint_070000; C rerun + blocks 1-3 start"
fi
# 2. C rerun, then G0 + blocks 1-3
if [ ! -f $M ]; then
  bash $S/run_bosung_c.sh
  ONLY_BLOCKS="1 2 3" bash $S/run_sau_train_2026-11-20.sh || { st "bosung blocks 1-3 FAILED: VLIA not resumed"; exit 1; }
  # 3. VLIA continues (the chain in tmux vlia100k is waiting on PAUSE)
  rm -f $O/STOPPED_BY_TRANG $O/PAUSE; touch $M; st "blocks 1-3 done; PAUSE removed, VLIA continues"
fi
pgrep -f run_train100k_chain.sh > /dev/null || tmux new -d -s vlia100k "bash $S/run_train100k_chain.sh vlia"
# 4. block 4 after VLIA is over (100k, FROZEN or a new STOPPED_BY_TRANG)
until { [ -f $CK/summary.json ] || [ -f $O/FROZEN ] || [ -f $O/STOPPED_BY_TRANG ]; } && ! pgrep -f train_intention_only_LIBERO7D.py > /dev/null; do sleep 300; done
st "VLIA over; block 4 starts"
ONLY_BLOCKS="4" bash $S/run_sau_train_2026-11-20.sh
