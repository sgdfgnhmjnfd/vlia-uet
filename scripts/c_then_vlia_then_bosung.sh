#!/usr/bin/env bash
# Trang 22/11 10:45: "C xong thi train vlia truoc". Order: C rerun -> VLIA 70k->100k -> G0 + blocks 1, 2, 3, 4
# (one block per call, so the 20:00 22/11 cut rule cannot reorder them). Re-runnable after a power cut.
O=/media/dhqg/d3/vlia_outputs/train_100k/vlia
CK=$O/intention_only/seed_0
S=/media/dhqg/d3/vlia-100k/scripts
B=/media/dhqg/d3/vlia_outputs/bosung_2026-11-20
STATE=/media/dhqg/d3/vlia_outputs/ke_hoach_cuoi/STATE.md
st() { echo "$(date '+%F %T') | bosung | $* | - | -" >> $STATE; echo "$(date '+%F %T') $*"; }
M2=$B/VLIA_RESUMED_AFTER_C
# 1. C rerun (extract -> cv -> predict); VLIA stays paused meanwhile
if [ ! -f $M2 ]; then
  bash $S/run_bosung_c.sh
  grep -q "^EXIT 0" /media/dhqg/d3/logs/bosung_c_predict.log && st "C rerun done (EXIT 0)" || st "C rerun NOT ok (see bosung_c_predict.log); block 1 will not run automatically"
  # 2. VLIA continues (the chain in tmux vlia100k waits on PAUSE)
  rm -f $O/STOPPED_BY_TRANG $O/PAUSE; touch $M2; st "PAUSE removed, VLIA continues from 70k"
fi
pgrep -f run_train100k_chain.sh > /dev/null || tmux new -d -s vlia100k "bash $S/run_train100k_chain.sh vlia"
# 3. bosung after VLIA is over (100k, FROZEN or a new STOPPED_BY_TRANG)
until { [ -f $CK/summary.json ] || [ -f $O/FROZEN ] || [ -f $O/STOPPED_BY_TRANG ]; } && ! pgrep -f train_intention_only_LIBERO7D.py > /dev/null; do sleep 300; done
st "VLIA over; G0 + blocks 1-4 start"
for b in 1 2 3 4; do ONLY_BLOCKS="$b" bash $S/run_sau_train_2026-11-20.sh || { st "block $b FAILED"; exit 1; }; done
st "bosung blocks 1-4 done"
