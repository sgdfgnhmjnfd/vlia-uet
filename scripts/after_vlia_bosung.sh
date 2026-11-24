#!/usr/bin/env bash
# Trang 21/11: VLIA 100k first, bosung after. Waits until the training chain is over (summary.json = 100k done, FROZEN,
# or STOPPED_BY_TRANG) and no training process runs, then C rerun (run_bosung_c.sh), then run_sau_train (G0, blocks 1-4).
O=/media/dhqg/d3/vlia_outputs/train_100k/vlia
S=/media/dhqg/d3/vlia-100k/scripts
while ! { [ -f $O/intention_only/seed_0/summary.json ] || [ -f $O/FROZEN ] || [ -f $O/STOPPED_BY_TRANG ]; } \
      || pgrep -f "train_intention_only_LIBERO7D.py" > /dev/null; do sleep 300; done
echo "$(date '+%F %T') training over; C rerun"
bash $S/run_bosung_c.sh
echo "$(date '+%F %T') C rerun finished; bosung blocks"
bash $S/run_sau_train_2026-11-20.sh
