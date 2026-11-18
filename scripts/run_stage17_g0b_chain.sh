#!/usr/bin/env bash
# G0b (exploration, CPU only): configs i, ii, iii in turn; 4 cores (12-15), nice 19, venv_mp. A guard stops G0b at
# once if VLIA 100k training is > 10% slower than 1.85 s/step (> 2.03 s/step over the last 5 min). Resumable.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
VP=/media/dhqg/d3/venv_mp/bin/python
L=/media/dhqg/d3/logs; ST=$L/stage17_g0b_status.txt
PROG=/media/dhqg/d3/vlia_outputs/train_100k/vlia/progress.tsv
say() { echo "$(date '+%F %T') $*" | tee -a $ST; }
speed() { awk -v a=$(( $(date +%s) - 300 )) '$1>=a' $PROG | awk 'NR==1{t0=$1;s0=$2} {t=$1;s=$2} END{if (s>s0) printf "%.3f", (t-t0)/(s-s0); else print "nan"}'; }
for cfg in i ii iii; do
  say "config $cfg start (train speed last 5 min: $(speed) s/step)"
  (cd $REPO && OMP_NUM_THREADS=4 nice -n 19 taskset -c 12-15 $VP scripts/run_stage17_hoi.py g0b --config $cfg) >> $L/stage17_g0b_$cfg.log 2>&1 &
  PID=$!
  while kill -0 $PID 2>/dev/null; do
    sleep 120
    sp=$(speed)
    if [ "$sp" != "nan" ] && awk -v s=$sp 'BEGIN{exit !(s > 2.03)}'; then
      kill $PID; say "STOP: training slowed to $sp s/step (> 2.03) during config $cfg"; exit 1
    fi
  done
  wait $PID || { say "config $cfg FAILED (see stage17_g0b_$cfg.log)"; exit 1; }
  say "config $cfg done (train speed last 5 min: $(speed) s/step)"
done
(cd $REPO && $VP scripts/run_stage17_hoi.py g0b-summary) >> $L/stage17_g0b_summary.log 2>&1
say "G0B DONE"
