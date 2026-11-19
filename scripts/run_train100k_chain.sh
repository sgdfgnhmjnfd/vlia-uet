#!/usr/bin/env bash
# Usage: scripts/run_train100k_chain.sh vlia|baseline
# 100k training chain (ke hoach cuoi 17/11, Part 4 Step 1). Run the same command again at any time (after a power cut,
# a pause, a crash) and it continues: it finds the newest COMPLETE checkpoint and resumes from it with --resume.
#   - complete = model.safetensors header parses and its size matches, config.json exists, training_state.pt is a
#     readable zip; a broken newest checkpoint (power cut) is moved aside to OUT/incomplete/, never deleted.
#   - pruning: after a new complete checkpoint, older checkpoints of THIS run are deleted except 25k/50k/75k/100k and
#     the two most recent. Nothing else is deleted.
#   - pause (e.g. for Stage 16): `touch OUT/PAUSE` -> training stops right after the next complete checkpoint;
#     `rm OUT/PAUSE` -> the chain resumes by itself.
#   - freeze: at 22:00 on 22/11/2026 training is stopped (FROZEN written); the milestone is then chosen by the rule.
#   - T1: seconds per step from steps 10..1,000 and the projected finish (24 h/day and 14 h/day) go to the chain log
#     and STATE.md. Milestones 25k/50k/75k/100k get one line in STATE.md.
# Logs: /media/dhqg/d3/logs/train100k_<model>.log (training), train100k_<model>_chain.log (this chain).
set -uo pipefail
MODEL=${1:?usage: run_train100k_chain.sh vlia|baseline}
REPO=$(cd "$(dirname "$0")/.." && pwd)
VPY=/home/dhqg/lerobot/.venv/bin/python
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python; [ -x "$PY" ] || PY=$HOME/envs/lerobot/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
case $MODEL in
  vlia)     SCRIPT=train_intention_only_LIBERO7D.py;     COND=intention_only ;;
  baseline) SCRIPT=train_matched_oracle_LIBERO7D_FIXED.py; COND=baseline ;;
  *) echo "MODEL must be vlia or baseline"; exit 1 ;;
esac
STEPS=100000; SAVE=2500
OUT=/media/dhqg/d3/vlia_outputs/train_100k/$MODEL
CK=$OUT/$COND/seed_0
LOG=/media/dhqg/d3/logs/train100k_$MODEL.log
CLOG=/media/dhqg/d3/logs/train100k_${MODEL}_chain.log
STATE=/media/dhqg/d3/vlia_outputs/ke_hoach_cuoi/STATE.md
FREEZE=$(date -d '2026-11-22 22:00' +%s)
mkdir -p $OUT /media/dhqg/d3/tmp /media/dhqg/d3/logs
say() { echo "$(date '+%F %T') $*" | tee -a $CLOG; }
state() { [ -f $STATE ] && echo "$(date '+%F %T') | Part 4 | train100k $MODEL | $* | $OUT | -" >> $STATE; }
venv_env() { unset PYTHONPATH; SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
  export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0 HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot
  export PYTHONPATH="$SITEPKG:$REPO"; }

complete() {  # $1 = checkpoint dir
  local c=$1 m=$1/pretrained_model
  [ -f "$m/model.safetensors" ] && [ -f "$m/config.json" ] && [ -f "$c/training_state.pt" ] || return 1
  "$PY" - "$m/model.safetensors" "$c/training_state.pt" <<'EOF' >/dev/null 2>&1
import json, os, struct, sys, zipfile
p, t = sys.argv[1], sys.argv[2]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
assert os.path.getsize(p) == 8 + n + max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
with zipfile.ZipFile(t) as z:
    assert z.namelist()
EOF
}
num() { basename "$1" | sed 's/checkpoint_0*//; s/^$/0/'; }

newest_complete() {  # moves broken checkpoints that are newer than the newest complete one to OUT/incomplete/
  local c
  for c in $(ls -d $CK/checkpoint_* 2>/dev/null | sort -r); do
    if complete $c; then echo $c; return; fi
    mkdir -p $OUT/incomplete; local dst=$OUT/incomplete/$(basename $c)_$(date +%Y%m%d_%H%M%S)
    mv $c $dst; say "checkpoint $(basename $c) incomplete (power cut?) -> moved to $dst" >&2
  done
}

prune() {
  local all keep c n
  all=$(ls -d $CK/checkpoint_* 2>/dev/null | sort)
  keep=$(echo "$all" | tail -2)
  for c in $all; do
    n=$(num $c)
    if echo "$keep" | grep -qx "$c"; then continue; fi
    case $n in 25000|50000|75000|100000) continue ;; esac
    complete $c || continue
    rm -rf "$c" && say "pruned $(basename $c)"
  done
}

last_step() { grep -oE "^step [0-9]+" $LOG 2>/dev/null | tail -1 | grep -oE "[0-9]+" | sed 's/^0*//; s/^$/0/'; }

t1_done() { grep -q "T1 timing" $CLOG 2>/dev/null; }
fails=0; seen_ms=""; [ -f $OUT/milestones_logged ] && seen_ms=$(cat $OUT/milestones_logged)

say "chain start ($MODEL, repo $(git -C $REPO rev-parse --short HEAD), freeze $(date -d @$FREEZE '+%F %T'))"
while true; do
  [ -f $CK/summary.json ] && [ "$($PY -c "import json;print(json.load(open('$CK/summary.json'))['steps'])")" = "$STEPS" ] && { say "TRAINING DONE (summary.json, $STEPS steps)"; state "TRAINING DONE"; break; }
  [ -f $OUT/FROZEN ] && { say "frozen at $(cat $OUT/FROZEN); not restarting"; break; }
  if [ "$(date +%s)" -ge $FREEZE ]; then L=$(newest_complete); echo "$(date '+%F %T') newest complete $(basename "${L:-none}")" > $OUT/FROZEN; say "FREEZE reached"; state "FROZEN at $(basename "${L:-none}")"; break; fi
  while [ -f $OUT/PAUSE ]; do sleep 30; done
  [ "$(df --output=avail -m /media/dhqg/d3 | tail -1)" -gt 20000 ] || { say "STOP: less than 20 GB free on /media/dhqg/d3"; exit 1; }
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then say "GPU busy (another job); waiting"; sleep 120; continue; fi
  L=$(newest_complete); RESUME=(); [ -n "$L" ] && RESUME=(--resume $L)
  say "start training ${RESUME[*]:-(fresh from lerobot/smolvla_base)}"
  t_start=$(date +%s); s_start=$(last_step)
  (venv_env; cd /home/dhqg/lerobot && exec $VPY $REPO/scripts/$SCRIPT --condition $COND --steps $STEPS --batch-size 32 \
      --save-freq $SAVE --log-freq 10 --seed 0 --output-root $OUT "${RESUME[@]}") >> $LOG 2>&1 &
  PID=$!; echo $PID > $OUT/train.pid; reason=""
  t10=""; last_ck=$(basename "${L:-none}")
  while kill -0 $PID 2>/dev/null; do
    sleep 60
    st=$(last_step); echo -e "$(date +%s)\t$st" >> $OUT/progress.tsv
    # T1 timing from the first 1,000 steps of the fresh run
    if ! t1_done && [ -n "$st" ]; then
      [ -z "$t10" ] && [ "$st" -ge 10 ] && t10="$(date +%s) $st"
      if [ -n "$t10" ] && [ "$st" -ge 1000 ]; then
        set -- $t10; sps=$($PY -c "print(round(($(date +%s)-$1)/($st-$2),3))")
        f24=$($PY -c "import time;print(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()+$sps*($STEPS-$st))))")
        f14=$($PY -c "import time;print(time.strftime('%Y-%m-%d %H:%M',time.localtime(time.time()+$sps*($STEPS-$st)*24/14)))")
        say "T1 timing: $sps s/step (steps $2-$st); finish at 24 h/day: $f24; at 14 h/day: $f14"; state "T1 $sps s/step, finish 24h/day $f24, 14h/day $f14"
      fi
    fi
    # new complete checkpoint -> prune, milestones, pause
    n=$(ls -d $CK/checkpoint_* 2>/dev/null | sort | tail -1)
    if [ -n "$n" ] && [ "$(basename $n)" != "$last_ck" ] && complete $n; then
      last_ck=$(basename $n); say "checkpoint $last_ck complete"; prune
      k=$(num $n)
      case $k in 25000|50000|75000|100000)
        if ! echo "$seen_ms" | grep -qw $k; then
          loss=$(grep -E "^step 0*$k " $LOG | tail -1 | grep -oE "loss100 [0-9.]+")
          say "MILESTONE $k ($loss)"; state "milestone $k reached ($loss)"; seen_ms="$seen_ms $k"; echo "$seen_ms" > $OUT/milestones_logged
        fi ;;
      esac
      if [ -f $OUT/PAUSE ]; then kill $PID; wait $PID 2>/dev/null; reason=pause; say "PAUSED after $last_ck (OUT/PAUSE)"; state "paused after $last_ck"; break; fi
    fi
    if [ "$(date +%s)" -ge $FREEZE ]; then kill $PID; wait $PID 2>/dev/null; reason=freeze; break; fi
  done
  wait $PID 2>/dev/null; rc=$?
  [ -n "$reason" ] && continue
  [ -f $CK/summary.json ] && continue
  if [ "$(last_step)" = "$s_start" ]; then fails=$((fails + 1)); else fails=0; fi
  [ $fails -ge 5 ] && { say "STOP: 5 restarts in a row without progress (see $LOG)"; state "STOPPED: 5 restarts without progress"; exit 1; }
  say "training process ended (rc $rc) without finishing; restart in 120 s"; sleep 120
done
