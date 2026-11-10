#!/usr/bin/env bash
# Token matrix chain (Stage 13 amendment 1). Resumable: re-run after any interruption; finished episode JSONs are skipped.
#   0. wait until the seed-1 evaluation is complete (or its chain failed) and no other GPU job is running
#   1. smoke (1 I-tok episode), then seed-0 matrix: I-tok for every task into $OUT/eval, build, analyze, push
#   2. seed-1 matrix into $OUT/eval_seed1 (only if the seed-1 checkpoint exists and passed its smoke), build, analyze, push
# Log: /media/dhqg/d3/logs/token_matrix.log; one status line per finished task: /media/dhqg/d3/logs/token_matrix_status.txt
set -uo pipefail
REPO=/media/dhqg/d3/vlia-libero2
LOG=/media/dhqg/d3/logs
STATUS=$LOG/token_matrix_status.txt
OUT=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1
PROC_SRC=/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
VPY=/home/dhqg/lerobot/.venv/bin/python
V2=$OUT/task_embeddings_v2.npz
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

venv_env() {  # environment of the original LIBERO training / eval
  unset PYTHONPATH
  SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
  export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0
  export HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot
  export PYTHONPATH="$SITEPKG:$REPO"
}

copy_processors() {  # same pre/post-processors as the 24% baseline eval
  for f in "$PROC_SRC"/policy_preprocessor* "$PROC_SRC"/policy_postprocessor*; do
    [ -e "$1/$(basename "$f")" ] || cp "$f" "$1/"
  done
}

rollouts() {  # $1 checkpoint, $2 out dir, $3 task, $4 extra args
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out "$2" --episodes-per-task 30 \
     --conditions I-tok --token-tasks 0-9 --tasks "$3" $4 --task-embeddings $V2 --policy.path="$1" --env.type=libero \
     --env.task=libero_10 --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" >> $LOG/token_matrix.log 2>&1)
}

gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

matrix() {  # $1 seed, $2 checkpoint, $3 eval dir
  copy_processors "$2"
  for t in 0 1 2 3 4 5 6 7 8 9; do
    n=$(ls $3/episodes/I-tok*_t${t}_e*.json 2>/dev/null | wc -l)
    [ "$n" -ge 240 ] && continue
    for attempt in 1 2 3; do
      rollouts "$2" "$3" "$t" "" && break
      say "seed $1 task $t attempt $attempt FAILED"; sleep 60
    done
    n=$(ls $3/episodes/I-tok*_t${t}_e*.json 2>/dev/null | wc -l)
    say "seed $1 task $t done: $n/240 I-tok episodes"
    [ "$n" -ge 240 ] || { say "seed $1 task $t incomplete; STOP"; exit 1; }
  done
  (cd $REPO && $PY scripts/token_matrix.py build --seed "$1" >> $LOG/token_matrix.log 2>&1) || { say "seed $1 build FAILED (missing entries)"; exit 1; }
  (cd $REPO && $PY scripts/token_matrix.py analyze > $LOG/token_matrix_analyze_seed$1.log 2>&1) || { say "seed $1 analyze FAILED"; exit 1; }
  (cd $REPO && git add results/token_matrix && git commit -qm "token matrix: seed-$1 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD:refs/heads/claude/stage13-human-video) \
    && say "seed $1 matrix analysed and pushed" || say "seed $1 commit/push FAILED"
}

say "token matrix chain start"
S1=$LOG/seed1_status.txt
until { [ "$(ls $OUT/eval_seed1/episodes/I-only_*.json $OUT/eval_seed1/episodes/I-wrong_*.json $OUT/eval_seed1/episodes/E2E_*.json 2>/dev/null | wc -l)" -ge 899 ] \
        || grep -q "FAILED" $S1 2>/dev/null; } && ! gpu_busy; do
  sleep 300
done
grep -q "FAILED" $S1 2>/dev/null && say "seed-1 chain reported FAILED; seed-0 matrix starts anyway"
say "queue released (seed-1 eval files: $(ls $OUT/eval_seed1/episodes 2>/dev/null | wc -l))"
CK0=$OUT/train/intention_only/seed_0/checkpoint_025000/pretrained_model
if [ ! -f /media/dhqg/d3/tmp/token_matrix_smoke/probe/I-tok1_t0_e0.json ]; then
  copy_processors $CK0
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out /media/dhqg/d3/tmp/token_matrix_smoke --probe 1 \
     --conditions I-tok --task-embeddings $V2 --policy.path=$CK0 --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" > $LOG/token_matrix_smoke.log 2>&1) || { say "smoke FAILED"; exit 1; }
fi
$PY - <<'PYEOF' || { say "smoke check FAILED"; exit 1; }
import json
r = json.load(open("/media/dhqg/d3/tmp/token_matrix_smoke/probe/I-tok1_t0_e0.json"))
o = json.load(open("/media/dhqg/d3/vlia-libero2/results/libero_goal_ablation_v1/instructions.json"))["original"]["1"]
assert r["token_task"] == 1 and r["z_int_task"] == o and r["instruction"] == "", r
PYEOF
say "smoke OK (I-tok1 on task 0: token_task 1, z_int_task = task 1 description)"
matrix 0 $CK0 $OUT/eval
CK1=$OUT/train/intention_only/seed_1/checkpoint_025000/pretrained_model
if [ -f $CK1/model.safetensors ] && grep -q "smoke OK" $S1 2>/dev/null; then
  until ! gpu_busy; do sleep 300; done
  matrix 1 $CK1 $OUT/eval_seed1
  say "TOKEN MATRIX DONE"
else
  say "seed-1 checkpoint missing or its smoke did not pass; seed-1 matrix skipped. TOKEN MATRIX DONE (seed 0 only)"
fi
