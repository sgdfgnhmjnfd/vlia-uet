#!/usr/bin/env bash
# Stage 15 unattended chain (resumable; re-run after an interruption, finished steps are skipped):
#   waits for a free GPU, then extract (agent, wrist) -> cv agent -> plan -> smoke -> E2E-sup rollouts p25, p50
#   -> cv wrist -> analyze. Stops at the first failed gate. Status: /media/dhqg/d3/logs/stage15_status.txt
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=/media/dhqg/d3/logs
STATUS=$LOG/stage15_status.txt
OUT=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1
W=/media/dhqg/d3/vlia_outputs/stage15
CK0=$OUT/train/intention_only/seed_0/checkpoint_025000/pretrained_model
V2=$OUT/task_embeddings_v2.npz
TOK=$REPO/results/stage15_supervised_early/e2e_sup_tokens.json
PROC_SRC=/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
VPY=/home/dhqg/lerobot/.venv/bin/python
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
venv_env() {
  unset PYTHONPATH
  SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
  export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0
  export HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot
  export PYTHONPATH="$SITEPKG:$REPO"
}
copy_processors() {
  for f in "$PROC_SRC"/policy_preprocessor* "$PROC_SRC"/policy_postprocessor*; do
    [ -e "$1/$(basename "$f")" ] || cp "$f" "$1/"
  done
}
S15() { (cd $REPO && $PY scripts/run_stage15_supervised.py "$@"); }
free_root() { [ "$(df --output=avail -m / | tail -1)" -gt 500 ] || { say "STOP: less than 500 MB free on /"; exit 1; }; }

say "chain start ($(git -C $REPO rev-parse --short HEAD))"
while pgrep -f "train_intention_only_LIBERO7D|libero_goal_ablation.py --out" > /dev/null; do sleep 60; done
say "GPU free"
free_root
S15 selftest > $LOG/stage15_selftest.log 2>&1 || { say "selftest FAILED"; exit 1; }
for cam in agent wrist; do
  S15 extract --camera $cam >> $LOG/stage15_extract_$cam.log 2>&1 || { say "extract $cam FAILED"; exit 1; }
  say "extract $cam done"
done
free_root
if [ ! -f $W/oof_agent.npz ]; then
  S15 cv --camera agent > $LOG/stage15_cv_agent.log 2>&1 || { say "cv agent FAILED (see stage15_cv_agent.log)"; exit 1; }
fi
say "cv agent done: $(grep -E 'GATE' $LOG/stage15_cv_agent.log | tr '\n' ' ')"
S15 plan > $LOG/stage15_plan.log 2>&1 || { say "plan FAILED"; exit 1; }
say "plan: $(grep -E '^E2E-sup' $LOG/stage15_plan.log | cut -d: -f1-2 | tr '\n' ' ')"
copy_processors $CK0
if ! grep -q "smoke OK" $STATUS; then
  rm -rf /media/dhqg/d3/tmp/stage15_smoke
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out /media/dhqg/d3/tmp/stage15_smoke --probe 1 \
     --conditions E2E-sup-p25 --e2e-tokens $TOK --task-embeddings $V2 --policy.path=$CK0 --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" > $LOG/stage15_smoke.log 2>&1) || { say "smoke FAILED"; exit 1; }
  $PY - <<EOF || { say "smoke FAILED (token)"; exit 1; }
import json
r = json.load(open("/media/dhqg/d3/tmp/stage15_smoke/probe/E2E-sup-p25_t0_e0.json"))
plan = json.load(open("$TOK"))
assert r["z_int_task"] == plan["tokens"]["E2E-sup-p25"]["0,0"] and r["instruction"] == "", r
EOF
  say "smoke OK"
fi
for p in 25 50; do
  free_root
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out $OUT/eval --episodes-per-task 30 \
     --conditions E2E-sup-p$p --e2e-tokens $TOK --task-embeddings $V2 --policy.path=$CK0 --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" >> $LOG/stage15_rollouts.log 2>&1) || { say "rollouts p$p FAILED"; exit 1; }
  say "rollouts p$p done"
done
if [ ! -f $W/oof_wrist.npz ]; then
  S15 cv --camera wrist > $LOG/stage15_cv_wrist.log 2>&1 || { say "cv wrist FAILED"; exit 1; }
fi
say "cv wrist done"
S15 analyze > $LOG/stage15_analyze.log 2>&1 || { say "analyze FAILED"; exit 1; }
say "STAGE15 DONE"
