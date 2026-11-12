#!/usr/bin/env bash
# Stage 14 part A unattended chain (resumable; re-run after an interruption, finished steps are skipped).
# GPU order (amendment 2026-11-12): probe -> gen-libero (6 p + blank) -> E2E-gen, E2E-gen-blank (p = 25)
#   -> E2E-gen-p50 (changed units only) -> gen-ego (k = 8, 2, 4 + blank) -> analyze.
# Status: /media/dhqg/d3/logs/stage14_status.txt
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=/media/dhqg/d3/logs
STATUS=$LOG/stage14_status.txt
OUT=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1
W=/media/dhqg/d3/vlia_outputs/stage14
CK0=$OUT/train/intention_only/seed_0/checkpoint_025000/pretrained_model
TOK=$REPO/results/stage14_generate_early/e2e_gen_tokens.json
EMB=$W/gen_embeddings.npz
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
S14() { (cd $REPO && $PY scripts/run_stage14_generate.py "$@"); }
free_root() { [ "$(df --output=avail -m / | tail -1)" -gt 500 ] || { say "STOP: less than 500 MB free on /"; exit 1; }; }
rollout() {  # condition, extra args
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out $OUT/eval --episodes-per-task 30 \
     --conditions "$1" --e2e-tokens $TOK --task-embeddings $EMB --policy.path=$CK0 --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" >> $LOG/stage14_rollouts.log 2>&1)
}

say "chain start ($(git -C $REPO rev-parse --short HEAD))"
while pgrep -f "train_intention_only_LIBERO7D|libero_goal_ablation.py --out|run_stage15_supervised" > /dev/null; do sleep 60; done
free_root
S14 selftest > $LOG/stage14_selftest.log 2>&1 || { say "selftest FAILED"; exit 1; }
grep -q "GATE M1.*PASS" $LOG/stage14_gate_m1.log 2>/dev/null || S14 gate-m1 > $LOG/stage14_gate_m1.log 2>&1 || { say "gate M1 FAILED"; exit 1; }
say "$(grep 'GATE M1' $LOG/stage14_gate_m1.log)"
if [ ! -f $W/probe/probe.json ]; then
  S14 probe > $LOG/stage14_probe.log 2>&1 || { say "probe FAILED"; exit 1; }
fi
say "probe: $($PY -c "import json;print(json.load(open('$W/probe/probe.json'))['info'])")"
free_root
S14 gen-libero --p 0,10,25,50,75,100 --blank >> $LOG/stage14_gen_libero.log 2>&1 || { say "gen-libero FAILED"; exit 1; }
say "gen-libero done"
S14 human-check > $LOG/stage14_human_check.log 2>&1 || { say "human-check FAILED"; exit 1; }
S14 plan > $LOG/stage14_plan.log 2>&1 || { say "plan FAILED (see stage14_plan.log)"; exit 1; }
say "plan: $(grep -E 'GATE E1|units' $LOG/stage14_plan.log | tr '\n' ' ')"
copy_processors $CK0
if ! grep -q "smoke OK" $STATUS; then
  rm -rf /media/dhqg/d3/tmp/stage14_smoke
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out /media/dhqg/d3/tmp/stage14_smoke --probe 1 \
     --conditions E2E-gen --e2e-tokens $TOK --task-embeddings $EMB --policy.path=$CK0 --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" > $LOG/stage14_smoke.log 2>&1) || { say "smoke FAILED"; exit 1; }
  $PY - <<EOF || { say "smoke FAILED (token)"; exit 1; }
import json
r = json.load(open("/media/dhqg/d3/tmp/stage14_smoke/probe/E2E-gen_t0_e0.json"))
assert r["z_int_task"] == json.load(open("$TOK"))["tokens"]["E2E-gen"]["0,0"] and r["instruction"] == "", r
EOF
  say "smoke OK"
fi
for c in E2E-gen E2E-gen-blank E2E-gen-p50; do
  free_root
  rollout $c || { say "rollouts $c FAILED"; exit 1; }
  say "rollouts $c done"
done
K=8,2,4
ETA=$($PY -c "import json;print(json.load(open('$W/probe/probe.json'))['info']['eta_ego_k248_hours'])")
if $PY -c "import sys; sys.exit(0 if $ETA > 12 else 1)"; then K=8,2; say "EgoIntent ETA ${ETA} h > 12 h: k = 8 and 2 only"; fi
free_root
S14 gen-ego --k $K --blank >> $LOG/stage14_gen_ego.log 2>&1 || { say "gen-ego FAILED"; exit 1; }
say "gen-ego done (k = $K)"
S14 human-check >> $LOG/stage14_human_check.log 2>&1 || { say "human-check FAILED"; exit 1; }
S14 analyze > $LOG/stage14_analyze.log 2>&1 || { say "analyze FAILED"; exit 1; }
say "STAGE14 DONE"
