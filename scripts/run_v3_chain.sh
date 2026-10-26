#!/usr/bin/env bash
# prereg v3 unattended chain (resumable; re-run after an interruption):
#   1. wait for the Stage 3 MLLM scoring, then analyse it (CPU)
#   2. Stage 4 eval-path smoke on the dry-run checkpoint (1 episode, I-only)
#   3. Stage 4 probe: 200 training steps -> s/step; ETA(25k) <= 40 h else STOP
#   4. Stage 4 matched 25k training (resumes from the latest checkpoint)
#   5. Stage 4 evaluation (I-only, I-wrong; 10 tasks x 10 episodes) + analysis
# Status lines: /media/dhqg/d3/logs/v3_status.txt
set -uo pipefail
REPO=/media/dhqg/d3/vlia-uet
LOG=/media/dhqg/d3/logs
STATUS=$LOG/v3_status.txt
OUT=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1
TRAIN_ROOT=$OUT/train
CKPT_DIR=$TRAIN_ROOT/intention_only/seed_0
PROC_SRC=/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
VPY=/home/dhqg/lerobot/.venv/bin/python
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
export HF_HOME=/media/dhqg/d3/hf_cache TMPDIR=/media/dhqg/d3/tmp
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

venv_env() {  # environment of the original LIBERO training / eval
  unset PYTHONPATH
  SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
  export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0
  export HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot
  export HF_HUB_OFFLINE=1  # all models are cached; avoid transient hub errors
  export PYTHONPATH="$SITEPKG:$REPO"
}

copy_processors() {  # same pre/post-processors as the 24% baseline eval
  for f in "$PROC_SRC"/policy_preprocessor* "$PROC_SRC"/policy_postprocessor*; do
    [ -e "$1/$(basename "$f")" ] || cp "$f" "$1/"
  done
}

say "chain start"
# ---- 1. Stage 3 ----
until grep -q '"eta_full_hours"' $LOG/v3_stage3_full.log 2>/dev/null; do
  if ! pgrep -f "bin/python scripts/v3_stage3_mllm.py" > /dev/null; then
    say "Stage 3 scoring not running -> (re)start (resumable)"
    (cd $REPO && nohup $PY scripts/v3_stage3_mllm.py >> $LOG/v3_stage3_full.log 2>&1 &)
  fi
  sleep 120
done
if [ ! -f $REPO/results/v3_stage3_mllm/summary.md ]; then
  (cd $REPO && $PY scripts/v3_stage3_mllm.py --analyze > $LOG/v3_stage3_analyze.log 2>&1) \
    && say "Stage 3 analysis done" || { say "Stage 3 analysis FAILED"; }
fi

# ---- 2. eval-path smoke on the dry-run checkpoint ----
SMOKE_CKPT=/media/dhqg/d3/tmp/intention_only_dryrun/intention_only/seed_0/checkpoint_000002/pretrained_model
if [ ! -f $OUT/eval_smoke/probe/I-only_t0_e0.json ]; then
  copy_processors $SMOKE_CKPT
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out $OUT/eval_smoke --probe 1 \
     --conditions I-only --policy.path=$SMOKE_CKPT --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" > $LOG/v3_stage4_eval_smoke.log 2>&1) \
    && say "Stage 4 eval-path smoke OK" || { say "Stage 4 eval-path smoke FAILED (see v3_stage4_eval_smoke.log)"; exit 1; }
fi

# ---- 3. probe ----
if [ ! -f $OUT/probe.json ]; then
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/train_intention_only_LIBERO7D.py --condition intention_only \
     --output-root /media/dhqg/d3/tmp/intention_only_probe --steps 200 --batch-size 32 --save-freq 100000 \
     --log-freq 10 --seed 0 > $LOG/v3_stage4_probe.log 2>&1) || { say "Stage 4 probe FAILED"; exit 1; }
  $PY - <<EOF
import json, re
txt = open("$LOG/v3_stage4_probe.log").read()
total = json.loads(txt[txt.rindex("{"):txt.rindex("}") + 1])["total_time_s"]
steps = [int(m) for m in re.findall(r"step (\d+) \|", txt)]
el = [float(x) for x in re.findall(r'"elapsed_s": ([0-9.]+)', txt)]
sps = total / 200
json.dump({"probe_steps": 200, "total_time_s": total, "seconds_per_step": sps, "eta_25k_hours": sps * 25000 / 3600,
           "baseline_25k_elapsed_s": 46129.0}, open("$OUT/probe.json", "w"), indent=2)
EOF
  say "Stage 4 probe: $(cat $OUT/probe.json | tr -d '\n ')"
fi
ETA=$($PY -c "import json;print(json.load(open('$OUT/probe.json'))['eta_25k_hours'])")
if ! $PY -c "import sys; sys.exit(0 if $ETA <= 40 else 1)"; then
  say "Stage 4 STOP: ETA ${ETA} h > 40 h (user decides)"; exit 0
fi

# ---- 4. matched 25k training (resumable) ----
if [ ! -f $CKPT_DIR/summary.json ]; then
  LAST=$(ls -d $CKPT_DIR/checkpoint_* 2>/dev/null | sort | tail -1)
  RESUME=()
  [ -n "$LAST" ] && RESUME=(--resume "$LAST") && say "resume training from $LAST"
  say "Stage 4 training start (ETA ${ETA} h)"
  (venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/train_intention_only_LIBERO7D.py --condition intention_only \
     --output-root $TRAIN_ROOT --steps 25000 --batch-size 32 --save-freq 1000 --log-freq 10 --seed 0 "${RESUME[@]}" \
     >> $LOG/v3_stage4_train.log 2>&1) || { say "Stage 4 training FAILED"; exit 1; }
  say "Stage 4 training done"
fi

# ---- 5. evaluation + analysis ----
FINAL=$CKPT_DIR/checkpoint_025000/pretrained_model
copy_processors $FINAL
(venv_env; cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py --out $OUT/eval --episodes-per-task 10 \
   --conditions I-only,I-wrong --policy.path=$FINAL --env.type=libero --env.task=libero_10 \
   --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" >> $LOG/v3_stage4_eval.log 2>&1) \
  || { say "Stage 4 eval FAILED"; exit 1; }
(cd $REPO && $PY scripts/libero_goal_ablation.py --analyze-intention --out $OUT/eval \
   --results results/libero_intention_only_v1 > $LOG/v3_stage4_analyze.log 2>&1) \
  && say "Stage 4 analysis done; ALL DONE" || say "Stage 4 analysis FAILED"
