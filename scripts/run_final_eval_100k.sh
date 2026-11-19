#!/usr/bin/env bash
# Final 100k evaluation (prereg_train100k.md, ke hoach cuoi Part 4 Step 2). Written 19/11; run only after training.
#   bash scripts/run_final_eval_100k.sh vlia|baseline CKPT_DIR [ROW,ROW,...] [--curve] [--dry-run]
# CKPT_DIR = .../checkpoint_NNNNNN (holds pretrained_model/). Rows run one after another in plan order; the same command
# continues after a power cut (finished episode files are skipped; episode files are written atomically). To stop
# cleanly, `touch <out>/STOP_AFTER_ROW`: the current row finishes, then the script exits (never in the middle of a row).
# --curve: learning-curve mode, 10 episodes per task, baseline `correct` / VLIA `I-only`.
# Refuses to run while any process is on the GPU (e.g. training). Baseline checkpoints copied from the rented machine
# must come with CKPT_DIR.sha256 (sha256sum list made on the rented machine); it is checked first.
set -uo pipefail
MODEL=${1:?vlia|baseline}; CKPT=${2:?checkpoint dir}; shift 2
ROWS=""; CURVE=0; DRY=0
for a in "$@"; do case $a in --curve) CURVE=1 ;; --dry-run) DRY=1 ;; *) ROWS=$a ;; esac; done
REPO=$(cd "$(dirname "$0")/.." && pwd)
VPY=/home/dhqg/lerobot/.venv/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
PROC_SRC=/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
TABLE=/media/dhqg/d3/vlia_outputs/train_100k/eval_token_table.npz
UNITS=$REPO/results/train_100k/unit_sentences.json
E2E=$REPO/results/stage12_end_to_end/e2e_tokens.json
ORDER="I-only I-wrong correct empty L-para I-para L-goal I-goal E2E L-E2E E2E-sup-p25 L-sup-p25 E2E-gen-p25 L-gen-p25"
case $MODEL in
  vlia)     MINE="I-only I-wrong I-para I-goal E2E E2E-sup-p25 E2E-gen-p25"; CURVE_ROW=I-only ;;
  baseline) MINE="correct empty L-para L-goal L-E2E L-sup-p25 L-gen-p25"; CURVE_ROW=correct ;;
  *) echo "MODEL must be vlia or baseline"; exit 1 ;;
esac
STEP=$(basename "$CKPT" | sed 's/checkpoint_0*//')
OUTROOT=/media/dhqg/d3/vlia_outputs/train_100k/eval/${MODEL}_${STEP}$([ $CURVE = 1 ] && echo _curve)
EPT=30; [ $CURVE = 1 ] && EPT=10
if [ $CURVE = 1 ]; then RUN="$CURVE_ROW"; else RUN=""; for r in $ORDER; do
  case " $MINE " in *" $r "*) if [ -z "$ROWS" ] || [[ ",$ROWS," == *",$r,"* ]]; then RUN="$RUN $r"; fi ;; esac; done; fi
LOG=/media/dhqg/d3/logs/final_eval_${MODEL}_${STEP}$([ $CURVE = 1 ] && echo _curve).log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
say "start $MODEL step $STEP rows:$RUN episodes/task $EPT dry-run $DRY"
[ -f "$CKPT/pretrained_model/config.json" ] || { say "STOP: $CKPT/pretrained_model/config.json missing"; exit 1; }
if [ $DRY = 0 ]; then
  if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -q .; then
    say "REFUSE: a process is on the GPU (training still running?):"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tee -a $LOG; exit 1
  fi
  if [ $MODEL = baseline ] && [[ "$CKPT" != /media/dhqg/d1/* ]]; then
    [ -f "$CKPT.sha256" ] || { say "REFUSE: $CKPT.sha256 missing (make it on the rented machine: cd CKPT && find . -type f | sort | xargs sha256sum > ../checkpoint_NNNNNN.sha256)"; exit 1; }
    (cd "$CKPT" && sha256sum --quiet -c "$CKPT.sha256") || { say "REFUSE: sha256 mismatch for $CKPT"; exit 1; }
    say "sha256 OK for $CKPT"
  fi
  for f in "$PROC_SRC"/policy_preprocessor* "$PROC_SRC"/policy_postprocessor*; do
    [ -e "$CKPT/pretrained_model/$(basename "$f")" ] || cp "$f" "$CKPT/pretrained_model/"
  done
fi
for r in $RUN; do
  [ -f $OUTROOT/STOP_AFTER_ROW ] && { say "STOP_AFTER_ROW found: stop before row $r"; exit 0; }
  OUT=$OUTROOT/$r; [ $DRY = 1 ] || mkdir -p $OUT
  ARGS=(--out $OUT --conditions $r --episodes-per-task $EPT --task-embeddings $TABLE --e2e-tokens $E2E --unit-sentences $UNITS)
  if [ $DRY = 1 ]; then
    (cd $REPO && $VPY scripts/libero_goal_ablation.py "${ARGS[@]}" --dry-run) 2>&1 | tee -a $LOG; continue
  fi
  say "row $r start ($(ls $OUT/episodes 2>/dev/null | grep -c json) episodes already done)"
  (unset PYTHONPATH; SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
   export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0 HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot PYTHONPATH="$SITEPKG:$REPO"
   cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py "${ARGS[@]}" \
     --policy.path=$CKPT/pretrained_model --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME") >> $LOG 2>&1 || { say "row $r FAILED (see $LOG)"; exit 1; }
  say "row $r done ($(ls $OUT/episodes | grep -c json) episode files)"
done
say "ALL ROWS DONE ($MODEL step $STEP)"
