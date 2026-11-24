#!/usr/bin/env bash
# Bosung 20/11 (results/bosung_2026-11-20/prereg.md). Run the same command again after a power cut: finished episodes
# are skipped (atomic files), finished blocks are skipped (DONE markers).
#   bash scripts/run_sau_train_2026-11-20.sh [--dry-run]
# Waits until VLIA training is over (checkpoint_100000, FROZEN or STOPPED_BY_TRANG) and the C rerun (tmux bosung_c) is
# done and the GPU has been empty for 10 minutes; then G0 -> block 1 (C closed loop) -> 2 (seed 2000) -> 3 (seed 3000)
# -> 4 (planned rows via run_final_eval_100k.sh). Block 5 (seed 1 VLIA) is never started here.
# Estimates: /media/dhqg/d3/vlia_outputs/bosung_2026-11-20/uoc_luong.txt
set -uo pipefail
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
VPY=/home/dhqg/lerobot/.venv/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
B=/media/dhqg/d3/vlia_outputs/bosung_2026-11-20
RES=$REPO/results/bosung_2026-11-20
LOG=/media/dhqg/d3/logs/bosung_2026-11-20.log
EST=$B/uoc_luong.txt
STATE=/media/dhqg/d3/vlia_outputs/ke_hoach_cuoi/STATE.md
T100=/media/dhqg/d3/vlia_outputs/train_100k/vlia
V25=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000
B25=/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000
# final VLIA milestone by the EVAL_PLAN rule: the largest of 100k / 75k / 50k that exists (decided when blocks start)
VFINAL=""; for m in 100000 075000 050000; do [ -d $T100/intention_only/seed_0/checkpoint_$m ] && { VFINAL=$T100/intention_only/seed_0/checkpoint_$m; break; }; done
OLD_VLIA=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/episodes
V2=/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings_v2.npz
CLTOK=$RES/e2e_sup_cl_tokens.json
PROC_SRC=/media/dhqg/d3/lerobot_outputs/smolvla_libero/checkpoints/025000/pretrained_model
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
DEADLINE=$(date -d '2026-11-22 20:00' +%s)
mkdir -p $B
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
state() { echo "$(date '+%F %T') | bosung Phan 2 | $* | $B | -" >> $STATE; }
copy_processors() { for f in "$PROC_SRC"/policy_preprocessor* "$PROC_SRC"/policy_postprocessor*; do
  [ -e "$1/pretrained_model/$(basename "$f")" ] || cp "$f" "$1/pretrained_model/"; done; }

# rollout CKPT OUT ROW SEED_BASE EPT [extra args]
rollout() {
  local ck=$1 out=$2 row=$3 sb=$4 ept=$5; shift 5
  local args=(--out $out --conditions $row --episodes-per-task $ept --seed-base $sb --task-embeddings $V2 "$@")
  if [ $DRY = 1 ]; then echo "  [dry] $row seed $sb+e, $ept/task, ckpt $(basename $ck) -> $out"
    (cd $REPO && $VPY scripts/libero_goal_ablation.py "${args[@]}" --dry-run) 2>&1 | sed 's/^/    /'; return 0; fi
  mkdir -p $out
  say "rollout $row seed_base $sb ckpt $ck ($(ls $out/episodes 2>/dev/null | grep -c json) done) -> $out"
  (unset PYTHONPATH; SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
   export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0 HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot PYTHONPATH="$SITEPKG:$REPO"
   cd /home/dhqg/lerobot && $VPY $REPO/scripts/libero_goal_ablation.py "${args[@]}" \
     --policy.path=$ck/pretrained_model --env.type=libero --env.task=libero_10 \
     --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME") >> $LOG 2>&1 || { say "FAILED $row (see $LOG)"; state "FAILED $row"; exit 1; }
  say "done $row: $(ls $out/episodes | grep -c json) episode files"
}
done_block() { [ -f $B/DONE_$1 ]; }
mark() { [ $DRY = 1 ] || { touch $B/DONE_$1; state "block $1 done"; [ -f $REPO/scripts/analyze_bosung.py ] && (cd $REPO && $PY scripts/analyze_bosung.py >> $LOG 2>&1); }; }

say "start (dry-run $DRY, repo $(git -C $REPO rev-parse --short HEAD))"
# ---------- wait ----------
if [ $DRY = 0 ]; then
  free=0
  while true; do
    tr_over=0; { [ -f $T100/intention_only/seed_0/summary.json ] || [ -f $T100/FROZEN ] || [ -f $T100/STOPPED_BY_TRANG ]; } && tr_over=1
    c_over=0; grep -q "^EXIT" /media/dhqg/d3/logs/bosung_c_predict.log 2>/dev/null && c_over=1
    if [ $tr_over = 1 ] && [ $c_over = 1 ] && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
      free=$((free + 1)); [ $free -ge 2 ] && break
    else free=0; say "waiting: training over $tr_over, C rerun over $c_over, GPU $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) process(es)"; fi
    sleep 300
  done
  for m in 100000 075000 050000; do [ -d $T100/intention_only/seed_0/checkpoint_$m ] && { VFINAL=$T100/intention_only/seed_0/checkpoint_$m; break; }; done
  say "GPU free for 10 min; start; final VLIA milestone $(basename $VFINAL)"; state "wait over, start G0; final VLIA milestone $(basename $VFINAL)"
  for c in $V25 $B25 $VFINAL; do copy_processors $c; done
fi
CL_OK=1; { [ -f $B/c_rerun/MISMATCH ] || ! grep -q "^EXIT 0" /media/dhqg/d3/logs/bosung_c_predict.log 2>/dev/null; } && CL_OK=0
[ $DRY = 1 ] && CL_OK=1

# ---------- G0 ----------
if ! done_block G0; then
  rollout $V25 $B/probe/vlia_25000_I-only_s1000 I-only 1000 1
  rollout $V25 $B/probe/vlia_25000_I-only_s2000 I-only 2000 1
  if [ $DRY = 0 ]; then
    $PY - $B $OLD_VLIA <<'EOF' | tee -a $LOG
import json, sys, glob
B, old = sys.argv[1], sys.argv[2]
def ld(d): return {(r["task"]): r for r in (json.load(open(p)) for p in glob.glob(f"{d}/episodes/*.json"))}
a, b = ld(f"{B}/probe/vlia_25000_I-only_s1000"), ld(f"{B}/probe/vlia_25000_I-only_s2000")
o = {t: json.load(open(f"{old}/I-only_t{t}_e0.json")) for t in range(10)}
same_old_succ = sum(a[t]["success"] == o[t]["success"] for t in range(10))
same_old_steps = sum(a[t]["n_steps"] == o[t]["n_steps"] for t in range(10))
ident = sum(a[t]["first_actions"] == b[t]["first_actions"] and a[t]["n_steps"] == b[t]["n_steps"] for t in range(10))
secs = sum(r["seconds"] for r in list(a.values()) + list(b.values())) / 20
g0 = {"seed1000_vs_old_success_equal": same_old_succ, "seed1000_vs_old_steps_equal": same_old_steps,
      "seed2000_identical_to_seed1000": ident, "seconds_per_episode": round(secs, 2), "seed_has_effect": ident < 10}
json.dump(g0, open(f"{B}/g0.json", "w"), indent=1); print("G0", json.dumps(g0))
EOF
  fi
  mark G0
fi
SPE=10; SEED_OK=1
if [ -f $B/g0.json ]; then
  SPE=$($PY -c "import json;print(json.load(open('$B/g0.json'))['seconds_per_episode'])")
  SEED_OK=$($PY -c "import json;print(int(json.load(open('$B/g0.json'))['seed_has_effect']))")
fi
# ---------- order and cut rule ----------
N1=299; N2=600; N3=600; N4=3088
ORDER="1 2 3 4"
[ $SEED_OK = 0 ] && { ORDER="1 4"; say "G0: seed 2000 identical to seed 1000 on all 10 units -> blocks 2-3 dropped"; state "seed has no effect, blocks 2-3 dropped"; }
[ $CL_OK = 0 ] && { ORDER=$(echo $ORDER | sed 's/^1 //'); say "C rerun mismatch or failed -> block 1 NOT run (waits for Trang)"; state "block 1 held: C rerun mismatch/failed"; }
# ONLY_BLOCKS="1 2 3" (env): run only these blocks now (Trang 21/11: blocks 1-3 at 70k pause, block 4 after VLIA 100k)
if [ -n "${ONLY_BLOCKS:-}" ]; then o=""; for b in $ORDER; do [[ " $ONLY_BLOCKS " == *" $b "* ]] && o="$o $b"; done; ORDER=$(echo $o); say "ONLY_BLOCKS: order $ORDER"; fi
end_of() { local s=0 b; for b in $1; do case $b in 1) s=$((s+N1));; 2) s=$((s+N2));; 3) s=$((s+N3));; 4) s=$((s+N4));; esac; done
  $PY -c "import time;print(int(time.time()+$s*$SPE))"; }
if [ $SEED_OK = 1 ] && [ "$(end_of "$ORDER")" -gt $DEADLINE ]; then
  ORDER=$(echo $ORDER | sed 's/ 3//')" 3"; say "cut rule: block 3 moved after block 4"
  [ "$(end_of "$(echo $ORDER | sed 's/ 3$//')")" -gt $DEADLINE ] && { ORDER=$(echo $ORDER | sed 's/ 2//')" 2"; say "cut rule: block 2 moved to the end"; }
  state "cut rule applied: order $ORDER"
fi
{ echo "updated $(date '+%F %T'); seconds/episode $SPE; order $ORDER"
  t=$(date +%s); for b in $ORDER; do eval n=\$N$b; t=$((t + n * ${SPE%.*})); echo "block $b: $n episodes, ends about $(date -d @$t '+%F %H:%M')"; done; } > $EST
cat $EST | tee -a $LOG

# ---------- blocks ----------
for b in $ORDER; do
  done_block $b && continue
  case $b in
    1) rollout $V25 $B/vlia_25000/E2E-sup-p25-CL E2E-sup-p25-CL 1000 30 --e2e-tokens $CLTOK ;;
    2|3) sb=$(( b == 2 ? 2000 : 3000 ))
       rollout $B25 $B/baseline_25000/correct_s$sb correct $sb 30
       rollout $V25 $B/vlia_25000/I-only_s$sb I-only $sb 30 ;;
    4) D=""; [ $DRY = 1 ] && D=--dry-run
       bash $REPO/scripts/run_final_eval_100k.sh baseline $B25 L-para,L-goal,L-E2E,L-sup-p25,L-gen-p25 $D || exit 1
       bash $REPO/scripts/run_final_eval_100k.sh vlia $V25 E2E-gen-p25 $D || exit 1
       bash $REPO/scripts/run_final_eval_100k.sh vlia $VFINAL $D || exit 1
       # learning curve (3): 10 episodes/task at every milestone below the final one (25k = existing I-only e0-9)
       for m in 050000 075000; do c=$T100/intention_only/seed_0/checkpoint_$m
         [ -d $c ] && [ "$c" != "$VFINAL" ] && [[ "$c" < "$VFINAL" ]] && { copy_processors $c; bash $REPO/scripts/run_final_eval_100k.sh vlia $c --curve $D || exit 1; }; done ;;
  esac
  mark $b
  # refresh estimate
  [ $DRY = 1 ] || { echo "$(date '+%F %T') block $b done" >> $EST; }
done
say "ALL BLOCKS DONE (order $ORDER${ONLY_BLOCKS:+, only $ONLY_BLOCKS}). Block 5 (seed 1 VLIA) is not started here."
[ $DRY = 1 ] || state "all blocks done (order $ORDER)"
