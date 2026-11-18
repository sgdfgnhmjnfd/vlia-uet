#!/usr/bin/env bash
# Stage 17 Step 2 chain: pause VLIA 100k at its next complete checkpoint (chain PAUSE file, as Stage 16), embed hand crops
# on the GPU, remove PAUSE at once, verify the resume step, then analyze on CPU. Resumable.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
L=/media/dhqg/d3/logs; ST=$L/stage17_step2_status.txt
O=/media/dhqg/d3/vlia_outputs/train_100k/vlia; CK=$O/intention_only/seed_0
TLOG=$L/train100k_vlia.log; CLOG=$L/train100k_vlia_chain.log
STATE=/media/dhqg/d3/vlia_outputs/stage17_hoi/STATE.md; PC=$REPO/results/stage17_hoi/pause_check.md
say() { echo "$(date '+%F %T') $*" | tee -a $ST; }
if [ ! -f /media/dhqg/d3/vlia_outputs/stage17_hoi/emb_ii/t9_r3_p3.npz ]; then
  last=$(grep -oE "^step [0-9]+" $TLOG | tail -1 | grep -oE "[0-9]+" | sed 's/^0*//')
  lastck=$(ls -d $CK/checkpoint_* | sort | tail -1 | xargs basename)
  say "before pause: last logged step $last, last checkpoint $lastck; PAUSE -> stop after next complete checkpoint"
  echo "$(date '+%F %T') | Step 2b pause | before pause: last step $last, last checkpoint $lastck | - | -" >> $STATE
  touch $O/PAUSE
  until grep -q "PAUSED after" <(tail -3 $CLOG) && ! kill -0 $(cat $O/train.pid) 2>/dev/null && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 10; done
  pck=$(grep "PAUSED after" $CLOG | tail -1 | grep -oE "checkpoint_[0-9]+")
  stopstep=$(grep -oE "^step [0-9]+" $TLOG | tail -1 | grep -oE "[0-9]+" | sed 's/^0*//')
  say "VLIA paused after $pck (last logged step $stopstep); embed start"
  echo "paused_after=$pck last_logged_step=$stopstep" > $O/../stage17_pause.txt
  (cd $REPO && $PY scripts/run_stage17_step2.py embed) >> $L/stage17_embed.log 2>&1 || { say "embed FAILED (PAUSE kept)"; exit 1; }
  say "embed done"
fi
rm -f $O/PAUSE && say "PAUSE removed"
source $O/../stage17_pause.txt
want=${paused_after#checkpoint_}; want=$((10#$want))
until grep -q "data order: resume at step $want:" <(tail -c 50000 $TLOG) && grep -qE "^step 0*$((want + 10)) " <(tail -c 20000 $TLOG); do sleep 15; done
res_line=$(grep "start training --resume" $CLOG | tail -1)
redo=$(( ${last_logged_step:-$want} - want ))
{ echo "# Stage 17 pause check"; echo; echo "- paused after: $paused_after (last logged step before stop: $last_logged_step)";
  echo "- chain: $res_line"; echo "- training log: $(grep 'data order: resume' $TLOG | tail -1)";
  echo "- steps before pause (checkpoint) = $want, resumed from = $want, lost = 0 (steps $want-$last_logged_step after the checkpoint were recomputed: $redo steps)"; } > $PC
cat $PC | tee -a $ST
echo "$(date '+%F %T') | Step 2b resume check | paused after $paused_after, resumed from step $want, lost 0 (recomputed $redo) | results/stage17_hoi/pause_check.md | -" >> $STATE
(cd $REPO && CUDA_VISIBLE_DEVICES="" nice -n 5 $PY scripts/run_stage17_step2.py analyze) >> $L/stage17_analyze.log 2>&1 || { say "analyze FAILED"; exit 1; }
say "STEP2 DONE"
