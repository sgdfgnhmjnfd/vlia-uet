#!/usr/bin/env bash
# Unattended remainder of job A + E1b analysis (prereg_ladder_v1 + amendment 1).
# 1) wait for the E1b v2 GPU run, analyse it (CPU)
# 2) wait for encoder downloads
# 3) 200-sample throughput probe; drop so400m, then clip-vit-large, if the
#    projected Stage 2 total (baseline actual + remaining) exceeds 3 h
# 4) extract remaining encoders, Stage 3, Stage 5
# Status lines go to $STATUS; full logs under /media/dhqg/d3/logs/.
set -uo pipefail
cd /media/dhqg/d3/vlia-uet
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_DISABLE_XET=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
LOG=/media/dhqg/d3/logs
STATUS=$LOG/jobA_status.txt
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

say "start"
until grep -q '"seconds_per_query"\|Traceback' $LOG/e1b_v2_full.log; do sleep 60; done
if grep -q Traceback $LOG/e1b_v2_full.log; then say "E1b run FAILED (see e1b_v2_full.log)"; else
  $PY scripts/run_e1b_smolvlm2_v2.py --analyze > $LOG/e1b_v2_analyze.log 2>&1 \
    && say "E1b analysis done" || say "E1b analysis FAILED"
fi

MODELS="google/siglip2-base-patch16-224 google/siglip-so400m-patch14-384 openai/clip-vit-large-patch14"
for m in $MODELS; do
  n=$(basename $m)
  until grep -q DONE $LOG/dl_$n.log 2>/dev/null; do
    if ! ps -eo args | grep -F "snapshot_download('$m'" | grep -vq grep; then  # restart a dead download
      say "restart download $m"
      nohup $PY -c "
from huggingface_hub import snapshot_download
p=snapshot_download('$m', token=False, allow_patterns=['*.json','*.safetensors','*.model','*.txt','spiece*','tokenizer*'])
print('DONE','$m',p,flush=True)
" >> $LOG/dl_$n.log 2>&1 &
    fi
    sleep 120
  done
done
say "downloads done"

ENC="google/siglip-base-patch16-384 google/siglip2-base-patch16-224 google/siglip-so400m-patch14-384 openai/clip-vit-large-patch14"
BASE_S=$($PY -c "import json;print(json.load(open('/media/dhqg/d3/vlia_outputs/ladder/siglip-base-patch16-224/meta.json'))['throughput']['wall_seconds'])")
for attempt in 1 2 3; do
  $PY scripts/extract_ladder_embeddings.py --encoders $ENC --limit 200 --preprocess-in-workers --decode-workers 8 \
    > $LOG/ladder_probe_$attempt.log 2>&1 || { say "probe FAILED"; exit 1; }
  PROJ=$(grep -o '"projected_full_wall_seconds": [0-9.]*' $LOG/ladder_probe_$attempt.log | awk '{print $2}')
  TOTAL=$($PY -c "print(int($BASE_S + $PROJ))")
  say "probe $attempt: encoders [$ENC] projected remaining ${PROJ}s, total with baseline ${TOTAL}s"
  [ "$TOTAL" -le 10800 ] && break
  if [[ $ENC == *so400m* ]]; then ENC=${ENC/google\/siglip-so400m-patch14-384/}; say "DROP so400m (projected > 3 h)"
  elif [[ $ENC == *clip-vit-large* ]]; then ENC=${ENC/openai\/clip-vit-large-patch14/}; say "DROP clip-vit-large (projected > 3 h)"
  else break; fi
done
$PY scripts/extract_ladder_embeddings.py --encoders $ENC --preprocess-in-workers --decode-workers 8 \
  > $LOG/ladder_extract_rest.log 2>&1 || { say "extraction FAILED"; exit 1; }
say "extraction done: $ENC"
$PY scripts/run_ladder_v1.py stage3 > $LOG/ladder_stage3.log 2>&1 || { say "stage3 FAILED"; exit 1; }
say "stage3 done"
$PY scripts/run_ladder_v1.py stage5 > $LOG/ladder_stage5.log 2>&1 || { say "stage5 FAILED"; exit 1; }
say "stage5 done; ALL DONE"
