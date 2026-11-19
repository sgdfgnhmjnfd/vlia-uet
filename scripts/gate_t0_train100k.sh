#!/usr/bin/env bash
# Gate T0 (100k resume check) for one model on this machine. A: 40 steps in one go; B: 20 steps, stop, resume, 20 more.
# Usage: gate_t0_train100k.sh vlia|baseline   Output: /media/dhqg/d3/tmp/train100k_check/<model>/ (deleted by the caller when done)
set -uo pipefail
MODEL=${1:?vlia|baseline}
REPO=$(cd "$(dirname "$0")/.." && pwd)
VPY=/home/dhqg/lerobot/.venv/bin/python
export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/media/dhqg/d3/tmp
T=/media/dhqg/d3/tmp/train100k_check/$MODEL; mkdir -p $T
case $MODEL in baseline) S=train_matched_oracle_LIBERO7D_FIXED.py; C=baseline;; vlia) S=train_intention_only_LIBERO7D.py; C=intention_only;; esac
ARGS="--condition $C --steps 100000 --batch-size 32 --lr 1e-4 --min-lr 2.5e-6 --warmup-steps 1000 --weight-decay 1e-4 --seed 0 --save-freq 20 --log-freq 1"
venv_env() { unset PYTHONPATH; SITEPKG=$($VPY -c 'import site; print(site.getsitepackages()[0])')
  export VLIA_ORACLE_ROOT=/media/dhqg/d3/datasets/libero_oracle/v0 HF_LEROBOT_HOME=/home/dhqg/.cache/huggingface/lerobot PYTHONPATH="$SITEPKG:$REPO"; }
train_until() { local stop=$1 out=$2 log=$3; shift 3
  (venv_env; cd /home/dhqg/lerobot && exec $VPY $REPO/scripts/$S $ARGS --output-root $out "$@") > $log 2>&1 &
  local pid=$! peak=0 m c
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    c=$(ls -d $out/*/seed_0/checkpoint_$stop 2>/dev/null | head -1)
    if [ -n "$c" ] && [ -f $c/training_state.pt ] && [ -f $c/pretrained_model/model.safetensors ]; then sleep 20; kill $pid; wait $pid 2>/dev/null; break; fi
    sleep 2
  done
  echo $peak > $log.peak_mib; }
t0=$(date +%s); train_until 000040 $T/A $T/A.log; echo $(( $(date +%s) - t0 )) > $T/A.seconds
train_until 000020 $T/B $T/B1.log
train_until 000040 $T/B $T/B2.log --resume $(ls -d $T/B/*/seed_0/checkpoint_000020)
echo "T0 RUN DONE"
