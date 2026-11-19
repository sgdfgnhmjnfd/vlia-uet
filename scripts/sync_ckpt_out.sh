#!/usr/bin/env bash
# Usage: scripts/sync_ckpt_out.sh DEST [SRC] [INTERVAL_S]   e.g. DEST=dhqg@100.104.152.61:/media/dhqg/d3/vlia_outputs/train_100k_rented/
#
# Every INTERVAL_S seconds (default 1200 = 20 min), copy every COMPLETE checkpoint and every new episode/result file under SRC
# (default /media/dhqg/d3/vlia_outputs/train_100k) to DEST with rsync -a --partial. DEST may be an ssh/Tailscale host path or a
# mounted folder (for example a mounted private Hugging Face repo). Nothing is ever deleted on DEST (no --delete).
#
# A checkpoint folder checkpoint_NNNNNN is COMPLETE when (same rule as the 100k chain):
#   pretrained_model/model.safetensors, pretrained_model/config.json and training_state.pt exist,
#   no file in it changed in the last 120 s, and the safetensors header parses with the env python.
# Incomplete checkpoints are skipped and retried at the next round. Log: one line per round on stdout.
set -uo pipefail
DEST=${1:?usage: sync_ckpt_out.sh DEST [SRC] [INTERVAL_S]}
SRC=${2:-/media/dhqg/d3/vlia_outputs/train_100k}
INTERVAL=${3:-1200}
PY=${PY:-$HOME/envs/lerobot/bin/python}
[ -x "$PY" ] || PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
SSH_OPTS=${SSH_OPTS:-"-o ServerAliveInterval=30 -o ConnectTimeout=20"}

complete() {  # $1 = checkpoint dir
  local c=$1 m=$1/pretrained_model
  [ -f "$m/model.safetensors" ] && [ -f "$m/config.json" ] && [ -f "$c/training_state.pt" ] || return 1
  [ -z "$(find "$c" -type f -newermt '-120 seconds' -print -quit)" ] || return 1
  "$PY" - "$m/model.safetensors" <<'EOF' >/dev/null 2>&1
import json, struct, sys, os
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
end = max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
assert os.path.getsize(p) == 8 + n + end
EOF
}

while true; do
  n_ok=0; n_skip=0; fail=0
  # 1. small files (logs, episode JSONs, summaries, configs), never checkpoint contents
  rsync -a --partial -e "ssh $SSH_OPTS" --prune-empty-dirs --exclude 'checkpoint_*/' --exclude '*.mp4' \
        "$SRC/" "$DEST" || fail=1
  # 2. complete checkpoints, one by one, keeping the relative path under SRC
  while IFS= read -r c; do
    if complete "$c"; then
      rel=${c#"$SRC"/}
      rsync -a --partial --relative -e "ssh $SSH_OPTS" "$SRC/./$rel" "$DEST" && n_ok=$((n_ok + 1)) || fail=1
    else
      n_skip=$((n_skip + 1))
    fi
  done < <(find "$SRC" -type d -name 'checkpoint_[0-9]*' -prune 2>/dev/null | sort)
  echo "$(date '+%F %T') sync to $DEST: $n_ok complete checkpoints synced, $n_skip incomplete skipped, rsync errors: $fail"
  sleep "$INTERVAL"
done
