#!/usr/bin/env bash
# One VLIA episode in a LIBERO-10 scene, goal given only by the intention sentence (empty instruction).
#   bash app/run_rollout.sh OUT_DIR TASK "SENTENCE" CHECKPOINT
# Environments (same split as in the experiments): PY builds the token, VPY (the lerobot venv) runs LIBERO.
set -euo pipefail
OUT=$1; TASK=$2; SENTENCE=$3; CKPT=$4
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
VPY=${VPY:-$HOME/lerobot/.venv/bin/python}
LEROBOT_DIR=${LEROBOT_DIR:-$HOME/lerobot}
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
# CPU mode (CUDA_VISIBLE_DEVICES set but empty): policy on CPU; MuJoCo still renders with EGL on GPU 0,
# because robosuite reads the EGL device from CUDA_VISIBLE_DEVICES and fails on an empty string.
DEVICE_ARGS=()
if [[ -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  export MUJOCO_EGL_DEVICE_ID=0
  DEVICE_ARGS=(--policy.device=cpu)
fi

"$PY" "$REPO/app/make_token.py" "$OUT" "$TASK" "$SENTENCE"
(
  unset PYTHONPATH
  SITEPKG=$("$VPY" -c 'import site; print(site.getsitepackages()[0])')
  export PYTHONPATH="$SITEPKG:$REPO"
  cd "$LEROBOT_DIR"
  "$VPY" "$REPO/scripts/libero_goal_ablation.py" --out "$OUT" --conditions E2E-gen --tasks "$TASK" \
    --episodes-per-task 1 --e2e-tokens "$OUT/tokens.json" --task-embeddings "$OUT/token.npz" \
    --policy.path="$CKPT" --env.type=libero --env.task=libero_10 \
    --eval.n_episodes=1 --eval.batch_size=1 --rename_map "$RENAME" "${DEVICE_ARGS[@]}"
)
