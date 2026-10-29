# Notes for agents and collaborators

This repo holds the code and evidence for the VLIA thesis (intention module + SmolVLA intention token). Read `README.md` first: it lists every stage, its script and its results folder.

## Layout rules

- Active scripts live in `scripts/`, one script (or one subcommand) per stage, each with a synthetic self-test (`scripts/test_*.py` or a `selftest` subcommand). Legacy scripts are in `scripts/archive/`; do not import from there.
- `results/` is evidence for the thesis. Never move, rename or rewrite a results folder or a number in it. New experiments get a new folder.
- `policies/smolvla/` follows the LeRobot module layout. The intention token is behind `use_intention_token`; with the flag off the model must match stock SmolVLA.
- Large checkpoints and datasets are not committed (see `.gitignore`). Thesis and paper drafts are also kept out of this public repo until submission.

## Lab machine (`dhqg-Precision-3680`)

- Main checkout: `~/vlia-uet` (= `/media/dhqg/d3/vlia-uet`). Its local `main` may hold unpushed work. **Never switch branches there.** Run side experiments in a separate worktree, for example `git -C /media/dhqg/d3/vlia-uet worktree add --detach /media/dhqg/d3/vlia-lit origin/main`.
- Always `export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1`. The home disk is nearly full: keep at least 500 MB free on `/`, write logs to `/media/dhqg/d3/logs/` and outputs to `/media/dhqg/d3/vlia_outputs/`.
- Python for analysis: `/home/dhqg/anaconda3/envs/lerobot/bin/python`. For LIBERO training and rollouts: `/home/dhqg/lerobot/.venv/bin/python` with cwd `/home/dhqg/lerobot` (see `venv_env` in `scripts/run_v3_chain.sh`). Do not pip install into either.
- One GPU job at a time. Check `nvidia-smi` before starting anything on the GPU.

## Evaluation rules

- **Never read or score the EgoIntent locked test split.** It was used once (`results/locked_test_final_v1/`) and is closed.
- Write the rules (grid, selection, pass/fail) in `results/prereg_*.md` and commit them before running. Record any deviation next to the results (for example `DEVIATIONS.md`).
- Use scene-grouped cross-validation, nested selection and the video-level bootstrap already implemented in `scripts/frozen_eval_lib.py` and `scripts/v3_common.py`. Reproduce the reference numbers (gates) before reporting anything new.
- Every number in a document must trace to a file under `results/`.
