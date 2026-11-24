# Pre-registration: supplementary measurements (bosung, 20/11/2026)

Written before any new rollout. Trang agreed on 20/11 (STOP A answers; 18:30: VLIA 100k training stopped at step
52,600 for lack of time, last complete checkpoint 52,500; final VLIA milestone by the EVAL_PLAN rule = 50,000).

## Purpose
(a) An independent closed-loop measurement of classifier C (Stage 15) driving VLIA.
(b) How much the evaluation itself varies with the rollout seed. This is NOT training-seed variation: the thesis still
reports a single training seed as a limitation.

## Old results are not replaced
`E2E-sup-p25` = 52/299 (built estimate: 298/299 units reuse `I-only` because C was right) stays in every table. The new
row is called `E2E-sup-p25-CL` and is reported as "independent supplementary measurement". Old episode files are never
overwritten; new episodes go to `/media/dhqg/d3/vlia_outputs/bosung_2026-11-20/` only (enforced in the harness).

## Conditions
- **C rerun.** Stage 15 code unchanged (`run_stage15_supervised.py`, 2e865a6), new work folder: SigLIP features
  extracted again from the raw LIBERO demo videos (agent camera), nested CV 5 outer x 4 inner with the same seeds
  (out-of-fold: each demo is predicted by a model that never saw it), primary model, strict top-1 at p = 25.
  Check: predictions equal the old ones on all 299 units (expected 299/299, 298 correct). If not, the closed-loop row
  is NOT run automatically (moved after all other blocks, waits for Trang).
- **`E2E-sup-p25-CL`.** VLIA `checkpoint_025000` seed 0 (the checkpoint of all 25k rows), empty instruction, token =
  description of the task C predicted (row of `task_embeddings_v2.npz`, as `E2E-sup-p25`), the 299 Stage 12 units,
  `init_state_id = e`, seed `1000 + e`, 520 steps, **all 299 units rolled out, no reuse**.
- **Seeds.** Baseline 25k `correct` and VLIA 25k `I-only`, 300 units each, seed `2000 + e` and `3000 + e`,
  `init_state_id = e`. Seed 1000 = the existing 25k files.

## Analysis (descriptive, no pass/fail threshold)
- Wilson 95% for each row; paired difference over units (bootstrap B = 10,000, seed 0) and exact McNemar for
  `E2E-sup-p25-CL` − `E2E` (F, 35/299), − `I-only` (same 299 units), − `I-wrong`; number of units whose result differs
  from the built estimate (2x2 table), per task.
- Seeds: successes per seed, mean ± sample SD over 3 seeds, VLIA − baseline per seed (paired bootstrap), per task,
  units that change result between seeds.

## Run order (fixed now, never by results)
G0 probe (VLIA 25k `I-only`, e = 0, 10 tasks, seeds 1000 + e and 2000 + e: same-seed reproducibility, whether the seed
changes anything, seconds per episode) → block 1 `E2E-sup-p25-CL` (299) → block 2 seed 2000 (baseline `correct`, then
VLIA `I-only`, 600) → block 3 seed 3000 (600) → block 4 the planned rows (`run_final_eval_100k.sh`: baseline 25k L-*,
VLIA 25k E2E-gen-p25, VLIA final milestone 50k all rows) → block 5 (seed 1 VLIA to 25k + I-only) is written but
never started without Trang.
- If seed 2000 gives exactly seed 1000 on all 10 G0 units (same first actions, same steps), blocks 2–3 are dropped.
- Cut rule: if after G0 the projected end of blocks 1–4 is later than 20:00 on 22/11, block 3 moves after block 4;
  if block 4 still ends after that, block 2 moves to the end too. Stops only at the end of a whole row.
