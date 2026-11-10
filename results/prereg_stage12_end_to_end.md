# Pre-registration: Stage 12, from video to robot action (2026-10-30)

Written and committed before any Stage 12 scoring or rollout. Only dataset metadata (task list, demos per task) was read first; that is how the amendment at the end was found. There is no training: F stays frozen and the VLIA checkpoint is the Stage 4 one. The design below is copied verbatim from the task. The amendments at the end take precedence where they differ.

> **video of the task being started → F picks the goal among the 10 LIBERO-10 tasks → the token of the picked task → VLIA acts.**

## Query videos

- **Source.** The LIBERO-10 episodes of `lerobot/libero` (`task_index` < 10).
- **Task mapping.** Map each dataset `task_index` to the eval task id 0–9 by exact match of the normalised task text (lower case, single spaces, no final period) against `results/libero_goal_ablation_v1/instructions.json`. Stop if any task does not map one-to-one.
- **Demo order.** Within each task, sort demos by `episode_index`. Demo d is the d-th demo of that task.
  - Demos d = 0..29 are the **rollout queries**.
  - Demos d = 30..49 (or up to the last one) are the **calibration set**. They are used only to set the "ask" threshold below.
- **Camera.** The agent-view camera, `observation.images.image`.
- **Orientation check.** Before any scoring, save one contact sheet: the first frame of demo 0 of every task, at `results/stage12_end_to_end/contact_sheet.png`. Look at it.
  - If the frames are upside-down, rotate every frame by 180°. This is a preprocessing fix decided from the picture only, not from any score.
  - Record the decision in `config.json`.
- **Prefix p** = the first p of the demo's frames, p ∈ {0, 10, 25, 50, 100}%.
  - Take 8 frames evenly spaced over the prefix, always including its last frame.
  - p = 0 means the first frame only: it shows the scene before anything moves.
  - **The primary prefix is p = 25%.**

## F (frozen, no change)

- **Model:** SigLIP so400m-patch14-384, the ladder image preprocessing, L2-normalised image embeddings, mean over the 8 frames.
- **Candidates:** the 10 original LIBERO-10 task descriptions in template T1, the text side exactly as in `scripts/v3_stage4_task_embeddings.py`. For the original strings, the candidate text embeddings must equal the token table rows (gate E1 below).
- **Scene-context removal** (primary). A query has no earlier steps, so F falls back to subtracting a fixed mean, as it does for the first step of an EgoIntent video. Here the fixed mean is the mean frame embedding of up to 2,000 randomly chosen (seed 0) **non-LIBERO-10** episodes of `lerobot/libero` (`task_index` ≥ 10), 8 frames each, spread over the whole episode. This uses no LIBERO-10 frame and no label.
- **Descriptive only** (no rule, never used to pick the token): no context removal (C0), and template T0 = the bare task text.
- **Outputs:** prediction = top-1 task; confidence margin = top-1 score minus top-2 score.

## Robot conditions

All use the VLIA checkpoint and an empty instruction, on the 300 units (task t, e = 0..29). For unit (t, e), the query is demo d = e of task t at p = 25%.

| condition | intention token |
|---|---|
| `E2E` (**new**) | T1 embedding of the **original description of the task F predicts** for demo e of task t |
| `E2E-ask` (computed, no new runs) | if F's margin < τ, the robot "asks" and the person names the task, so the token is the correct one (the `I-only` episode); otherwise it is the `E2E` episode |
| `I-only` (existing) | T1 embedding of the correct task's original description |
| `I-wrong` (existing) | T1 embedding of the original description of task (t+5) mod 10 |

- τ is the 25% quantile of F's margins on the calibration demos (d ≥ 30, all tasks, p = 25%). It is fixed before any rollout.
- `z_int_task` in each `E2E` episode JSON must hold the predicted description string.

## Pre-registered rules

Recognition uses 10-way top-1 accuracy, where a tie counts as wrong. Its 95% CI comes from a bootstrap over demos, clustered by task (B = 10,000, seed 0). Robot deltas use `paired_bootstrap` over the 300 (task, episode) pairs.

- **P1 (F reads the task from the start of the video).** Accuracy at p = 25% on demos 0..29 has CI lower bound > 10% (chance).
- **P2 (it uses the motion, not only the scene).** Accuracy at p = 25% minus accuracy at p = 0, paired over the same demos, has CI lower bound > 0.
- **E1 (the loop carries the task).** success(`E2E`) − success(`I-wrong`) has CI lower bound > 0.
- **Descriptive**, with no pass/fail:
  - `E2E` − `I-only`, the cost of recognition errors;
  - `E2E-ask` − `E2E`, and the ask rate;
  - the accuracy for every p, plus C0 and T0;
  - the 10 × 10 confusion matrix at p = 25%;
  - the success of `E2E` split by whether F was right.

## Gates (all before any rollout)

- **E1:** for the 10 original strings, the candidate text embeddings equal the token table rows (max abs diff < 1e-6).
- **M1:** the task mapping is one-to-one and every task has at least 30 demos.
- **R1:** re-load 3 existing `I-only` episode JSONs and check that their `z_int_task` equals the original description of their own task.
- **D1 (descriptive).** For the units where F is right, the `E2E` token equals the `I-only` token. Report how often the two episodes have the same success. If it is not 100%, say so; this measures run-to-run noise.

## Amendments (made before any scoring or rollout)

**A1. Task 8 has 29 demos, so M1 as written fails.** Found from the dataset metadata. Eval task 8 ("put both moka pots on the stove") has 29 LIBERO-10 demos; the other tasks have 33–49. Trang decided (2026-10-30, before any scoring):
- the unit (t = 8, e = 29) is dropped from `E2E` and from every comparison, which leaves **299 units**. Every "300" above reads 299, and P1 and P2 use the 299 demos;
- M1 becomes: the mapping is one-to-one and every task has at least 29 demos;
- τ uses the 80 calibration demos that exist. Task 8 has none.

**A2. Text side of the fallback (disclosed, not changed).** F's real fallback for the first step of an EgoIntent video (`run_ladder_v1.processed`) subtracts a mean from the frames and also from the text embeddings. This design subtracts a mean from the frames only and keeps the raw text embeddings, as gate E1 requires. Stage 12 follows the design above. The scoring is: unit(frame − mean) per frame, unit(mean of the 8), cosine with the raw text embedding, i.e. `run_ladder_v1.aggregate` with A0.

**A3. Implementation details fixed here.**
- Prefix length n = max(1, ceil(L · p / 100)) frames of a demo of length L. Frame indices = round(linspace(0, n − 1, 8)).
- Context episodes: 8 frames at round(linspace(0, L − 1, 8)). There are 1,314 non-LIBERO-10 episodes, fewer than 2,000, so all of them are used.
- The clustered bootstrap resamples the 10 tasks with replacement; P2 uses the same bootstrap on the paired per-demo differences.
