# Stage 12 within-scene re-scoring: post-hoc analysis plan (2026-11-10)

**Status.** This is a **post-hoc** analysis, decided **after** seeing the Stage 12 results. In those results P2 failed: the first frame alone gave 64.9%, and the errors stay inside look-alike scenes. The analysis is **descriptive only and has no pass/fail rule**. It runs no new model and no rollout. It uses only the already saved Stage 12 scores (`/media/dhqg/d3/vlia_outputs/stage12/scores.npz`) under the Stage 12 primary setting: C2 (fixed non-LIBERO-10 mean), T1, demos 0..29 of each task (299 demos; task 8 has 29).

**Question.** Once the scene is fixed, does watching more of the demo help F choose between the tasks that share that scene?

## Groupings

- **Scene** of each task: read from the LIBERO-10 BDDL file names of the benchmark used by `libero_goal_ablation.py` (`<SCENE>_<task text>.bddl` in the `libero` package of `~/lerobot/.venv`). A task is matched to its file by its normalised text. The mapping is written to `config.json`.
- **Scene family**: the scene name without its number (`LIVING_ROOM`, `KITCHEN`, `STUDY`).

## Measures

All measures are computed for every saved prefix p ∈ {0, 10, 25, 50, 100}%.

1. **Within-scene accuracy.**
   - Covers each demo whose scene holds 2 or more of the 10 tasks.
   - Prediction = top-1 among only the tasks of its own scene; a tie counts as wrong.
   - Reported: accuracy, chance level (mean of 1/|scene| over those demos), and a 95% CI from a bootstrap clustered by demo (B = 10,000, seed 0, `cluster_bootstrap`).
   - Paired differences p25 − p0 and p50 − p0, over the same demos, same bootstrap.
2. **Within-family accuracy.** The same with the scene-family grouping. Families with one task are excluded.
3. **Confuser pairs** (0, 1), (7, 5), (5, 9):
   - 2-way accuracy between the two tasks on the demos of those two tasks; correct means the true task scores strictly higher;
   - reported at every p, with the paired differences p25 − p0 and p50 − p0.
4. **Margin.**
   - Mean top-1 − top-2 margin of the 10-way scores per p, over all 299 demos and over the within-scene demos.
   - For the within-scene demos, also the margin among only their scene's tasks.

Helpers `top2`, `strict_correct` and `cluster_bootstrap` are imported from `scripts/run_stage12_e2e.py`, which is not edited.
