# LIBERO-10 goal-information ablation: interpretation (facts only)

Policy: `/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000/pretrained_model`. The saved config was asserted to have action dim 7 and `empty_cameras = 1`. This is the checkpoint of the 24%/100-episode reference eval (`~/lerobot/outputs/eval/2026-10-06/14-29-45_libero_smolvla`). The run used the same lerobot env (`~/lerobot/.venv`) and rename map.

Design: 3 conditions × 10 tasks × 10 episodes = 300 rollouts. This is the top ladder level: the 3-episode probe measured 10.7 s/episode, ETA 0.89 h ≤ 8 h. Episode e of every task uses LIBERO init state e and seed 1000 + e in all three conditions. Only the instruction string differs: original, empty, or the instruction of task (t+5) mod 10.

## Results

| condition | successes / n | rate | Wilson 95% |
|---|---:|---:|---|
| correct | 19 / 100 | 0.190 | [0.125, 0.278] |
| empty | 0 / 100 | 0.000 | [0.000, 0.037] |
| wrong | 0 / 100 | 0.000 | [0.000, 0.037] |

Per task (correct condition): 0, 1, 3, 7, 1, 3, 3, 0, 0, 1 successes out of 10 for tasks 0–9. The empty and wrong conditions had no successes on any task. All 200 empty/wrong rollouts ran to the 520-step limit; 81 of the 100 correct rollouts did.

## Verdicts (paired over the 100 (task, episode) units; bootstrap B=10000, seed 0)

- **L1: PASS.** success(correct) − success(empty) = +0.190, 95% CI [+0.120, +0.270].
- **L2: FAIL.** success(empty) − success(wrong) = 0.000, CI [0.000, 0.000]. Both conditions had 0 successes, so the rule cannot be met.

## Videos (d3)

Every rollout was recorded under `/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/videos/`. Examples:
- correct, success: `correct_t1_e4.mp4`
- correct, failure: `correct_t0_e0.mp4`
- empty, failure: `empty_t0_e0.mp4`
- wrong, failure: `wrong_t0_e0.mp4`

No success video exists for empty or wrong.

## Limitations

- One checkpoint, one training seed, 10 episodes per task.
- The correct-condition rate (19/100) uses init states 0–9 and seeds 1000–1009 per task. It is not the same episode set as the 24% reference eval, and the two are not compared here.
- The empty and wrong prompts are out of the training distribution. This tests dependence on the language channel. It does not show that an intention signal would help.
- This is a motivation experiment, not evidence for VLIA.
