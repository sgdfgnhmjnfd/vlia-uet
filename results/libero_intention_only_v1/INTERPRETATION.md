# LIBERO-10 intention-only: interpretation (facts only)

Policy: VLIA SmolVLA trained with condition `intention_only` (seed 0, 25 000 steps, batch 32, same recipe as the matched LIBERO7D baseline: `lerobot/smolvla_base`, AdamW lr 1e-4, cosine decay, 335 episodes / 88 302 frames of `lerobot/libero`). During training the language instruction was replaced by `""`. The policy received the task as an intention vector instead: the SigLIP so400m text embedding (template T1, dim 1152) of the task instruction, projected by `IntentionAdapter` (Linear 1152 → 960) and inserted before the state token. Checkpoint: `/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000`.

Training was interrupted twice: a power cut, and a stop at the user's request. It resumed from `checkpoint_008000` and then from `checkpoint_012000`, with optimizer and scheduler state restored. The probe measured 1.845 s/step.

Eval design: 2 conditions × 10 tasks × 10 episodes = 200 rollouts. Every rollout used an empty instruction string. Episode e of task t used LIBERO init state e and seed 1000 + e, as in `results/libero_goal_ablation_v1/`.
- **I-only:** intention vector of task t.
- **I-wrong:** intention vector of task (t+5) mod 10.

The `empty` and `correct` reference rows are the rollouts of `libero_goal_ablation_v1`. They come from a different policy: the language-conditioned matched baseline checkpoint.

## Results

| condition | policy | successes / n | rate | Wilson 95% |
|---|---|---:|---:|---|
| I-only | intention-only (this run) | 20 / 100 | 0.200 | [0.133, 0.289] |
| I-wrong | intention-only (this run) | 0 / 100 | 0.000 | [0.000, 0.037] |
| empty | language baseline (v1) | 0 / 100 | 0.000 | [0.000, 0.037] |
| correct | language baseline (v1) | 19 / 100 | 0.190 | [0.125, 0.278] |

Per-task I-only successes for tasks 0–9: 0, 0, 3, 7, 0, 8, 1, 0, 0, 1 out of 10. I-wrong had no successes on any task. 80 of the 100 I-only rollouts and all 100 I-wrong rollouts ran to the 520-step limit.

## Verdicts (paired over the 100 (task, episode) units; bootstrap B=10000, seed 0)

- **B1: PASS.** success(I-only) − success(empty, v1) = +0.200, 95% CI [+0.120, +0.280].
- **B2: PASS.** success(I-only) − success(I-wrong) = +0.200, 95% CI [+0.120, +0.280].
- Descriptive, no verdict: success(I-only) − success(correct, v1) = +0.010, CI [−0.070, +0.090].

## Videos (d3)

All rollouts are under `/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/videos/`. Examples:
- I-only, success: `I-only_t2_e6.mp4`
- I-only, failure: `I-only_t0_e0.mp4`
- I-wrong, failure: `I-wrong_t0_e0.mp4`

No I-wrong success video exists.

## Limitations

- One training seed, one checkpoint, 10 episodes per task.
- B1 compares two different policies: an intention-trained policy versus a language-trained policy given an empty prompt.
- The intention vectors are text embeddings of the oracle task instructions. They are not intentions predicted from video.
- I-wrong success is 0/100, the same as the language `wrong` condition in v1.
