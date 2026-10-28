# LIBERO-10 intention-only (prereg v3, Stage 4)

100 paired (task, episode) units; seeds and init states identical to libero_goal_ablation_v1.

| condition | source | successes / n | rate | Wilson 95% |
|---|---|---:|---:|---|
| I-only | this run | 20 / 100 | 0.200 | [0.133, 0.289] |
| I-wrong | this run | 0 / 100 | 0.000 | [0.000, 0.037] |
| empty | libero_goal_ablation_v1 | 0 / 100 | 0.000 | [0.000, 0.037] |
| correct | libero_goal_ablation_v1 | 19 / 100 | 0.190 | [0.125, 0.278] |

| task | I-only | I-wrong | empty (v1) | correct (v1) |
|---:|---:|---:|---:|---:|
| 0 | 0/10 | 0/10 | 0/10 | 0/10 |
| 1 | 0/10 | 0/10 | 0/10 | 1/10 |
| 2 | 3/10 | 0/10 | 0/10 | 3/10 |
| 3 | 7/10 | 0/10 | 0/10 | 7/10 |
| 4 | 0/10 | 0/10 | 0/10 | 1/10 |
| 5 | 8/10 | 0/10 | 0/10 | 3/10 |
| 6 | 1/10 | 0/10 | 0/10 | 3/10 |
| 7 | 0/10 | 0/10 | 0/10 | 0/10 |
| 8 | 0/10 | 0/10 | 0/10 | 0/10 |
| 9 | 1/10 | 0/10 | 0/10 | 1/10 |

## Verdicts (paired bootstrap over (task, episode), B=10000, seed 0)

- B1 I-only − empty: +0.200 [+0.120, +0.280] → PASS
- B2 I-only − I-wrong: +0.200 [+0.120, +0.280] → PASS
- descriptive I-only − correct language: +0.010 [-0.070, +0.090]

```
{
  "I-only": {
    "success": "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/videos/I-only_t2_e6.mp4",
    "failure": "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/videos/I-only_t0_e0.mp4"
  },
  "I-wrong": {
    "success": null,
    "failure": "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/videos/I-wrong_t0_e0.mp4"
  }
}
```
