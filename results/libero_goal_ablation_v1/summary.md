# LIBERO-10 goal-information ablation (prereg v2, Stage 4)

100 paired (task, episode) units per condition; same seed and init state across conditions.

| condition | successes / n | success rate | Wilson 95% |
|---|---:|---:|---|
| correct | 19 / 100 | 0.190 | [0.125, 0.278] |
| empty | 0 / 100 | 0.000 | [0.000, 0.037] |
| wrong | 0 / 100 | 0.000 | [0.000, 0.037] |

| task | instruction | correct | empty | wrong (uses task) |
|---:|---|---:|---:|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0/10 | 0/10 | 0/10 (task 5) |
| 1 | put both the cream cheese box and the butter in the basket | 1/10 | 0/10 | 0/10 (task 6) |
| 2 | turn on the stove and put the moka pot on it | 3/10 | 0/10 | 0/10 (task 7) |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 7/10 | 0/10 | 0/10 (task 8) |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 1/10 | 0/10 | 0/10 (task 9) |
| 5 | pick up the book and place it in the back compartment of the caddy | 3/10 | 0/10 | 0/10 (task 0) |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 3/10 | 0/10 | 0/10 (task 1) |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0/10 | 0/10 | 0/10 (task 2) |
| 8 | put both moka pots on the stove | 0/10 | 0/10 | 0/10 (task 3) |
| 9 | put the yellow and white mug in the microwave and close it | 1/10 | 0/10 | 0/10 (task 4) |

## Verdicts (paired bootstrap over (task, episode), B=10000, seed 0)

- L1 correct − empty: +0.190 [+0.120, +0.270] → PASS
- L2 empty − wrong: +0.000 [+0.000, +0.000] → FAIL

## Example videos (on d3)

```
{
  "correct": {
    "success": "/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/videos/correct_t1_e4.mp4",
    "failure": "/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/videos/correct_t0_e0.mp4"
  },
  "empty": {
    "success": null,
    "failure": "/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/videos/empty_t0_e0.mp4"
  },
  "wrong": {
    "success": null,
    "failure": "/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/videos/wrong_t0_e0.mp4"
  }
}
```
