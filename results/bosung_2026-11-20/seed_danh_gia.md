# Blocks 2–3: evaluation seed (rollout seed, not training seed), 25k checkpoints, 300 units

| model | seed 1000 | seed 2000 | seed 3000 | mean ± SD (successes) |
|---|---:|---:|---:|---:|
| baseline `correct` | 55/300 (14.4–23.1) | 60/300 (15.9–24.9) | 61/300 (16.2–25.2) | 58.7 ± 3.2 |
| VLIA `I-only` | 52/300 (13.5–22.0) | 54/300 (14.1–22.7) | 66/300 (17.7–27.0) | 57.3 ± 7.6 |

VLIA − baseline per seed (paired over units):

| seed | VLIA − baseline (pp) | bootstrap 95% | VLIA won / baseline won | McNemar p |
|---:|---:|---|---:|---:|
| 1000 | -1.0 | -5.7 … +3.7 | 23 / 26 | 0.775 |
| 2000 | -2.0 | -7.3 … +3.3 | 30 / 36 | 0.539 |
| 3000 | +1.7 | -3.7 … +7.0 | 37 / 32 | 0.63 |

Units whose result changes between seeds (same model, same unit):

- baseline: 300 units; success in all 3 seeds 17, in none 194, **changes 89**
- vlia: 300 units; success in all 3 seeds 16, in none 196, **changes 88**

Per task (successes, seeds 1000 / 2000 / 3000):

| task | baseline | VLIA |
|---:|---|---|
| 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| 1 | 1 / 2 / 0 | 0 / 2 / 1 |
| 2 | 13 / 10 / 10 | 7 / 9 / 9 |
| 3 | 21 / 21 / 26 | 18 / 18 / 21 |
| 4 | 1 / 0 / 1 | 0 / 0 / 0 |
| 5 | 9 / 15 / 8 | 18 / 11 / 17 |
| 6 | 5 / 4 / 5 | 1 / 2 / 2 |
| 7 | 0 / 0 / 0 | 0 / 0 / 0 |
| 8 | 1 / 2 / 4 | 0 / 2 / 3 |
| 9 | 4 / 5 / 7 | 8 / 10 / 13 |
