# Stage 12 within-scene re-scoring (post-hoc, descriptive)

Saved Stage 12 scores, C2, T1, demos 0..29. No new model run. No pass/fail rule.

| grouping / pair | n | p=0 | p=10 | p=25 | p=50 | p=100 | p25−p0 | p50−p0 | chance |
|---|---:|---|---|---|---|---|---|---|---:|
| within-scene | 60 | 53.3 [41.7; 66.7] | 50.0 [38.3; 63.3] | 50.0 [38.3; 63.3] | 50.0 [38.3; 63.3] | 43.3 [31.7; 56.7] | -3.3 [-18.3; 11.7] | -3.3 [-18.3; 11.7] | 50.0 |
| within-family | 269 | 72.5 [66.9; 77.7] | 76.6 [71.4; 81.4] | 77.7 [72.5; 82.5] | 77.3 [72.1; 82.2] | 75.5 [70.3; 80.7] | 5.2 [1.5; 9.3] | 4.8 [1.1; 8.9] | 22.2 |
| pair 0 vs 1 | 60 | 53.3 [41.7; 66.7] | 50.0 [38.3; 63.3] | 50.0 [38.3; 63.3] | 50.0 [38.3; 63.3] | 43.3 [31.7; 56.7] | -3.3 [-18.3; 11.7] | -3.3 [-18.3; 11.7] | 50.0 |
| pair 7 vs 5 | 60 | 98.3 [95.0; 100.0] | 95.0 [88.3; 100.0] | 50.0 [36.7; 61.7] | 78.3 [68.3; 88.3] | 93.3 [86.7; 98.3] | -48.3 [-60.0; -36.7] | -20.0 [-31.7; -10.0] | 50.0 |
| pair 5 vs 9 | 60 | 53.3 [40.0; 65.0] | 50.0 [38.3; 63.3] | 55.0 [41.7; 66.7] | 50.0 [38.3; 63.3] | 50.0 [38.3; 63.3] | 1.7 [0.0; 5.0] | -3.3 [-8.3; 0.0] | 50.0 |

| p | mean 10-way margin (all) | 10-way margin (within-scene demos) | margin among scene tasks |
|---:|---:|---:|---:|
| 0 | 0.0287 | 0.0089 | 0.0230 |
| 10 | 0.0312 | 0.0107 | 0.0424 |
| 25 | 0.0353 | 0.0171 | 0.0455 |
| 50 | 0.0311 | 0.0105 | 0.0367 |
| 100 | 0.0300 | 0.0085 | 0.0296 |

## What it means (descriptive, post hoc)

- Only one LIBERO-10 scene holds more than one task: `LIVING_ROOM_SCENE2` (tasks 0 and 1). Within it, F stays at chance: 53.3% at p = 0, 50.0% at p = 25 and p = 50, against a chance of 50%. The difference p25 − p0 is −3.3 [−18.3; +11.7]. On these saved scores, watching more of the demo does not help F tell the two tasks of that scene apart.
- Within a scene family (`LIVING_ROOM`: 0, 1, 4, 6, 7; `KITCHEN`: 2, 3, 8, 9), accuracy rises from 72.5% at p = 0 to 77.7% at p = 25 (+5.2 [+1.5; +9.3]). A family contains several distinct scenes, so this gain mixes scene and motion cues.
- For the pair 7 vs 5 (different scenes), 2-way accuracy falls from 98.3% at p = 0 to 50.0% at p = 25 (−48.3 [−60.0; −36.7]). Here, more of the demo moves F away from the right task.
