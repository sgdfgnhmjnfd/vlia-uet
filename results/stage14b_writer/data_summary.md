# Stage 14b W: data summary

## EgoIntent (development steps only; locked test never opened)

- 27 videos, 13 scenes in 8 scene groups (scene-held-out), 2503 steps (units).
- Distinct sentences: WHY 2032, WHAT 2144, NEXT 2009.
- 8 frames per step, evenly from step start to the pre-outcome cutoff; step length mean 2.53 s (min 0.000, max 10.02).

| k | s seen | s left to step end | s to next step start | n with next step |
|---:|---:|---:|---:|---:|
| 2 | 0.36 | 2.17 | 2.83 | 2476 |
| 4 | 1.09 | 1.45 | 2.11 | 2476 |
| 6 | 1.81 | 0.72 | 1.38 | 2476 |
| 8 | 2.53 | 0.00 | 0.66 | 2476 |

## LIBERO-10 (agent view)

- 379 demonstrations, 10 tasks, 5 folds over demonstrations (Stage 15); the 299 robot units are a subset.
- 8 frames per unit from the 4 fps grid (dataset 10 fps), spread over the first p% of the demo.

| p | s seen | s left |
|---:|---:|---:|
| 0 | 0.0 | 26.7 |
| 10 | 2.7 | 24.0 |
| 25 | 6.7 | 20.0 |
| 50 | 13.3 | 13.3 |
| 75 | 20.0 | 6.7 |
| 100 | 26.7 | 0.0 |

| task | demos | fps | mean s | min s | max s |
|---:|---:|---:|---:|---:|---:|
| 0 | 33 | 10 | 28.9 | 24.4 | 36.6 |
| 1 | 49 | 10 | 25.8 | 23.3 | 32.1 |
| 2 | 41 | 10 | 26.4 | 21.7 | 32.2 |
| 3 | 35 | 10 | 24.4 | 21.6 | 30.5 |
| 4 | 38 | 10 | 25.7 | 21.3 | 32.9 |
| 5 | 41 | 10 | 18.5 | 14.9 | 25.8 |
| 6 | 36 | 10 | 25.0 | 20.2 | 34.1 |
| 7 | 43 | 10 | 26.6 | 21.8 | 32.2 |
| 8 | 29 | 10 | 40.6 | 34.0 | 50.4 |
| 9 | 34 | 10 | 29.3 | 22.3 | 44.8 |

Seconds per task at each p: seen = p% of the task's mean duration; left = the rest (exact per-demo values in `libero_writer.csv` are averaged over all tasks).
