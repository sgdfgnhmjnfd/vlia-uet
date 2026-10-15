# Locked candidate test: paired analysis

This file is a post-hoc analysis of the one locked evaluation, not a new evaluation or a model-selection step. Candidate test remains video-UID-held-out only; it is not scene-disjoint or guaranteed untouched outside the inspected local artifacts.

Exact multi-positive random-ranking chance: MRR 0.0318, Top-1 0.0071, Top-3 0.0208.

| Frames | Top-1 error seed-examples | Same-task errors | Cross-task errors | Same-task share |
|---:|---:|---:|---:|---:|
| 2 | 1214 | 553 | 661 | 0.456 |
| 4 | 1211 | 563 | 648 | 0.465 |
| 6 | 1210 | 605 | 605 | 0.500 |
| 8 | 1209 | 640 | 569 | 0.529 |

## Paired prefix differences

The paired ΔMRR uses the same candidate test examples and averages across seeds per example. The intervals resample the four video UIDs (50,000 replicates) and are exploratory given only four independent clusters.

| Pair | Paired ΔMRR | 4-video cluster bootstrap 95% interval | Top-1 gains/losses (seed-examples) |
|---|---:|---:|---:|
| 2→4 | -0.0001 | [-0.0059, +0.0082] | 8 / 5 |
| 4→6 | +0.0006 | [-0.0004, +0.0018] | 5 / 4 |
| 6→8 | +0.0020 | [+0.0000, +0.0039] | 2 / 1 |
| 2→8 | +0.0024 | [-0.0034, +0.0119] | 9 / 4 |

These results are descriptive only. There are few source-video clusters and low Top-1 counts; no claim of statistical superiority, calibrated uncertainty, semantic sufficiency, or broad generalization is supported.
