# Selective prediction on the frozen pipeline: interpretation (facts only)

Scores are the out-of-fold K=5 cosine scores of the nested-selected ladder pipeline, `siglip-so400m-patch14-384 | A0 | C2 | T1`, selected in every fold. They were recomputed from the ladder cache with each fold's training-fold C1 means. They reproduce `results/ladder_v1/summary.csv` exactly: 5-way hits 853/874/907/969 and 10-way accuracies at k = 2, 4, 6, 8 (`reproduction_check.csv`). 2503 train/val queries, 27 videos. The test split was not read.

## Verdict S1b (primary score `margin`, K=5)

**S1b: PASS.** The CI lower bound of acc@50% − acc@100% is > 0 at 4/4 prefixes (video bootstrap B=2000, seed 0, re-sorted per resample):

| k | acc@100% | acc@50% | acc@20% | acc@50 − acc@100 [95% CI] | AURC | random-rejection AURC | AURC excess [95% CI] |
|---:|---:|---:|---:|---|---:|---:|---|
| 2 | 0.3408 | 0.4097 | 0.5329 | +0.0690 [+0.0387, +0.0964] | 0.5488 | 0.6592 | +0.1104 [+0.0773, +0.1365] |
| 4 | 0.3492 | 0.4177 | 0.5569 | +0.0686 [+0.0452, +0.0952] | 0.5341 | 0.6508 | +0.1167 [+0.0833, +0.1417] |
| 6 | 0.3624 | 0.4409 | 0.5768 | +0.0785 [+0.0534, +0.1062] | 0.5174 | 0.6376 | +0.1202 [+0.0867, +0.1447] |
| 8 | 0.3871 | 0.4569 | 0.6088 | +0.0697 [+0.0403, +0.0972] | 0.4974 | 0.6129 | +0.1154 [+0.0837, +0.1387] |

## Leakage-free operational thresholds (per-fold quantile of the margin outside the held-out fold)

- Realized coverages are 0.800 to 0.801, 0.499 to 0.503 and 0.197 to 0.205 for targets 0.8, 0.5 and 0.2.
- At target 0.5, selective 5-way accuracy is 0.407 to 0.461. At target 0.2 it is 0.533 to 0.609. For both targets the CI lower bound exceeds the prefix's overall accuracy at every k.
- At target 0.8 (0.355 to 0.400), the CI contains the overall accuracy at every k (`operational_thresholds.csv`).

## Secondary (no verdict)

- K=10 margin: acc@50 − acc@100 is +0.046 to +0.057, with the CI lower bound > 0 at 4/4 prefixes.
- `confidence` and `entropy_neg` at T = 0.07, and temperatures 0.02 and 0.2, are reported in `summary.csv` and `summary.md`.

## Limitations

- 27 source videos; same-video distractors may be paraphrases.
- The frozen config was chosen by nested selection on these same train/val queries. The scores are out-of-fold per outer fold, but the choice of this config family used all folds' training data.
- No independent calibration split; no untouched test.
- Margin, confidence and entropy are decision-score baselines, not a validated measure of semantic evidence sufficiency.
