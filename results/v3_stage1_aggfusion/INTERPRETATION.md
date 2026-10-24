# v3 Stage 1: aggregation × context fusion: interpretation (facts only)

Grid: 5 aggregators (A0, A3, A4 with h = 1, 2, 4) × λ ∈ {0, 0.25, 0.5, 1, 2, ∞} = 30 configs on so400m / T1. Nested selection per outer fold on training-fold queries (mean K=5 accuracy over k); 2503 train/val queries, 27 videos, scene-grouped 8 folds. F reproduced exactly (853/874/907/969). The test split was not read.

Selected per fold: A4h2|λ=2 in 5 folds, A4h1|λ=2 in 1, A4h1|λ=1 in 1, A4h2|λ=∞ in 1 (`selected_by_fold.csv`).

## V1: FAIL ("improves on F" not met)

- 5-way delta vs F: CI lower bound > 0 at 0/4 prefixes. The deltas are +0.0004, +0.0068, +0.0052 and −0.0044 at k = 2, 4, 6, 8, and every CI contains 0.
- Gallery-MRR delta vs F: +0.0078 to +0.0182, with CI lower bound > 0 at 4/4 prefixes. The MRR part of the rule (CI upper bound > 0 at all 4) holds.
- 10-way delta: CI lower bound > 0 only at k=8 (+0.0152 [+0.0016, +0.0299]).

The descriptive full grid is in `grid_descriptive.csv` (not selection-corrected; no verdict).

## Limitations
27 videos; same-video distractors may be paraphrases; selection uses training folds only; no untouched test (the locked test is spent).
