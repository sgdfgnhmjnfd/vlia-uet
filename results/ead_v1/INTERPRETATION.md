# Evidence-accumulation early decision (Stage 4): interpretation (facts only)

Pipeline: baseline only, `siglip-base-patch16-224 | A0 | C0 | T0` on the float32 re-extraction (amendment 1); no config selection. 2503 train/val queries, 27 videos, scene-grouped 8 folds. The locked test split was not read.

## Verdict S2

**S2: FAIL.** The early-decision minus interpolated fixed-k difference has a CI lower bound > 0 at only 1 of 4 budgets (budget 5); the rule requires ≥ 2.

| budget | realized mean frames | early-decision acc [95% CI] | fixed-k acc interpolated | delta [95% CI] |
|---:|---:|---|---:|---|
| 3 | 3.007 | 0.2793 [0.2485, 0.3131] | 0.2816 | −0.0024 [−0.0093, +0.0046] |
| 4 | 3.999 | 0.2817 [0.2484, 0.3174] | 0.2753 | +0.0064 [−0.0004, +0.0133] |
| 5 | 5.006 | 0.2885 [0.2534, 0.3264] | 0.2745 | +0.0139 [+0.0020, +0.0263] |
| 6 | 6.003 | 0.2865 [0.2524, 0.3230] | 0.2809 | +0.0056 [−0.0026, +0.0139] |

- tau was chosen per outer fold on training-fold queries only. The realized pooled mean frames are within 0.01 of each budget. Per-fold held-out mean frames vary, for example 2.62 to 3.25 at budget 3 (`thresholds.csv`).
- The stopping distribution is bimodal: most queries stop at k=2 or run to k=8 (`summary.csv`, `stop_k_distribution`).
- The fixed-k accuracy curve is nearly flat: 0.2757 (k=1) to 0.2885 (k=8). Its CIs overlap at every k (`fixed_k.csv`). The mean top-1 minus top-2 margin is ≈ 0.0137 at every k.
- The comparator interpolates per-query fixed-k hits at the pooled realized mean frames. The bootstrap holds that interpolation point fixed.

## Limitations

- 27 source videos; same-video distractors may be paraphrases; no untouched test.
- Margin is a decision-score baseline, not a validated measure of semantic evidence sufficiency.
- Tie rule for tau (not fixed by the preregistration): the closest training mean frames to the budget, with ties broken toward the larger tau.
