# Selective prediction on SigLIP baseline scores: interpretation (facts only)

Inputs: float32 re-extraction of `google/siglip-base-patch16-224` frame and T0 text embeddings (amendment 1), canonical `build_kway` candidates, 2503 train/val queries, 27 videos, 13 scenes. The amended reproduction gate passed with Δ = 0 hits at every k (`results/ladder_v1/gate_amended.json`). The locked test split was not read.

## Verdict S1 (primary score `margin`, K=5)

**S1: PASS.** The 95% video-bootstrap CI (B=2000, seed 0, re-sorted per resample) lower bound of acc@50% − acc@100% is > 0 at 4 of 4 prefixes:

| k | acc@100% | acc@50% | acc@50 − acc@100 [95% CI] | AURC | random-rejection AURC |
|---:|---:|---:|---|---:|---:|
| 2 | 0.2773 | 0.3035 | +0.0262 [+0.0004, +0.0457] | 0.6863 | 0.7227 |
| 4 | 0.2753 | 0.3011 | +0.0258 [+0.0058, +0.0483] | 0.6780 | 0.7247 |
| 6 | 0.2809 | 0.3123 | +0.0314 [+0.0099, +0.0507] | 0.6704 | 0.7191 |
| 8 | 0.2885 | 0.3235 | +0.0350 [+0.0168, +0.0539] | 0.6596 | 0.7115 |

- The k=2 lower bound (+0.0004) is close to 0.
- The effect size is small: accepting the top half by margin raises 5-way accuracy by 2.6 to 3.5 points. At 20% coverage, accuracy is 0.333 to 0.363.
- Leakage-free operational thresholds (per-fold quantile of the margin outside the held-out fold) realize coverages of 0.80/0.50/0.20 within ±0.01 on average. At target 0.5 they give selective accuracy 0.302 to 0.323, and every CI contains that prefix's overall accuracy. At target 0.2 they give 0.329 to 0.363; the CI lower bound exceeds the overall accuracy only at k=6 (0.2888 vs 0.2809) and k=8 (0.2952 vs 0.2885) (`operational_thresholds.csv`).
- Secondary scores (`confidence`, `entropy_neg`, T=0.07) and temperatures 0.02 and 0.2 show gains of similar size (`summary.md`). They do not change the verdict.
- For K=10 (secondary), the margin gain CI lower bound is > 0 at k=2, 4 and 6, and not at k=8.

## Limitations

- 27 source videos; the bootstrap resamples whole videos, so intervals are wide and the effective sample size is small.
- Same-video K-way distractors can be paraphrases of the positive, which caps attainable accuracy and adds label noise to "correct".
- No independent calibration split exists. The only out-of-sample thresholding is the scene-grouped fold quantiles.
- No untouched test set was evaluated.
- Margin, confidence and entropy are decision-score baselines. They are not a validated measure of semantic evidence sufficiency.
