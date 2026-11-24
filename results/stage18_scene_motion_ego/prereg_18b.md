# Stage 18b pre-registration (side check, descriptive): why is EgoIntent at chance?

Written 2026-11-24 before computing any number of 18b. prereg.md (Stage 18) is unchanged.
Data/features/classifier exactly as Stage 18: 1,001 steps, 10 verbs, S / M / S+M at p50 and p100 (S has no p),
torch LogReg with class-balanced weights, C grid {0.01..100}, one x10 edge extension, standardisation on train only.

- (a) within-video: random step-level 5-fold, stratified by label (Stage 15 strat_folds, seed 0); steps of one video
  may be in train and test. Inner CV: strat_folds(train labels, 4, seed 1 + outer fold), train only.
- (b) train accuracy: fit on all 1,001 steps with C = 100 (largest grid value), score the same steps. Explanation only.
- (c) sanity: (a) repeated with labels permuted (numpy default_rng(0).permutation), all five features.
- Metric: balanced accuracy (chance 0.10); also accuracy, macro-F1, majority-class accuracy (0.255).
  CI: bootstrap over whole videos, 10,000 draws, seed 0, percentile 95%; paired differences on the same draws.

Reading rules (fixed now):
- (c) gate: if any feature under (c) has CI lower bound > 0.10, stop and report a likely code bug.
- If (a) S+M p50 CI lower bound >= 0.20: features DO carry verb information; Stage 18 chance level is due to new
  scenes. Then: if the paired CI of (a) (S+M p50) - S contains 0, "S ~ S+M": what is learned within video is mainly
  scene/objects, not motion; otherwise motion adds information.
- If (a) S+M p50 CI lower bound < 0.20: frozen SigLIP + linear classifier is not enough to recognise EgoIntent verbs
  even in seen scenes; change features (temporal video features), not the split.
