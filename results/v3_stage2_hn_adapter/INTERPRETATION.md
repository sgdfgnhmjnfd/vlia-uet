# v3 Stage 2: hard-negative residual adapter: interpretation (facts only)

Input: F's aggregation (A0 | C2 | T1, so400m), because V1 failed. z' = norm(z + Wz), W zero-initialised, text frozen. InfoNCE (τ = 0.05) with all other distinct same-video WHY texts as hard negatives plus other-video in-batch texts. AdamW lr 1e-3. Inner leave-one-scene-out selection chose weight decay 1.0 and 5 epochs, the strongest regularisation and the shortest training in the grid, in 8/8 folds. Held-out metrics are averaged over seeds 0/1/2. Held-out-fold queries were asserted absent from every training batch. 2503 train/val queries; the test split was not read.

## V2: FAIL

- 5-way delta vs F: −0.021 to −0.040, with CI upper bound < 0 at 4/4 prefixes.
- Gallery-MRR delta: −0.016 to −0.018, with CI upper bound < 0 at 4/4 prefixes.
- 10-way delta: −0.033 to −0.041, CI below 0 at 4/4 prefixes.

Control (random other-video negatives only, same selected hyper-parameters; descriptive): 5-way −0.025 to −0.037 and MRR −0.016 to −0.021 vs F, CI below 0 at 4/4 prefixes for both.

## Limitations
27 videos; same-video distractors may be paraphrases; one adapter form and the preregistered grid only; no untouched test.
