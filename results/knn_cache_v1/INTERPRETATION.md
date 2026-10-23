# Cache-kNN text-space correction (Stage 5): interpretation (facts only)

Run after Stage 3, on the baseline `siglip-base-patch16-224 | A0 | C0 | T0` float32 embeddings. M=20 neighbours, softmax(cos/0.05) weights, beta ∈ {0, 0.25, 0.5, 1, 2} chosen per outer fold by leave-one-scene-out inside the training scenes. Neighbours never come from the held-out fold. 2503 train/val queries, 27 videos; the locked test split was not read.

**This method uses training-scene WHY labels: it is a non-parametric trained method, not zero-parameter.**

## Selected beta

Beta was 0.25 in 5 folds (1, 3, 4, 5, 7) and 0.0 in 3 folds (0, 2, 6). Folds with beta = 0 reduce to the baseline.

## Verdict K1

**K1: FAIL.** No prefix has a CI lower bound > 0 for either MRR or 5-way. Every point estimate is negative:

| k | MRR delta [95% CI] | 5-way delta [95% CI] |
|---:|---|---|
| 2 | −0.0054 [−0.0103, −0.0012] | −0.0044 [−0.0171, +0.0078] |
| 4 | −0.0063 [−0.0112, −0.0021] | −0.0056 [−0.0146, +0.0041] |
| 6 | −0.0044 [−0.0096, +0.0004] | −0.0064 [−0.0154, +0.0025] |
| 8 | −0.0071 [−0.0135, −0.0015] | −0.0056 [−0.0138, +0.0019] |

The MRR CI lies entirely below 0 at k=2, 4 and 8. The 10-way delta is negative at all k (CI below 0 at k=8).

## Limitations

- 27 source videos; same-video distractors may be paraphrases; no untouched test.
- Only the baseline embedding was used, as preregistered; kNN was not combined with the Stage 3 selected pipeline.
