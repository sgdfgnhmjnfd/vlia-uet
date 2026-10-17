# Scene-grouped CV: early goal ranking (out-of-fold)

8 scene-grouped folds over 27 videos (train+val only; locked test never read). Seeds [0, 1, 2]; epoch chosen on an inner split of training scenes. Lift = metric / exact random-ranking chance (1.0 = chance). 95% intervals come from a bootstrap over source videos (resampling whole videos); they are descriptive and do not cover scene/domain shift beyond these videos.

## target = sentence

| k | MRR | MRR lift [95% CI] | Top-1 lift | Top-3 lift | 5-way acc (chance 0.20) | 5-way lift [CI] | 10-way acc (chance 0.10) |
|---:|---|---|---|---|---|---|---|
| 2 | 0.0335 ± 0.0012 | 1.13 [0.86, 1.43] | 1.15 | 1.18 | 0.2163 ± 0.0055 | 1.08 [1.00, 1.16] | 0.1131 ± 0.0046 |
| 4 | 0.0345 ± 0.0013 | 1.16 [0.88, 1.48] | 1.59 | 1.20 | 0.2149 ± 0.0046 | 1.07 [1.00, 1.16] | 0.1109 ± 0.0054 |
| 6 | 0.0353 ± 0.0016 | 1.18 [0.90, 1.52] | 1.61 | 1.24 | 0.2124 ± 0.0027 | 1.06 [0.98, 1.16] | 0.1109 ± 0.0048 |
| 8 | 0.0347 ± 0.0010 | 1.17 [0.87, 1.51] | 1.47 | 1.20 | 0.2193 ± 0.0103 | 1.10 [1.01, 1.20] | 0.1156 ± 0.0054 |

## target = smolvlm_pca

| k | MRR | MRR lift [95% CI] | Top-1 lift | Top-3 lift | 5-way acc (chance 0.20) | 5-way lift [CI] | 10-way acc (chance 0.10) |
|---:|---|---|---|---|---|---|---|
| 2 | 0.0340 ± 0.0013 | 1.14 [0.95, 1.36] | 1.26 | 1.24 | 0.2188 ± 0.0038 | 1.09 [1.02, 1.17] | 0.1176 ± 0.0062 |
| 4 | 0.0338 ± 0.0016 | 1.14 [0.94, 1.35] | 1.31 | 1.17 | 0.2217 ± 0.0065 | 1.11 [1.03, 1.19] | 0.1123 ± 0.0052 |
| 6 | 0.0346 ± 0.0019 | 1.16 [0.95, 1.39] | 1.36 | 1.34 | 0.2233 ± 0.0050 | 1.12 [1.04, 1.20] | 0.1199 ± 0.0036 |
| 8 | 0.0354 ± 0.0025 | 1.19 [0.98, 1.42] | 1.40 | 1.22 | 0.2297 ± 0.0073 | 1.15 [1.07, 1.23] | 0.1156 ± 0.0074 |

## Paired differences (to minus from; same examples; bootstrap over videos)

| comparison | metric | from -> to | delta [95% CI] |
|---|---|---|---|
| prefix | MRR | sentence@k2 -> sentence@k4 | +0.0010 [-0.0015, +0.0033] |
| prefix | kway5 | sentence@k2 -> sentence@k4 | -0.0013 [-0.0089, +0.0063] |
| prefix | kway10 | sentence@k2 -> sentence@k4 | -0.0021 [-0.0081, +0.0036] |
| prefix | MRR | sentence@k4 -> sentence@k6 | +0.0008 [-0.0007, +0.0024] |
| prefix | kway5 | sentence@k4 -> sentence@k6 | -0.0025 [-0.0107, +0.0054] |
| prefix | kway10 | sentence@k4 -> sentence@k6 | -0.0000 [-0.0054, +0.0057] |
| prefix | MRR | sentence@k6 -> sentence@k8 | -0.0006 [-0.0021, +0.0010] |
| prefix | kway5 | sentence@k6 -> sentence@k8 | +0.0069 [-0.0006, +0.0140] |
| prefix | kway10 | sentence@k6 -> sentence@k8 | +0.0047 [-0.0030, +0.0121] |
| prefix | MRR | sentence@k2 -> sentence@k8 | +0.0012 [-0.0014, +0.0039] |
| prefix | kway5 | sentence@k2 -> sentence@k8 | +0.0031 [-0.0069, +0.0136] |
| prefix | kway10 | sentence@k2 -> sentence@k8 | +0.0025 [-0.0054, +0.0100] |
| prefix | MRR | smolvlm_pca@k2 -> smolvlm_pca@k4 | -0.0001 [-0.0027, +0.0024] |
| prefix | kway5 | smolvlm_pca@k2 -> smolvlm_pca@k4 | +0.0029 [-0.0047, +0.0103] |
| prefix | kway10 | smolvlm_pca@k2 -> smolvlm_pca@k4 | -0.0053 [-0.0094, -0.0015] |
| prefix | MRR | smolvlm_pca@k4 -> smolvlm_pca@k6 | +0.0007 [-0.0013, +0.0028] |
| prefix | kway5 | smolvlm_pca@k4 -> smolvlm_pca@k6 | +0.0016 [-0.0069, +0.0101] |
| prefix | kway10 | smolvlm_pca@k4 -> smolvlm_pca@k6 | +0.0076 [+0.0026, +0.0126] |
| prefix | MRR | smolvlm_pca@k6 -> smolvlm_pca@k8 | +0.0008 [-0.0010, +0.0027] |
| prefix | kway5 | smolvlm_pca@k6 -> smolvlm_pca@k8 | +0.0064 [-0.0023, +0.0151] |
| prefix | kway10 | smolvlm_pca@k6 -> smolvlm_pca@k8 | -0.0043 [-0.0101, +0.0011] |
| prefix | MRR | smolvlm_pca@k2 -> smolvlm_pca@k8 | +0.0014 [-0.0016, +0.0043] |
| prefix | kway5 | smolvlm_pca@k2 -> smolvlm_pca@k8 | +0.0109 [+0.0023, +0.0198] |
| prefix | kway10 | smolvlm_pca@k2 -> smolvlm_pca@k8 | -0.0020 [-0.0094, +0.0050] |
| target (sentence minus smolvlm_pca) | MRR | smolvlm_pca@k2 -> sentence@k2 | -0.0005 [-0.0063, +0.0055] |
| target (sentence minus smolvlm_pca) | kway5 | smolvlm_pca@k2 -> sentence@k2 | -0.0025 [-0.0212, +0.0164] |
| target (sentence minus smolvlm_pca) | kway10 | smolvlm_pca@k2 -> sentence@k2 | -0.0045 [-0.0190, +0.0091] |
| target (sentence minus smolvlm_pca) | MRR | smolvlm_pca@k4 -> sentence@k4 | +0.0006 [-0.0056, +0.0069] |
| target (sentence minus smolvlm_pca) | kway5 | smolvlm_pca@k4 -> sentence@k4 | -0.0068 [-0.0232, +0.0090] |
| target (sentence minus smolvlm_pca) | kway10 | smolvlm_pca@k4 -> sentence@k4 | -0.0013 [-0.0154, +0.0119] |
| target (sentence minus smolvlm_pca) | MRR | smolvlm_pca@k6 -> sentence@k6 | +0.0007 [-0.0052, +0.0065] |
| target (sentence minus smolvlm_pca) | kway5 | smolvlm_pca@k6 -> sentence@k6 | -0.0109 [-0.0276, +0.0071] |
| target (sentence minus smolvlm_pca) | kway10 | smolvlm_pca@k6 -> sentence@k6 | -0.0089 [-0.0211, +0.0030] |
| target (sentence minus smolvlm_pca) | MRR | smolvlm_pca@k8 -> sentence@k8 | -0.0007 [-0.0069, +0.0056] |
| target (sentence minus smolvlm_pca) | kway5 | smolvlm_pca@k8 -> sentence@k8 | -0.0104 [-0.0263, +0.0058] |
| target (sentence minus smolvlm_pca) | kway10 | smolvlm_pca@k8 -> sentence@k8 | +0.0000 [-0.0135, +0.0127] |

## Notes

- Paraphrase diagnostic (sentence encoder): {'frac_queries_with_other_WHY_in_same_video_cos>=0.8': 0.6656012784658409, 'frac_queries_with_other_WHY_in_same_video_cos>=0.9': 0.4450659208949261}. Exact-text positives treat paraphrases as wrong, so absolute scores are capped.
- `smolvlm_pca` (if present) uses a PCA fit on the pilot training videos, not fold-wise.
- K-way distractors are drawn once per query from the same video (different WHY text), independent of model, prefix and seed.
