# Locked candidate test (one-shot): interpretation (facts only)

Run once by the user on 2026-10-23, 18:20–18:38, after creating `USER_APPROVED_TEST_EVAL`. FROZEN.json sha256 `ef0cd19d…aa484`; code at commit `f0a50e4`. Test set: 407 samples / 4 video UIDs of `results/test_protocol_full_v1/candidate_test_manifest.csv`. The C2 fallback-to-C1 fraction is 0.98%. Text truncations: 0. Control donors were always from another scene (0 fallbacks).

## Pre-registered verdicts (bootstrap over 4 test videos, B=10000, seed 0; exploratory: 4 clusters)

- **T1: PASS.** The frozen pipeline's 5-way lift CI lower bound is > 1 at 4/4 prefixes. Accuracy is 0.219, 0.246, 0.278 and 0.302 at k = 2, 4, 6, 8; lift lower bounds are 1.037 to 1.293.
- **T2: PASS.** The paired frozen − baseline 5-way delta CI lower bound is > 0 at 3/4 prefixes (k = 4, 6, 8): +0.034 [+0.013, +0.048], +0.054 [+0.010, +0.097], +0.089 [+0.044, +0.148]. At k=2 it is +0.015 [−0.010, +0.038].

## Other facts (no verdict)

- Gallery MRR: frozen 0.046 to 0.062, baseline 0.059 to 0.066 (chance 0.032). Frozen − baseline MRR is negative at all 4 prefixes, with CI below 0 at k=2 (−0.017 [−0.032, −0.002]).
- 10-way: frozen − baseline is +0.034 to +0.084, with CI lower bound > 0 at 4/4 prefixes.
- Mismatched-video control (51 queries, k=8), 5-way: frozen real 0.314 vs control 0.216; baseline real 0.176 vs control 0.216.

## Limitations

- Only 4 test videos: every interval rests on 4 bootstrap clusters and is exploratory.
- This is a candidate test that is video-UID-disjoint but not scene-disjoint (`results/test_protocol_full_v1/README.md`).
- Same-video distractors may be paraphrases.
- This test is used once; it must not be re-run or used for selection.
