# Locked candidate test results

Evaluated once using the protocol in `test_protocol_full_v1/locked_protocol.json`. The reported test is candidate-only, video-UID-held-out, not scene-disjoint, and not guaranteed untouched outside the inspected local artifacts.

| Frames | MRR mean ± SD | Top-1 mean ± SD | Top-3 mean ± SD | Top-5 mean ± SD | Test examples/seed |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0290 ± 0.0028 | 0.0057 ± 0.0014 | 0.0172 ± 0.0000 | 0.0287 ± 0.0038 | 407 |
| 4 | 0.0288 ± 0.0033 | 0.0082 ± 0.0038 | 0.0147 ± 0.0043 | 0.0246 ± 0.0000 | 407 |
| 6 | 0.0294 ± 0.0045 | 0.0090 ± 0.0051 | 0.0172 ± 0.0049 | 0.0221 ± 0.0043 | 407 |
| 8 | 0.0314 ± 0.0035 | 0.0098 ± 0.0049 | 0.0197 ± 0.0025 | 0.0270 ± 0.0025 | 407 |

Exact multi-positive random-ranking chance on this candidate gallery is MRR 0.0318, Top-1 0.0071, and Top-3 0.0208. Thus the highest mean MRR (8 frames) is approximately at chance, while Top-1 is only slightly above this gallery-specific chance; this does not demonstrate useful generalization.

Checkpoint selection used validation MRR only. No test-based threshold tuning or model selection was performed. Seed variation and repeated samples within each source video are not independent video-level replicates. Three initial attempts stopped before any metric was produced because of implementation/runtime issues; the failures and corrective addendum are recorded in `../test_protocol_full_v1/protocol_addendum.json`. Only the final successful run produced test metrics.

This test candidate contains four source UIDs from two scenes; one full-test UID overlapping pilot validation was excluded before protocol lock. It shares the garden scene with pilot validation, so the result is not scene-independent.
