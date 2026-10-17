# E1 zero-shot interpretation

All numbers below are computed from the saved E1 CSVs. The evaluation uses train/validation only and is not an untouched-test result.

## Main table

| Branch | k | MRR | Chance | MRR lift [95% CI] | Top-1 | Top-3 | 5-way acc. | 5-way lift [95% CI] | 10-way acc. |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| full | 2 | 0.0671 | 0.02977 | 2.25 [1.87, 2.75] | 0.0172 | 0.0479 | 0.2773 | 1.39 [1.23, 1.56] | 0.1570 |
| full | 4 | 0.0715 | 0.02977 | 2.40 [2.02, 2.88] | 0.0212 | 0.0563 | 0.2753 | 1.38 [1.22, 1.55] | 0.1614 |
| full | 6 | 0.0734 | 0.02977 | 2.47 [2.06, 2.97] | 0.0220 | 0.0571 | 0.2809 | 1.40 [1.24, 1.58] | 0.1674 |
| full | 8 | 0.0765 | 0.02977 | 2.57 [2.13, 3.14] | 0.0236 | 0.0595 | 0.2885 | 1.44 [1.27, 1.62] | 0.1774 |

## Preregistered decisions

- **G1 (full-frame branch): FAIL.** MRR-lift lower CI > 1 at 4/4 prefixes ([2, 4, 6, 8]); 5-way-lift lower CI > 1 at 4/4 ([2, 4, 6, 8]); 5-way accuracy >= 0.30 at 0/4 ([]).
- **G2 (HOI helps): SKIPPED.** The preregistered detection gate failed; hand/center branches were not evaluated.

## Detection gate and branch fallbacks

- Overall hand detection rate: 0.5819.
- Gate: **FAIL** (overall >= 0.60 and every scene >= 0.30).
- Scene `art_studio` detection rate: 0.6181.
- Scene `bedroom` detection rate: 0.6721.
- Scene `clothing_closet` detection rate: 0.8882.
- Scene `deck` detection rate: 0.1026.
- Scene `farm` detection rate: 0.3218.
- Scene `garage` detection rate: 0.6184.
- Scene `hallway` detection rate: 0.8453.
- Scene `kitchen` detection rate: 0.7156.
- Scene `laundry_room` detection rate: 0.1533.
- Scene `living_room` detection rate: 0.6193.
- Scene `space` detection rate: 0.2478.
- Scene `workshop` detection rate: 0.7620.
- Scene `yard` detection rate: 0.1587.

## Secondary, preregistered diagnostics

These variants do not affect G1/G2 decisions.

| Branch | k | Clean 5-way acc. | Valid | Excluded | Paraphrase-aware MRR | Chance | Lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 2 | 0.2877 | 2503 | 0 | 0.1003 | 0.05264 | 1.906 [1.537, 2.368] |
| full | 4 | 0.2940 | 2503 | 0 | 0.1046 | 0.05264 | 1.987 [1.638, 2.430] |
| full | 6 | 0.2980 | 2503 | 0 | 0.1072 | 0.05264 | 2.036 [1.664, 2.512] |
| full | 8 | 0.3080 | 2503 | 0 | 0.1120 | 0.05264 | 2.127 [1.713, 2.662] |

## Limitations

- The evaluated set contains 27 source videos and 13 scenes.
- Exact-text positives are used for the primary MRR; paraphrases may be counted as errors.
- The SigLIP tokenizer truncated 0 unique WHY strings at max_length=64.
- Same-video K-way distractors may still be semantic paraphrases; the clean 5-way variant is secondary.
- No untouched test set was evaluated. The results do not establish generalization, significance, or superiority.
- Any HOI crops are fixed geometric proxies based on hand landmarks and are not an object detector.
