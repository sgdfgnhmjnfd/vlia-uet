# Scene-grouped CV interpretation

## Run facts

The completed run evaluated 2,503 examples from 27 source videos and 13 scenes
using 8 scene-grouped folds, 3 seeds, prefixes of 2/4/6/8 frames, and targets
`sentence` and `smolvlm_pca`. Only the `train` and `val` cache directories were
loaded; the locked `test` split was not read. Every 5-way and 10-way query had a
valid candidate set (2,503 each). The mean exact-gallery random-ranking MRR
chance was 0.02977; fixed K-way chance was 0.20 and 0.10, respectively.

## Main results

Values are mean ± standard deviation across seeds. Lift intervals are 95%
source-video bootstrap intervals; they are descriptive.

| Target | Frames | MRR | MRR lift [95% CI] | Top-1 | Top-3 | 5-way acc. | 5-way lift [95% CI] | 10-way acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence | 2 | 0.0335 ± 0.0012 | 1.13 [0.86, 1.43] | 0.0067 | 0.0202 | 0.2163 ± 0.0055 | 1.08 [1.00, 1.16] | 0.1131 ± 0.0046 |
| sentence | 4 | 0.0345 ± 0.0013 | 1.16 [0.88, 1.48] | 0.0092 | 0.0206 | 0.2149 ± 0.0046 | 1.07 [1.00, 1.16] | 0.1109 ± 0.0054 |
| sentence | 6 | 0.0353 ± 0.0016 | 1.18 [0.90, 1.52] | 0.0093 | 0.0213 | 0.2124 ± 0.0027 | 1.06 [0.98, 1.16] | 0.1109 ± 0.0048 |
| sentence | 8 | 0.0347 ± 0.0010 | 1.17 [0.87, 1.51] | 0.0085 | 0.0206 | 0.2193 ± 0.0103 | 1.10 [1.01, 1.20] | 0.1156 ± 0.0054 |
| smolvlm_pca | 2 | 0.0340 ± 0.0013 | 1.14 [0.95, 1.36] | 0.0073 | 0.0213 | 0.2188 ± 0.0038 | 1.09 [1.02, 1.17] | 0.1176 ± 0.0062 |
| smolvlm_pca | 4 | 0.0338 ± 0.0016 | 1.14 [0.94, 1.35] | 0.0076 | 0.0201 | 0.2217 ± 0.0065 | 1.11 [1.03, 1.19] | 0.1123 ± 0.0052 |
| smolvlm_pca | 6 | 0.0346 ± 0.0019 | 1.16 [0.95, 1.39] | 0.0079 | 0.0230 | 0.2233 ± 0.0050 | 1.12 [1.04, 1.20] | 0.1199 ± 0.0036 |
| smolvlm_pca | 8 | 0.0354 ± 0.0025 | 1.19 [0.98, 1.42] | 0.0081 | 0.0210 | 0.2297 ± 0.0073 | 1.15 [1.07, 1.23] | 0.1156 ± 0.0074 |

## Pre-registered decision rules

- **R1, above chance:** MRR lift lower bounds exceed 1 at 0/4 prefixes for
  `sentence` and 0/4 for `smolvlm_pca`; neither target meets the MRR criterion.
  For 5-way lift, lower bounds exceed 1 at 2/4 prefixes for `sentence` (2 and
  8 frames) and 4/4 for `smolvlm_pca`; both meet the literal 5-way criterion.
- **R2, adopt the sentence target:** the paired `sentence - smolvlm_pca` lower
  confidence bound is above zero at 0/4 prefixes for MRR and 0/4 for 5-way
  accuracy. The adoption criterion is not met.
- **R3:** record a negative result for adopting the sentence target under this
  rule. The above-chance 5-way results do not establish a sentence-target
  advantage. No claim of statistical significance or generalization is made.

The interval lower bounds were compared before rounding, directly from
`paired_differences.csv` and `summary.csv`.

## Paired prefix differences

On identical out-of-fold examples, the sentence-target MRR change from 2 to 8
frames was +0.00119 [−0.00136, +0.00387]; the 5-way accuracy change was
+0.00306 [−0.00688, +0.01361]. For `smolvlm_pca`, the corresponding changes
were +0.00142 [−0.00156, +0.00434] MRR and +0.01092 [+0.00227, +0.01977]
5-way accuracy. These are video-bootstrap intervals, not independent-video
replications across a broader population.

## Video-level heterogeneity

Per-prefix counts of videos with MRR lift below 1:

| Target | 2 frames | 4 frames | 6 frames | 8 frames |
|---|---:|---:|---:|---:|
| sentence | 11/27 | 12/27 | 8/27 | 10/27 |
| smolvlm_pca | 10/27 | 10/27 | 9/27 | 12/27 |

Video UIDs below chance at one or more prefixes (scene in parentheses):

- `sentence` (14): `02d1f024-1470-4ce6-acde-b938f7847eb6` (farm),
  `2927605b-bdb2-4b36-8181-12f9e898bf2d` (space),
  `3ae7ab6b-793e-4a77-9749-010c94b1b0ee` (workshop),
  `6269fa52-58af-46d3-9a4b-7aca66c3b622` (garage),
  `6d738e16-abe9-4e7f-870b-1c60ea2cd257` (kitchen),
  `93232ea0-b8eb-4df8-85b8-e8ff189b41f2` (yard),
  `93357145-49fb-4128-b5da-25bc9602daf0` (garage),
  `97c1c805-291f-4b2f-bef9-0495d9d2fc92` (kitchen),
  `b2cdc6c2-5ecc-4236-aae2-947cbf54fe49` (workshop),
  `ca6e9ec9-46ac-4a02-8034-7c31157dc52c` (kitchen),
  `edf2c742-f89c-402b-9228-11fda87201f1` (workshop),
  `ee05e77f-d52f-48d7-b19f-6b14411a7849` (laundry_room),
  `f6fea609-23c4-431c-8ac2-484f334a3a83` (bedroom),
  `f8945922-b433-4f9a-a652-b9e4e1499add` (hallway).
- `smolvlm_pca` (17): `02d1f024-1470-4ce6-acde-b938f7847eb6` (farm),
  `1fd2bbfe-80e2-4543-a9fd-3a68bf6fdfd7` (farm),
  `2927605b-bdb2-4b36-8181-12f9e898bf2d` (space),
  `3ae7ab6b-793e-4a77-9749-010c94b1b0ee` (workshop),
  `6269fa52-58af-46d3-9a4b-7aca66c3b622` (garage),
  `6d738e16-abe9-4e7f-870b-1c60ea2cd257` (kitchen),
  `8762cf76-5526-43a2-b825-a0c396494c87` (clothing_closet),
  `8bef8b33-a1ee-43e7-a2d0-8cf68653cb0c` (art_studio),
  `93232ea0-b8eb-4df8-85b8-e8ff189b41f2` (yard),
  `93357145-49fb-4128-b5da-25bc9602daf0` (garage),
  `97c1c805-291f-4b2f-bef9-0495d9d2fc92` (kitchen),
  `9a473135-6125-4081-9327-6c4afb4348af` (garage),
  `b2cdc6c2-5ecc-4236-aae2-947cbf54fe49` (workshop),
  `c2e6d807-d903-4b64-98e1-2c07ca700c78` (living_room),
  `ca6e9ec9-46ac-4a02-8034-7c31157dc52c` (kitchen),
  `f6fea609-23c4-431c-8ac2-484f334a3a83` (bedroom),
  `f8945922-b433-4f9a-a652-b9e4e1499add` (hallway).

## Additional diagnostics and limitations

- Mean selected epoch was 57.15 for `sentence`, 68.51 for `smolvlm_pca`, and
  62.83 averaged over all target-prefix summaries.
- In the sentence-embedding paraphrase diagnostic, 66.56% of queries had a
  different WHY in the same video with cosine similarity at least 0.8; 44.51%
  had similarity at least 0.9. Retrieval positives use exact text, so
  paraphrases are counted as non-matches.
- The split contains only 27 videos and 13 scenes. Video-bootstrap intervals
  describe this small set and cannot establish broad generalization or fully
  cover domain shift.
- `smolvlm_pca` uses a PCA fit on pilot training videos, not a fold-wise fit.
- K-way distractors are sampled once per query from the same video, and the
  reported K-way task is limited to those candidate sets.
- This is train/validation-only cross-validation, not an untouched test result.
