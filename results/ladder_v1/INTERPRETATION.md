# Ladder v1 Stage 3 (nested pipeline ladder): interpretation (facts only)

Data: 2503 train/val queries, 27 videos, 13 scenes, scene-grouped 8 outer folds (`assign_folds(scenes, 8)`). The locked test split was not read.

Stage 2 cache: all 5 preregistered encoders were extracted; none was dropped. The 200-sample probe projected 5600 s in total, under the 3 h limit. The baseline `siglip-base-patch16-224` is float32 (amendment 1); it reproduced E1's K=5 hits exactly (Δ = 0 at k = 2, 4, 6, 8). The other encoders are float16 frames per the preregistration. Text truncations were 0 for every encoder and template. Frames of every later pass matched the first pass's SHA1 hashes.

Grid: 5 encoders × 3 aggregators × 3 centerings × 2 templates = 90 configs. C2 parsed numeric `step_id`; the fallback-to-C1 fraction (no earlier step in the same video) is 1.08%.

## Selection

The same config was selected in **8 of 8 folds**: `siglip-so400m-patch14-384 | A0 (mean) | C2 (causal earlier-step context) | T1 (template)`. Its training-fold mean 5-way accuracy was 0.348 to 0.371, against 0.263 to 0.295 for the baseline on the same training folds. There was no tie at the maximum in any fold.

## Held-out results (pooled out-of-fold; video bootstrap B=10000, seed 0)

| k | baseline 5-way | nested 5-way | 5-way delta [95% CI] | baseline MRR | nested MRR | MRR delta [95% CI] |
|---:|---:|---:|---|---:|---:|---|
| 2 | 0.2773 | 0.3408 | +0.0635 [+0.0389, +0.0892] | 0.0671 | 0.0747 | +0.0076 [−0.0037, +0.0195] |
| 4 | 0.2753 | 0.3492 | +0.0739 [+0.0491, +0.0999] | 0.0715 | 0.0777 | +0.0062 [−0.0059, +0.0183] |
| 6 | 0.2809 | 0.3624 | +0.0815 [+0.0510, +0.1120] | 0.0734 | 0.0825 | +0.0091 [−0.0024, +0.0206] |
| 8 | 0.2885 | 0.3871 | +0.0987 [+0.0621, +0.1344] | 0.0765 | 0.0850 | +0.0085 [−0.0026, +0.0195] |

The 10-way delta is +0.049 to +0.058, with the CI lower bound > 0 at all 4 prefixes. Nested lifts: MRR 2.51 to 2.86 (CI lower bounds 1.94 to 2.24); 5-way 1.70 to 1.94 (CI lower bounds 1.52 to 1.72).

## Verdicts

- **N1: FAIL ("Selection does not help" rule not met).** The 5-way CI lower bound is > 0 at 4/4 prefixes, but the gallery-MRR CI lower bound is > 0 at 0/4. The rule requires both metrics.
- **N2: PASS.** Nested-selected 5-way accuracy is ≥ 0.30 at 4/4 prefixes; the lift CI lower bound is > 1 for MRR at 4/4 and for 5-way at 4/4.
- **G1** from the earlier E1 evaluation remains **FAIL and unchanged**. N2 is the new practical bar, not a re-evaluation of G1.

## Control (every 8th query by sample_id, k=8, frames from a different video and scene)

| config | n | real 5-way | control 5-way |
|---|---:|---:|---:|
| so400m / A0 / C2 / T1 (selected) | 313 | 0.3962 | 0.1885 |
| baseline | 313 | 0.2620 | 0.1949 |

## Descriptive full grid (`grid_descriptive.csv`; not selection-corrected; no verdict)

- The top 6 configs by mean 5-way over k all use so400m with C2 (0.349 to 0.360).
- The same pipeline without C2 (`so400m | A0 | C0 | T0`) gives 0.311 to 0.330. Baseline encoder with C2 and T1 (`siglip-base-224 | A0 | C2 | T1`) gives 0.291 to 0.321.
- The lowest configs are CLIP ViT-L with C0 (0.267 to 0.272).

## Secondary (no verdict)

Clean 5-way (distractor MiniLM cosine < 0.8): nested 0.342 to 0.394 vs baseline 0.288 to 0.308. Paraphrase-aware MRR: nested 0.101 to 0.115 vs baseline 0.100 to 0.112.

## Limitations

- Only 27 source videos. All CIs bootstrap whole videos.
- Same-video distractors may be paraphrases of the positive.
- Grid selection used training scenes only. Because the same config won every fold, the selection is stable but was made on overlapping training sets.
- C2 uses frames of earlier steps of the same video. This is causal within a video, but it assumes the earlier step clips are available at decision time.
- The baseline is float32 while the other encoders are float16 frames (per preregistration plus amendment 1).
- No untouched test set was evaluated.
