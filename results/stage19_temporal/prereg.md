# Stage 19 pre-registration: temporal video features (method B only), scene vs motion

Written 2026-11-24 before any Stage 19 accuracy. Trang (STOP A): method B only; method A only if time remains in
the 4-hour budget (started 16:39). Order fixed by Trang: own videos p25/50/100 first, then EgoIntent p50, then
EgoIntent p100 only if time remains.

## Encoder
MCG-NJU/videomae-base-finetuned-ssv2 (frozen, CPU, 2 threads). 16 frames, shorter side resized to 224 (bilinear),
center crop 224, ImageNet mean/std. Feature = fc_norm(mean of last-layer tokens) (the input to the SSv2
classifier), L2-normalised. transformers 5.5 leaves the q/k/v biases un-loaded; they are copied by hand from the
checkpoint (q_bias, v_bias; key bias 0), checked by plausible SSv2 top-1 labels on 7 clips (no accuracy computed).

## Variants
- S: the unit's first frame repeated 16x (static clip: scene only). Independent of p.
- V (= S+M): 16 frames evenly spaced over the observed window.
- M: V - S.

## Windows
- Own videos (Stage 16, e41e72c): window [t_on, t_on + p% (t_end - t_on)], t_end = duration - 1 s, p in {25, 50, 100};
  first frame = frame 0 of the clip (idle part). Shared tables A-D, 81 clips.
- EgoIntent (Stage 18): 1,001 kept steps, 10 verbs. Window = frame 0 to the 4th of the 8 Stage 18 cached frame
  positions (p50, 3/7 of the step) or to the last frame (p100), so that the observed part equals Stage 18's.

## Classifier, splits, metrics (as Stage 16 / 18)
Stage 18 torch LogReg, class-balanced weights, standardised on train, C grid {0.01..100} + one edge extension.
- Own videos: per-table leave-one-person-out over the shared tables, C by inner leave-one-rep-out (Stage 16).
  Accuracy pooled over A-D, chance 0.444. CI: bootstrap of clips within table, 10,000, seed 0.
- EgoIntent: Stage 18 scene-grouped 8 folds, inner 4 folds of whole training videos, balanced accuracy, chance
  0.10, CI by video bootstrap 10,000 seed 0.

## Rules
- R1: S >= 0.80 -> "scene suffices".
- R2: (S+M p50) - S, paired CI lower bound > 0.
- R3: M p50 CI lower bound > chance.
- R4: B S+M p50 - previous S+M p50 (EgoIntent: Stage 18 S+M p50 per step from ego_oof.npz; own videos: Stage 16 F+
  p50 per clip, recomputed from Stage 16 SigLIP embeddings with the Stage 16 unweighted LogReg; must reproduce
  0.741), paired on the same units, CI lower bound > 0. Only B is run, so B is "the best method".
- Conclusion per dataset as Stage 18: R1 holds -> scene suffices; else R2 and R3 -> evidence for action; else
  insufficient evidence.
- Sanity (required): S+M p50 with labels permuted (seed 0) must be near chance (CI should contain chance).
- Descriptive: train accuracy (C = 100) for S, M, S+M p50; EgoIntent confusion matrix S+M p50; curve.png.
