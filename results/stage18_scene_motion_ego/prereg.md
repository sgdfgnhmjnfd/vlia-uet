# Stage 18 pre-registration: scene vs motion on LIBERO-10, own videos, EgoIntent

Written 2026-11-24, after STOP A (label counts only; no accuracy of any kind computed yet).
Question (advisor): does the model infer intent from the *action*, or from the *static scene*?

## Features (shared)
Frozen SigLIP so400m-patch14-384 image embeddings, L2-normalised (existing caches). z_1 = first frame of the
unit, z_t = frames observed up to prefix p (k frames).
- **S** (scene-only): z_1. Independent of p.
- **M** (motion-only): mean_{t=2..k}(z_t - z_1). Undefined ("-") when k = 1.
- **S+M**: concat[z_1, M].
- **F+** (descriptive only, Stage 16 definition): concat[z_1, mean_{t=1..k} z_t].
- **S-vid** (EgoIntent only, descriptive): mean of z_1 over all kept steps of the same video (scene identity, no step info).

## Classifier (all datasets, all features)
Multinomial logistic regression, Stage 15/16 implementation (torch L-BFGS, objective C*sum CE + 0.5||W||^2),
class-balanced sample weights (n / (K n_c), computed on the training fold), features standardised on the training
fold only. C in {0.01, 0.1, 1, 10, 100} by inner CV on training units only; if the best C is on the grid edge, the
grid is extended once by x10 / /10 and recorded. One model per (feature, p). Ties -> smaller C.

## EgoIntent
- Units: steps of the 27 development videos (2,503 steps); locked test (5 video_uids in
  results/test_protocol_full_v1/locked_protocol.json) is never read (asserted).
- Labels: first verb of WHAT by the fixed rule in label_rule.md (lower-case, strip punctuation, skip fixed leading
  non-verbs/adverbs, inflection-only lemma, particle up/down/out/off/away/back merged). Keep verbs with >= 30 steps
  and >= 8 videos, top 10: place, pick_up, grasp, wipe, hold, position, adjust, remove, align, walk
  (K = 10, 1,001 steps, largest class 25.5%). Other steps dropped.
- Prefix: 8 cached frames spread linearly from first to last frame of the step. p25 = frames 1-2, p50 = 1-4,
  p100 = 1-8. Seconds reported = time of the last observed frame, (k-1)/7 x step duration (from the mp4 header).
- Split (main): the existing scene-grouped 8 folds (run_scene_cv.assign_folds, 13 scenes); no video and no scene in
  both train and test. Inner CV: 4 folds of whole training videos (greedy balance by step count, deterministic).
  No person/session id exists in the metadata, so no leave-one-person-out; the same camera wearer may appear in
  train and test (stated as a limitation).
- Main metric: balanced accuracy (chance 1/K = 0.10). Also accuracy, macro-F1, train-fold majority-class baseline.
- CI: bootstrap over test videos (whole videos resampled), 10,000 draws, seed 0, percentile 95%.
  Paired differences use the same draws on the same steps.

## LIBERO-10 (agent view)
- Same 379 demos, outer 5 folds copied from Stage 15 config.json (outer_fold.agent), inner folds
  strat_folds(labels of sorted training demos, 4, seed 1+fold) as Stage 15. p in {10, 25, 50, 100}; frames = Stage 15
  sample8 of the 4 fps cache up to p% of the demo. S = first frame (Stage 15 p0).
- Main metric: accuracy (classes nearly balanced; chance 0.10); bootstrap stratified by task (Stage 15 strat_boot),
  10,000 draws, seed 0.
- Consistency check (gate): S accuracy within 88.7 +/- 3 points and S+M at p25 within 99.7 +/- 3 points of Stage 15.
  Otherwise stop and report.
- Caveat: robot demos, no human; each task has its own table layout, so an unseen demo is still a seen scene.

## Own videos
Not recomputed. Copied from results/stage16_scene_motion/accuracy.csv at commit e41e72c (branch
claude/stage16-scene-motion, Stage 16 prereg sha256 4e041e2e...), table "A,pooled" (S, M and F+ at p25/50/100), LOPO.

## Reading rules (fixed now)
- **R1** (scene suffices): S >= 0.80 (LIBERO, own videos: accuracy; EgoIntent: balanced accuracy).
- **R2** (EgoIntent main; also computed for LIBERO): (S+M) - S at p50, paired CI lower bound > 0.
- **R3** (EgoIntent main; also computed for LIBERO): M at p50, CI lower bound > 1/K.
- Conclusion per dataset (by rule only):
  - R1 holds -> "Scene suffices; cannot claim action understanding."
  - else R2 and R3 hold -> "Evidence the model uses action, not only scene."
  - else -> "Insufficient evidence."
  Own videos: Stage 16 verdicts (H1-a/b/c PASS) copied; R1 applied to its S.
- Descriptive only: accuracy-vs-p curves for S, M, S+M, F+; S-vid; confusion matrix S+M p50 (EgoIntent); per-video
  accuracy (S, S+M p50); best/worst classes for M p50.
