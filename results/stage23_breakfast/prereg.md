# Stage 23 pre-registration: Breakfast, "same action, different intent"

Written 2026-11-29 ~01:15 (Asia/Ho_Chi_Minh), after Part 0 inventory and BEFORE any accuracy is computed.
Only inventory statistics (counts, A* table, feature/label alignment, frame-diff magnitudes of the two halves) were seen.

## Data
- MS-TCN Breakfast package (zenodo record 3625992, `data.zip`), `data/breakfast/`: I3D features (2048, T) float32 per video,
  frame-level fine labels (`groundTruth`, 48 labels incl. `SIL`), 4 official splits (`splits/*.bundle`).
- Frame rate: 15 fps (as used by MS-TCN, Abu Farha & Gall CVPR 2019, arXiv 1903.01945; not re-checked in a README, none in zip).
- 1712 videos, 52 people, 10 dishes, 503 recordings (person+dish), 5 camera ids. Feature length == label length for all videos.
- Two halves of x_t: half1 = dims 0..1023, half2 = dims 1024..2047. No source found for which half is RGB / flow, so they are
  called half1 / half2. (Descriptive hint only: half2 changes ~2.2x more between consecutive frames.)
  Where the prompt says "RGB half" we use half1 and where it says "flow half" we use half2, and say so in every table.

## Common definitions
- Frame index t = 0..T-1. Second k = frames [15k, min(15k+15, T)); `seq1s[k]` = mean of x_t in second k (2048-d);
  `lab1s[k]` = majority fine label in second k (ties -> lowest label id).
- Observed prefix for fraction p: n = max(15, round(p*T)) frames (n <= T). Seconds "fully observed" = k with 15k+15 <= n
  (at least second 0 since n >= 15).
- Fine-label history (H, HIST) uses 47 non-SIL labels. Bag = counts of non-SIL segments, normalised to sum 1 (zeros if none);
  last2 = one-hot of the last and the second-to-last non-SIL segment label (94-d, zeros if missing). Total 141-d.

## Features, Part A (video-level, prefix p)
- S: mean of half1 over frames [0, 15) (first 1 s). Constant in p. Limitation: each I3D frame feature already sees a
  multi-frame window, so S is not a truly static frame (unlike SigLIP frame in Stage 18).
- M-Δ (main M): mean over t in [0, n) of (x_t - x_0), 2048-d.
- M-flow: mean of half2 over [0, n).
- S+M: [S, M-Δ] (3072-d).
- H-oracle: bag+last2 over true fine segments whose start < n.
- H-pred: bag+last2 over runs of the smoothed predicted labels of the fully observed seconds (see frame classifier).
- S+M+H-pred: concatenation.
- T (temporal model): PCA-256 of `seq1s` vectors, fit on train-fold seconds only (random subsample of at most 40,000 train
  seconds, seed 0, to fit in RAM). Input = PCA vectors of fully observed seconds. Model: Conv1d(256->128,k3,pad1)-ReLU-
  Conv1d(128->128,k3,pad1)-ReLU-masked mean pool over time-Linear(128->10). Loss: cross-entropy with balanced class weights
  (computed on the train part). Adam lr 1e-3, weight decay 1e-4, batch 32, max 30 epochs, early stopping on validation
  balanced accuracy (patience 5, keep best). Validation = 20% of train PEOPLE (chosen with the run's seed). Seeds 0,1,2,
  one model per (split, p, seed). If one split (all p, all seeds) takes > 30 min, channels -> 64 (logged in DEVIATIONS).
  T statistic = mean over seeds of balanced accuracy; each seed also reported.
- Secondary (descriptive) Part A variant: p measured from the first non-SIL frame f0: observed frames [f0, f0+n'), n' =
  max(15, round(p*(T-f0))); S = mean half1 over [f0, f0+15); M-Δ relative to x_{f0}. Linear rows S, M-Δ, M-flow, S+M only.

## Frame classifier (for H-pred and HIST-pred)
- Input: `seq1s` -> StandardScaler -> PCA-256 (fit on the same train data, subsample <= 40,000 seconds) ; target: `lab1s`
  (48 classes incl. SIL). Multinomial LogisticRegression, C = 1.0 fixed, max_iter 300, lbfgs, trained on at most 60,000
  train seconds (random, seed 0).
- Smoothing: causal majority vote over predicted labels in seconds [k-2, k] (trailing 3-s window, so no future frame is used).
- Cross-fitting for train features: train people of a split are divided into 3 groups (sorted people ids, round-robin);
  the H-pred/HIST-pred features of train videos come from a frame classifier trained on the other 2 groups. Test videos use
  the classifier trained on all train people of the split.

## Linear video/segment classifier (all non-T rows)
- StandardScaler fit on train only; LogisticRegression(class_weight="balanced", lbfgs, max_iter 2000).
- C chosen from {0.01, 0.1, 1, 10} by 3-fold GroupKFold over train people (score = balanced accuracy). If the chosen C is at
  an edge, extend the grid once (x0.1 below 0.01 -> 0.001, or x10 above 10 -> 100) and re-select; logged in config.json.

## Splits and statistics
- Main: 4 official Breakfast splits (13 test people each, disjoint, every video in test exactly once). Predictions pooled
  over the 4 splits.
- Bootstrap unit = person: 10,000 resamples of the 52 people with replacement (seed 0); a resampled person brings all their
  samples. 95% percentile CI. Paired differences computed on the same resample.
- Main metric: balanced accuracy (chance = 1/K). Also accuracy, macro-F1; majority baseline = accuracy of predicting the
  train-majority class (its balanced accuracy is 1/K).
- All cameras are used as samples. Extra descriptive row: one camera per recording (camera preference order by global
  count: cam01 > webcam01 > webcam02 > stereo01 > cam02), evaluated on the same predictions restricted to that subset.
- Label-shuffle check: within each train person, permute dish labels across that person's videos; rerun S+M at p25 once
  (per split). Expect ~chance.

## Part A: early dish prediction
- Samples: videos; label: dish (K=10); p in {10, 25, 50, 100}%.
- Rows: S, M-Δ, M-flow, S+M, T, H-oracle, H-pred, S+M+H-pred.
- HA-1: if S at p25 balanced accuracy >= 0.80 -> "scene already sufficient; cannot claim action understanding on Breakfast".
- HA-2 (main): (S+M) - S at p25: CI lower bound > 0.
- HA-3 (main): M-Δ at p25: CI lower bound > 1/K = 0.10.
- HA-4 (main, improvement): T - (S+M) at p25: CI lower bound > 0 AND point difference >= 0.03.
  H-pred and S+M+H-pred are secondary (descriptive only).
- Descriptive: H-oracle - H-pred (points lost to fine-action recognition errors).

## Part B: same action, different intent
- A* (fixed in Part 0 by the rule: >=3 dishes, each kept dish >=20 segments of >=1 s from >=8 people, top 8 by count):
  pour_milk (K=4: cereals, coffee, milk, pancake), crack_egg (3: friedegg, pancake, scrambledegg),
  take_plate (6: friedegg, juice, pancake, salat, sandwich, scrambledegg), butter_pan (3: friedegg, pancake, scrambledegg),
  take_cup (3: coffee, milk, tea), pour_oil (3: friedegg, pancake, scrambledegg). Only 6 actions pass.
- Samples: each segment [s, e) of action a (>= 1 s, any camera) in a video whose dish is among a's K_a dishes; label = dish.
- Rows:
  - SEG: mean of x_t (2048) over [s, e).  SEG-half2 / SEG-half1: mean of half2 / half1 over [s, e) (descriptive).
  - CTX-S: mean half1 over [max(0, s-15), s) (zero vector if s == 0).
  - HIST-oracle: bag+last2 over true non-SIL segments that end <= s (i.e. before segment a; a itself excluded).
  - HIST-pred: bag+last2 over runs of smoothed predicted labels of seconds k with 15k+15 <= s.
  - PRE: mean x_t over [0, s) (zero if s == 0).
  - SEG+HIST-pred, SEG+CTX-S: concatenations.
- Leakage asserts: CTX-S, HIST-*, PRE use only frames < s; SEG only frames in [s, e).
- One classifier per action a and split. Per-a balanced accuracy (chance 1/K_a) and MACRO over a (main), macro chance =
  mean_a 1/K_a = (1/4 + 1/3 + 1/6 + 1/3 + 1/3 + 1/3)/6 = 0.2917.
- HB-1: SEG macro CI lower bound > 0.2917.
- HB-2 (main): HIST-pred - SEG (macro): CI lower bound > 0.
- HB-3: HIST-pred - CTX-S (macro): CI lower bound > 0.
- Descriptive: SEG-half2 vs SEG-half1; HIST-oracle - HIST-pred; per-a pattern; PRE.
- Conclusion sentence chosen by rule:
  1. HB-2 and HB-3 pass -> "On Breakfast, a single action is not enough to know the intent; the preceding action sequence
     predicts the intent better than both the action segment and the scene just before it."
  2. HB-2 passes, HB-3 fails -> "The action sequence helps more than a single action, but not more than the scene just before."
  3. HB-2 fails -> "Not enough evidence that the action sequence predicts intent better than a single action."

## Anything changed after seeing results goes to DEVIATIONS.md and is labelled exploratory.
