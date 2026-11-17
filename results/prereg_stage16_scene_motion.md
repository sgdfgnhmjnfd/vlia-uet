# Pre-registration: Stage 16, does the model read the action or the table? (written 2026-11-17, before any accuracy)

- **Sources:** "kế hoạch cuối (17/11/2026)", Part 2, and Appendix A (Stage 16 prompt of 16/11).
- **Trang's answers in chat on 17/11:**
  - t_on is labelled by hand, option (a);
  - table C is included;
  - the continuous curve extends to start_s + 2 s and is reported against both t_on and start_s;
  - scene-only and full are compared after the touch as well;
  - the main question is how many seconds earlier motion recognises the task than the table alone.
- **What was known before writing this file:** only Step 0. That is the inventory, the D1 gate, the manual t_on and the hand-visible shares. No classifier was trained and no accuracy was computed.
- **Changes after this point** go to `results/stage16_scene_motion/DEVIATIONS.md`.

## Data

- **Clips:** 90 clips in `/media/dhqg/d3/datasets/own_libero10` (`labels_all.csv`): 10 tasks × 3 persons × 3 reps.
- **Tables:** A = {0, 1, 7}, B = {4, 6}, C = {2, 8}, D = {3, 9}, E = {5}.
  - The shared tables A, B, C and D (81 clips) enter every within-table analysis. All four pass gate D1.
  - E is used only in the 10-way reference.
- **start_s, end_s:** from `labels_all.csv`. start_s is the first touch.
- **t_on:** the manual label in `results/stage16_scene_motion/t_on_manual.csv`, the first 0.25 s grid frame with the acting hand in view.
- **t_end:** clip duration − 1 s.
- **Hand-visible shares,** known now (a cut at start_s − H counts as hand-visible if cut ≥ t_on):

  | H | 2 s | 1 s | 0.5 s | 0.25 s |
  |---|---:|---:|---:|---:|
  | share | 0% | 0% | 35.6% | 82.2% |

  start_s − t_on: min 0, median 0.25 s, max 0.75 s.

## Frames and features

- **Embeddings:** frozen SigLIP so400m-patch14-384 image embeddings, L2-normalised. Frames come from the 4 fps grid t = 0, 0.25, … (nearest video frame; cv2 applies any rotation; all clips have rotation 0).
- **Frames per window:** at most 16, evenly spaced indices of the grid frames inside the window.
- **Reference (still table):** frames in [0, r] with r = min(1.0, t_on − 0.25, cut). r = 1.0 for every clip at every cut ≥ 1 s. A query with fewer than 3 reference frames is excluded and listed. z_ref is their mean.
- **The three feature sets:**
  - **S (scene-only)** = z_ref. It does not depend on the cut.
  - **M (motion-only)** = mean of (z_t − z_ref) over the observed frames. It is the zero vector when there is no observed frame.
  - **F+ (full)** = [S, mean of z_t over the observed frames]. It is [S, S] when there is no observed frame.
- **Part A observed window** = [t_on, t_on + p/100 · (t_end − t_on)], p ∈ {0, 25, 50, 75, 100}.
  - p = 0 has no observed frame: F+ = [S, S] and M = 0.
  - Seconds observed are reported per p (mean and range).
- **Part A2 and curve observed window** = (r, cut]. Only frames with timestamp ≤ cut enter a query. This is checked by a unit test, and max(frame time − cut) is logged and must be ≤ 0.

## Classifier and split

- **Classifier:** the Stage 15 multinomial logistic regression (torch, L-BFGS, loss C · summed cross-entropy + ½‖W‖²). Features are standardised with the training-fold mean and sd.
- **Choosing C:** C ∈ {0.01, 0.1, 1, 10, 100} by inner leave-one-rep-out (reps 1, 2, 3) on the training persons only.
  - Ties go to the smaller C.
  - If the best C is at an edge, the grid is extended once by ×10 on that side, and this is reported.
- **Split:** within-table, one model per table, over that table's tasks. LOPO: each person is held out in turn, and the model trains on the other two persons' clips.
- **10-way reference:** the same procedure, over all 90 clips.
- **Prediction:** argmax of the probabilities (ties go to the lower task id).

## Metrics

- **Accuracy and macro-F1,** pooled over held-out persons, per table and pooled over shared tables, with Wilson 95% CIs.
- **Chance:** 1/k per table. Pooled chance is clip-weighted: (27 · 1/3 + 54 · 1/2) / 81 = 0.444.
- **Paired differences:** bootstrap over clips, resampled within table (B = 10,000 for the rules, B = 2,000 for curve points, seed 0).

## Part A rules (fixed now)

- **H1-a (main):** pooled acc(F+, p50) − acc(S), paired. The CI lower bound must be > 0.
- **H1-b (design check):**
  - PASS if the Wilson lower bound of pooled S accuracy is ≤ pooled chance;
  - "inconclusive" if the upper bound is > chance + 0.25;
  - if H1-b fails, the scene leaks the task and H1-a is not interpretable.
- **H1-c:** the Wilson lower bound of pooled acc(M, p50) must be > pooled chance.
- **Descriptive:**
  - curves against p and against seconds observed, per table;
  - the 10-way accuracy (S can be high there because tables differ);
  - untrained F, i.e. the Stage 13 pipeline unchanged: siglip so400m, mean of 8 frames on [0, start_s + 0.5 (end_s − start_s)], C-scene context (idle frames of the other clips of the same table and person), T1 template, within-table argmax over that table's LIBERO descriptions, scored strict;
  - most-frequent class per table (the training-fold majority; ties go to the lower id);
  - confusion matrix at p50 for F+;
  - per held-out person at p50 for F+ and S.

## Part A2: before the first touch

- **Cuts:** cut = start_s − H, H ∈ {2.0, 1.0, 0.5, 0.25} s. Each query is tagged hand-visible (cut ≥ t_on) or no-hand. No-hand queries are scene-only queries by construction and are never reported as anticipation.
- **Training:** a separate LOPO model per H and per feature set, trained on cuts at the same H.
- **Table A:**
  - primary: 3-class (0, 1, 7);
  - secondary (pre-registered): a separately trained 2-class model, "soup first" (0 + 7) vs 1.
  - Pooled results use the 3-class labels.
- **H1-d:** hand-visible queries at H = 0.5; pooled F+ − S on the same clips; the CI lower bound must be > 0.
  - It is "not testable" if fewer than 60% of clips have a hand-visible query at 0.5 s.
  - That share is 35.6% (known now), so **H1-d is "not testable"**. Its numbers are still reported, as descriptive.
- **Wording:** this is task-level anticipation before the first touch, not prediction of each next primitive.

## Continuous curves (descriptive) and the main question

- **Two alignments,** one LOPO model per cutoff, per feature set (S, M, F+) and per table:
  - **(i) on start_s:** cut = start_s + c, c ∈ {−2.5, −2.25, …, +2.0} (19 cutoffs). c ≤ 0 is "before touch" and c > 0 is "after touch". The two regimes are reported in separate tables and on separate panels.
  - **(ii) on t_on:** cut = t_on + d, d ∈ {−1.0, −0.75, …, +3.0} (17 cutoffs).
- **Per cutoff:** accuracy, macro-F1, n, hand-visible share, and the paired F+ − S difference with its CI.
- **Main question:** how many seconds earlier does motion recognise the task than the table alone? Answered on both alignments, pooled over the shared tables, using:
  - **τ_X:** for X ∈ {S, M, F+}, the earliest cutoff at which the Wilson lower bound of pooled accuracy is > pooled chance at that cutoff and at every later cutoff ("stable recognition onset"). S is constant over cutoffs, so τ_S is either the first cutoff or "never".
  - **τ_gain:** the earliest cutoff at which the paired F+ − S CI lower bound is > 0 at that and every later cutoff.
  - **Lead of motion over table:**
    - if τ_S exists: τ_S − τ_F+ seconds;
    - if S is never stably above chance: "the table alone never recognises the task in the window". The lead is then reported as at least (last cutoff − τ_F+), and τ_F+ is stated relative to start_s and to t_on.
  - **Uncertainty:** a 95% bootstrap interval for τ_F+, τ_M and τ_gain (2,000 resamples of clips within table; per-clip predictions fixed; the same definition is recomputed in each resample).
  - **Time to 80%:** the earliest cutoff at which pooled F+ accuracy is ≥ 0.80 and stays there (descriptive).
- **Stability, pre-touch grid (alignment i, c ≤ 0), F+:**
  - the switch rate between consecutive cutoffs;
  - the stable-correct lead time start_s − t*, where t* is the earliest cutoff after which the prediction is correct at every later pre-touch cutoff (median and IQR, plus the share of clips with no stable-correct lead);
  - the same quantity relative to t_on.
- **Interpretation (fixed now):**
  - if accuracy rises towards the touch, the extra observation (the reach) carries task information;
  - if accuracy is already high at −2 s, compare with S at the same cutoff: a high S means scene or object bias, not anticipation;
  - if accuracy rises but predictions switch often, prediction is not yet stable.

## Part B (H3): asking back with calibrated confidence (CPU)

- **Probabilities:** the LOPO probabilities of F+ at p25 and p50 (Part A).
- **Calibration:** a temperature T for each outer fold, fitted on the training persons only. The inner leave-one-person-out probabilities of the two training persons (same C) are used, and T minimises NLL on a log grid of 0.05–20. ECE uses 10 bins and is reported before and after.
- **Policy:** ask back when the calibrated top probability is < τ. τ is the 20th percentile of the training persons' inner calibrated top probabilities (asking rate ≤ 20% on training), applied to the held-out person. When the model asks, the simulated answer is the true label. This is called a "simulated ask-back", not HRI.
- **Baselines:**
  - no asking;
  - random asking at the same rate per fold: 1,000 draws for the mean and 95% interval; the expected value per clip is used for the paired test;
  - for reference, the Stage 12 margin rule (τ = 0.0106) applied to untrained F.
- **H3-a:** pooled accuracy with confidence-based asking − expected accuracy with random asking at the same rate. Paired bootstrap over clips within table; the CI lower bound must be > 0.
- **H3-b:** pooled ECE after calibration < ECE before.
- **Also reported:** accuracy against asking rate (0–50%), in `ask_curve.csv` and `.png`.

## Outputs

`results/stage16_scene_motion/`:
- `accuracy.csv`
- `rules.md`
- `per_person.csv`
- `confusion_p50.csv`
- `a2_cutoffs.csv`
- `curve_start.csv`, `curve_ton.csv`, `curve_seconds.png`
- `onset.csv`
- `stability.csv`
- `ask_curve.csv`, `ask_curve.png`
- `config.json`
- `DEVIATIONS.md`

Embeddings are stored in `/media/dhqg/d3/vlia_outputs/stage16/emb/` (not committed). Plots show no video frames.
