# Pre-registration: Stage 15, supervised early intention recognition (2026-11-12)

Committed and pushed before any feature extraction, training or test score. Part 1 is copied verbatim from the task. Part 2 fixes the implementation details the task leaves open; it is also written before any number is seen. Nothing below will be tuned on results.

## Part 1: design (verbatim from the task)

### Problem

- **Input:** the first p% of a video, p ∈ {0 (first frame only), 10, 25, 50, 75, 100}.
- **Output:** the task label, one of K known tasks (K = 10 for LIBERO-10).
- **Train and test:** the model is trained on labelled training videos, and every reported accuracy is on videos that the model, and every choice of its settings, never saw.

### Features

1. Decode each demo at 4 frames per second, keeping the last frame. Store the SigLIP embedding of every decoded frame, with its time, under `/media/dhqg/d3/vlia_outputs/stage15/feat/`.
2. For a prefix p, take the frames whose time is ≤ p × the demo duration. Sample 8 of them evenly, always including the last one.
3. At p = 0, use the first frame repeated.

### Models (all trained only on training folds)

| id | model | input |
|---|---|---|
| `svm-lin` | linear SVM (C chosen by inner CV) | mean of the 8 frame embeddings |
| `svm-rbf` | RBF SVM (C and γ chosen by inner CV) | mean of the 8 frame embeddings |
| `logreg` | multinomial logistic regression (L2 chosen by inner CV) | mean of the 8 frame embeddings |
| `tcnn` | temporal CNN: 2 × Conv1d over time (kernel 3, 256 channels, ReLU, dropout 0.3), time max-pool, linear head; AdamW, 100 epochs, early stopping on an inner validation split; 3 seeds, averaged | the 8 frame embeddings in order |

- **Training prefixes.** Every model is trained on prefixes drawn from the training videos at all p ∈ {10, 25, 50, 75, 100}: one sample per (video, p). One model therefore serves every prefix, as a robot would need.
- **Primary model.** It is chosen by **inner-CV** accuracy averaged over p = 25, 50, 75 and 100, never by test accuracy. Record the choice in each outer fold.

### Splits

- **Outer 5-fold cross-validation over demos, stratified by task** (seed 0). Each demo is tested exactly once, by a model that did not see it.
- Inner 4-fold CV inside each training part chooses the hyper-parameters and the primary model.

### Pre-registered rules (primary model, agent view)

- **R1 (full video).** Accuracy at p = 100 has 95% CI lower bound > 90%.
- **R2 (half video).** Accuracy at p = 50 has CI lower bound > 50%.
- **R3 (not only the scene).** For the two tasks that share a scene (0 and 1, `LIVING_ROOM_SCENE2`), 2-way accuracy, defined as P(true task scores higher than the other one) on demos of those two tasks, at p = 50 has CI lower bound > 50%.
- **CIs.** Bootstrap over demos, stratified by task (B = 10,000, seed 0).

### Descriptive, no pass/fail

- the accuracy table for every model × every p, for the agent view and the wrist view, with macro-F1;
- the confusion matrix at p = 25 and p = 50;
- the first-frame-only classifier (p = 0), which shows how much the scene alone gives away;
- references: the zero-shot F from Stage 12, and chance (10%).

### Robot (closed loop with the supervised label)

- For each unit (task t, episode e = 0..29), the query is demo e of task t at p = 25 (primary) and p = 50.
- The label comes from the outer-fold model that did not train on that demo.
- The token is the T1 table row of the predicted task. VLIA runs with an empty instruction, `init_state_id = e` and seed 1000 + e (condition **`E2E-sup`**).
- **Units whose prediction is the true task** reuse the existing `I-only` episode, because the token, init state and seed are identical. Only the remaining units get new rollouts. Record how many.
- **Rule R4.** success(`E2E-sup`, p = 25) − success(`I-wrong`) has CI lower bound > 0 (`paired_bootstrap`).
- **Descriptive:** `E2E-sup` against `I-only` (the upper bound), Stage 12 `E2E` (zero-shot F) and `I-wrong`, at p = 25 and p = 50.

### Later: Trang's own videos (code now, run when they exist)

Use the same pipeline on her clips, with K = 10 tasks and the same 5 tables.

- **Prefix.** p is measured on the action part, from `start_s` to `end_s` of `labels.csv`. p = 0 is the idle part only.
- **Split.** Leave-one-repetition-out over reps, so that every rep of every task is tested by a model trained on the other reps. If several people exist, also leave one person out.
- **Rules.** R1–R3 apply. R3 becomes the **within-scene accuracy** of Stage 13: the top-1 among the tasks of the clip's own table, against its chance level.
- **Robot.** As above, with the clip → unit mapping from the Stage 13 plan.
- **Cross-domain (descriptive).** Train on LIBERO demos and test on her clips.

## Part 2: implementation details fixed before any result

### Data

- **Demos.** All LIBERO-10 demos of `lerobot/libero` (`task_index` < 10), mapped to eval ids with `map_tasks` (Stage 12). The demo index d of task t is the rank of its episode index, as in Stage 12. Stage 12 found 33, 49, 41, 35, 38, 41, 36, 43, 29 and 34 demos (379 in all).
- **Orientation.** No rotation, as decided in Stage 12 (`rotate180 = false`).
- **Times.** The dataset runs at 10 fps. A decoded frame at time s is dataset frame `round(10 s)`, for s = 0, 0.25, 0.5, ... The last dataset frame (L − 1) is always added. The demo duration is the time of that last frame, (L − 1) / 10.
- **Sampling 8 of n frames.** Use positions `round(linspace(0, n − 1, 8))` of the n prefix frames, so repeats occur when n < 8. A prefix always holds at least the first frame.
- **Mean input.** The mean of the 8 unit frame embeddings, L2-normalised again.

### Models

The lab environments have no scikit-learn (`PY` and `VPY`), and installing packages is not allowed. As the task allows, all four models are written in torch (CPU, float64 for the three kernel and linear models).

- **`svm-lin`.**
  - One-vs-rest linear SVM with squared hinge loss and bias. This is the default loss of scikit-learn's `LinearSVC`.
  - Objective: C · Σ squared hinge + ½‖w‖². Solved with L-BFGS.
  - Grid: C ∈ {0.1, 1, 10, 100, 1000}.
- **`svm-rbf`.**
  - One-vs-rest kernel SVM with squared hinge loss, solved in the primal over kernel coefficients (f(x) = Σᵢ αᵢ K(xᵢ, x) + b; objective C · Σ squared hinge + ½ αᵀKα; L-BFGS).
  - K(x, y) = exp(−γ‖x − y‖²), with γ = g / (D · var(X_train)) (scikit-learn's `gamma="scale"` times g).
  - Grid: C ∈ {0.1, 1, 10, 100, 1000}, g ∈ {0.25, 1, 4, 16}.
- **`logreg`.**
  - Multinomial (softmax) logistic regression with bias. Objective: C · Σ cross-entropy + ½‖W‖². L-BFGS.
  - Grid: C ∈ {0.01, 0.1, 1, 10, 100}.
- **`tcnn`.**
  - Architecture as in Part 1, with input of shape 8 × 1152.
  - Training: AdamW (lr 1e-3, weight decay 1e-2), batch 64, at most 100 epochs.
  - Early stopping: patience 10 on validation accuracy (ties broken by lower validation loss). The weights of the best epoch are kept.
  - Validation split: 20% of the training demos, stratified by task, with all prefixes of a demo on the same side.
  - Seeds 0, 1 and 2. Their softmax outputs are averaged.
  - No hyper-parameter grid.
- **Scores.** `svm-lin` and `svm-rbf` use the one-vs-rest decision values; `logreg` and `tcnn` use the class probabilities. The prediction is the arg-max. A tie for the top counts as wrong (strict top-1, as in Stage 12).
- **Grid ties.** The grid point with the best inner-CV accuracy wins. Ties go to the smaller C, then the smaller g.

### Splits

- **Grouping.** Every split is over demos, so all prefixes of one demo are always on the same side.
- **Outer folds.** Within each task, shuffle the demos (`numpy.random.default_rng(0)`) and deal them to folds 0–4 in turn.
- **Inner folds.** Same procedure with 4 folds and seed 1 + outer fold index, over the training part only.
- **Inner-CV score of a model.** Accuracy over all inner-held-out (demo, p) for p ∈ {25, 50, 75, 100}, using that model's best grid point.
- **Primary model per outer fold.** The model with the highest inner-CV score. Ties are broken in the order `logreg`, `svm-lin`, `svm-rbf`, `tcnn`.
- **Refit.** After inner CV, each model is refit on the whole training part with its chosen settings. For `tcnn` this means a new 20% validation split.
- **Primary accuracy.** Each test demo is scored by the primary model of its own outer fold, and the out-of-fold predictions are pooled.

### Rules and statistics

- **R1 and R2.** Pooled out-of-fold accuracy of the primary model on all demos, agent view.
- **R3.**
  - The demos of tasks 0 and 1 at p = 50, primary model.
  - A demo counts as correct when its true task's score is strictly greater than the other task's score.
  - CI by bootstrap over those demos, stratified by task.
- **Bootstrap.** Within each task, resample its demos with replacement (B = 10,000, `default_rng(0)`). Report the percentile 2.5–97.5 interval.
- **Macro-F1.** The mean over the 10 tasks of F1, where a task never predicted gets F1 = 0.
- **Zero-shot reference.** Stage 12 `C2_T1` recognition accuracy at each common p (demos 0..29, `results/stage12_end_to_end/recognition.csv`). It is shown as given and not re-computed.

### Robot

- **Units.** The 299 Stage 12 units, under Stage 12 amendment 1: (t, e) for e = 0..29, without (8, 29), since task 8 has 29 demos.
- **Labels.** The out-of-fold primary-model prediction for demo e of task t, at p = 25 and at p = 50.
- **Reuse.** When the prediction equals t, the unit's success is read from `$OUT/eval/episodes/I-only_t{t}_e{e}.json`. Before reuse, check that this file's `z_int_task`, `seed` and `init_state_id` match the unit.
- **New rollouts.** Every other unit is rolled out with `scripts/libero_goal_ablation.py --conditions E2E-sup-p25` (then `E2E-sup-p50`) into `$OUT/eval`. Files are `E2E-sup-p{p}_t{t}_e{e}.json`, on the VLIA seed-0 checkpoint.
- **Known limit, disclosed now.** Stage 12 D1 found that re-running a unit with the identical token reproduced its `I-only` outcome in 92.8% of units, not 100% (simulator non-determinism). Reuse is used as designed. The descriptive comparison with `I-only` is therefore exact on reused units by construction.
- **R4.** `G.paired_bootstrap` (B = 10,000, seed 0) over the 299 units of success(`E2E-sup`, p = 25) − success(`I-wrong`).

### Trang's clips (`human` subcommand)

- **Frames.** Decoded at 4 fps with the Stage 13 decoder (`decode`) and its rotation fix (`--rotate`, gate O1). The last frame is always kept.
- **Prefixes.**
  - p > 0: frames with time ≤ `start_s` + p/100 · (`end_s` − `start_s`).
  - p = 0: frames with time ≤ `start_s` (the idle part).
  - 8 frames are sampled as for LIBERO.
- **Training prefixes.** p ∈ {10, 25, 50, 75, 100}, as for LIBERO.
- **Splits.**
  - Outer: leave one rep out (reps 1, 2, 3), primary person `p1`.
  - Inner: leave one of the remaining reps out.
  - Otherwise the same models, grids and choice rule as for LIBERO.
  - If more than one person exists, also train on all clips of the other persons and test on each person (descriptive).
- **R3 for clips.**
  - Within-scene accuracy at p = 50: the strict top-1 among the tasks of the clip's scene table (`SCENE_TASKS` of Stage 13), on the 27 clips whose scene has more than one task.
  - Its chance level is 12/27, as in Stage 13 DEVIATIONS item 2.
  - The rule passes if the CI lower bound is > 12/27.
- **Robot.** Unit (t, e) uses clip `t{t}_r{e % 3 + 1}` (Stage 13 `rep_of`), at p = 25 and p = 50. Reuse follows the same rule as above. Condition names: `E2E-sup-human-p{p}`.
- **Cross-domain (descriptive).** Each of the four models is refit on all LIBERO demos (agent view), with the most frequent inner-CV choice of the LIBERO outer folds. It is then tested on every clip at every p.

## Gates

- **M1.** The task mapping is one-to-one and every task has ≥ 29 demos (as Stage 12 amendment 1).
- **F1.**
  - Every demo has a feature file whose frame times start at 0, increase strictly, and end at (L − 1)/10.
  - Every embedding has unit norm (|‖x‖ − 1| < 1e-4).
- **Self-test.** `run_stage15_supervised.py selftest` and `scripts/test_stage15.py` pass, together with `test_stage11.py`, `test_stage12.py`, `test_stage13.py` and `test_token_matrix.py`.
- **Leak check (L0).** In every outer fold, the training and test demo sets are disjoint. Their union is all demos, and each demo is tested exactly once. The inner folds use training demos only.
- **Smoke.** 1 `E2E-sup-p25` episode with `--probe 1` into a temporary directory. Its `z_int_task` must equal the planned token of unit (0, 0).
