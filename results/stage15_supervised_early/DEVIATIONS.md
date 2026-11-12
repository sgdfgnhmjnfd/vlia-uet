# Stage 15 deviations and notes

1. **Models written in torch.** Neither `PY` nor `VPY` has scikit-learn, and installing packages is not allowed.
   - All four models are torch implementations, as the task allows. The pre-registration (Part 2, written before any number) fixes them: the squared-hinge one-vs-rest SVMs (the RBF one in the primal over kernel coefficients), multinomial logistic regression and the temporal CNN.
   - They are not scikit-learn's `SVC` / `LinearSVC` / `LogisticRegression`. Numbers may differ slightly from scikit-learn's solvers.

2. **Run order.** The chain first extracted agent and wrist features, then ran CV.
   - SigLIP encoding took about 9 minutes per video file, so after 4 of 11 agent files it was restarted in a new order: agent → CV agent → plan → robot → analyze, then wrist → CV wrist → analyze.
   - Extraction is resumable per demo; the 149 finished demos were kept and none was recomputed.
   - No method, grid or rule changed.

3. **Seed 1 pause.** Seed-1 training and the token-matrix chain were paused at 13:12 on 2026-11-12, after `checkpoint_015000` (log step 15,390), so that Stage 15 could run alone on the GPU. All files are kept.

4. **Clips: `tcnn` with no validation rows** (code for later; the LIBERO run is not affected).
   - With leave-one-rep-out on 30 clips, an inner training part can hold one clip per task. The pre-registered 20% stratified validation split is then empty.
   - In that case only, `tcnn` trains all 100 epochs and keeps the last epoch. On LIBERO every task has ≥ 15 training demos per inner fold, so the split is never empty.

5. **Clips: robot references** (code for later).
   - Stage 12 `E2E` has 299 units and no unit (8, 29), while the clip plan uses all 300 Stage 13 units.
   - The clip robot table is therefore compared with `I-only` and `I-wrong` only.

6. **Notes, no deviation.**
   - **Reuse.** The `I-only` reuse rule makes `E2E-sup` equal `I-only` on every reused unit by construction. At p = 25, 298 / 299 units were reused and 1 was rolled out (unit (0, 19), predicted task 1). At p = 50 all 299 were reused. As disclosed in the pre-registration, Stage 12 D1 found that re-running a unit with the same token reproduces its `I-only` outcome in 92.8% of units, not 100%.
   - **Grid edges.**
     - `svm-rbf` chose g = 16 (the largest g) in all 5 folds.
     - `logreg` chose C = 100 (the largest C) in all 5 folds.
     - Many inner-CV accuracies tie at 1.000, and ties went to the smaller C or g as pre-registered.
     - The grids were not changed.
   - **Primary model per outer fold** (by inner CV): `svm-lin`, `svm-lin`, `svm-rbf`, `svm-lin`, `logreg`.
