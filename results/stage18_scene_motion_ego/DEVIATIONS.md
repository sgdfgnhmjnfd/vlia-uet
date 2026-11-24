# Stage 18 deviations and notes

1. **Label rule (before any accuracy, at STOP A).** The first draft skipped every leading word ending in "-ly", which
   also skipped the verb `apply`. Replaced by a fixed adverb list (label_rule.md). The kept 10-verb set did not change.
2. **Classifier implementation.** scikit-learn is not installed in the `lerobot` env (no pip install allowed); the
   Stage 15/16 torch L-BFGS LogReg was used (as written in the prereg) with class-balanced sample weights added.
3. **C at grid edge** happened in many outer folds (config.json `C_edge_extensions`), extended once as pre-registered.
   On EgoIntent the inner-CV scores are nearly flat across C (all near chance), so the chosen C is close to arbitrary.
4. **EgoIntent prefix seconds.** The 8 cached frames span first to last frame of the step, so "p25" (frames 1-2)
   ends at 1/7 of the step (0.36 s on average), "p50" (frames 1-4) at 3/7 (1.09 s), "p100" at 2.54 s. The labels
   p25/p50 are nominal, as in the prompt. Mean step duration from mp4 headers: 2.51 s (prompt said ~2.56 s).
5. **No person/session id** in EgoIntent metadata (only Ego4D video_uid), so no leave-one-person-out split; the main
   split is scene-grouped (stricter than video-grouped), but the same camera wearer may be in train and test.
6. **Context.** Stage 16 was no longer running (`vlia-lit` is on `claude/stage17-hoi-own`); the only active job was
   the 100k evaluation (baseline_25000). Stage 18 ran on CPU only (2 threads, nice 19), ~50 min total.
7. **Worktree base** is `origin/main` after `git fetch` (c0f04c8).
8. **LIBERO consistency gate passed:** S 91.0 vs Stage 15 primary 88.7 (|d| = 2.4 pts); S+M p25 99.5 vs 99.7.
   Stage 15's p0 primary was SVM-lin in most folds, so a small gap is expected.
