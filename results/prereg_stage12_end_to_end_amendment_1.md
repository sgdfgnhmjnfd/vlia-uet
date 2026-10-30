# Amendment 1 to the Stage 12 pre-registration (2026-10-30)

Recorded before any Stage 12 scoring or rollout. Only dataset metadata had been read. `results/prereg_stage12_end_to_end.md` (commit 8e65afb) is not edited. That file already carries the same decision under its own "Amendments" section (A1); this file is the authoritative record of it.

**Reason.** Eval task 8 ("put both moka pots on the stove") has only 29 LIBERO-10 demos, so gate M1 as first written (at least 30 demos per task) fails and demo d = 29 does not exist for task 8.

**Decision by Trang:**
- Use 299 units: drop (t = 8, e = 29) because task 8 has only 29 demos. It is dropped from `E2E` and from every comparison, so every "300" in the design reads 299.
- Gate M1 becomes: every task has ≥ 29 demos. The one-to-one mapping requirement stays.
- τ is computed on the 80 calibration demos of the other 9 tasks. Task 8 has no calibration demos.
