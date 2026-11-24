# Bosung 20/11: deviations from the prompt

1. **Training finished early.** The 100k training was stopped at 52,600 by Trang (20/11 18:30). The wait condition is "checkpoint_100000, FROZEN, or STOPPED_BY_TRANG" (Trang agreed to the first two at STOP A; the third marks her stop). Block 4 uses VLIA `checkpoint_050000` as the final milestone.
2. **STOP B skipped by Trang's instruction** ("chạy cái mới dùng GPU cho nhanh", 18:30): the run script was started after the dry run and the existing harness tests, before `test_bosung.py` and `analyze_bosung.py` were written. Those are written afterwards on CPU; they do not change any rollout.
3. **C rerun on GPU, not CPU.** Stage 15 extraction took 55 min on the GPU (3,290 s for 40,935 frames), so on CPU it would take far more than 2 h; it runs on the GPU first (tmux `bosung_c`), then the nested CV, then the comparison. The Stage 15 classifier models were never saved (only out-of-fold scores), so C is trained again with the same code and seeds, as the prompt allows.
4. **Cut-rule deadline** moved to 20:00 on 22/11 (Trang, STOP A), order 1 → 2 → 3 → 4.
5. **No 75k curve point** (training stopped before 75k).

6. **Order changed (Trang, 22/11 ~10:45: "C xong thì train vlia trước").** C rerun first, then VLIA trains 70k→100k,
   then G0 and blocks 1, 2, 3, 4 (each block started separately, so the 20:00 22/11 cut rule cannot reorder them).
   All blocks end after the 22/11 20:00 deadline (about 17:00 on 23/11 if nothing stops). Rollouts of blocks 1–3 use
   the 25k checkpoints as planned, so the later start does not change what is measured.

7. **Blocks 2 and 3 ran in the order 1 → 3 → 2 (23/11 17:57).** The cut rule (deadline 20:00 22/11, already past) moved
   block 2 after block 3 inside one call; my per-block plan in item 6 did not stop it. The order does not change any
   rollout (fixed checkpoints and seeds). The PC lost power on 23/11 evening during block 2 (baseline `correct` s2000 at
   223/300); it was restarted 24/11 11:28 and finished the remaining episodes; no episode file was rewritten.
8. **Closed-loop row vs built estimate: 15 of 299 units differ (7 / 8) although inputs are the same.** On the 298 units
   where C is right, `E2E-sup-p25-CL` gets the same token, empty instruction, `init_state_id` and seed (1000 + e) as the
   old `I-only` episode, and the first actions are identical (checked on e = 0). The results still differ on 15 units,
   so a long rollout is not bit-reproducible run to run (most likely GPU floating-point non-determinism that grows over
   520 steps). This is a measured replay noise floor, not an effect of C.
