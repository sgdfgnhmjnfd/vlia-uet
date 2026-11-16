# Stage 14 deviations and notes

1. **A3 source.** The updated pause prompt that holds the full A3 text was not on the lab machine. A3 was copied verbatim from the bullet list of the amendment of 2026-11-12, as Part 1b of the pre-registration states. Any A3 detail that is only in the missing prompt is not covered.

2. **Scope.** Only part A ran (A1, A2, A3). Part B (the commit rules and GB1) is fixed in the pre-registration for later and was not run.

3. **Blank control generated once.** The grey-image control (448×448) was generated once per prompt, once for LIBERO and once for EgoIntent. Greedy decoding on a fixed input is deterministic, so one sentence stands for every unit.

4. **Sentence cut.** `clean_sentence` cuts at the first `.`, `!` or `?` that is followed by a space or the end of the text. Closing quotes may sit between the mark and the space.
   - The selftest found the quote case before any sentence was generated. The code was committed at 17:39 on 2026-11-12, and generation started at 17:41.
   - Re-applying the committed `clean_sentence` to the stored raw outputs gives the stored sentence for all 1,794 LIBERO files.

5. **Tokens are computed in `plan`.** The free-sentence tokens (T1 template, SigLIP) are computed once in `plan` and stored in `gen_embeddings.npz` and `e2e_gen_tokens.json`. They are not encoded inside the rollout.
   - Gate E1 checks that the stored table matches on-the-fly encoding: max abs diff 3.5e-07, PASS.
   - Every episode file is checked against its planned token in `analyze`.

6. **p = 75 frames.** Stage 12 had no p = 75 frames, so `gen-libero` extracted them into `vlia_outputs/stage12/frames/t{t}_d{d}_p75.npy`. Only new files were written, and no Stage 12 file was changed.

7. **Snap condition pending.** The snap-to-nearest condition needs `matrix_seed0.npz` from the token-matrix job. That job is paused, as the amendment requires, so snap is reported as pending.

8. **Interruptions** (no rule or method changed):
   - **Pause by the user:** 2026-11-12 at 17:51, during `gen-libero` (488 of 1,794 sentences done). Resumed on 2026-11-13 at 10:49.
   - **Power loss:** about 15:59 on 2026-11-13, during `gen-ego` (about 2,260 of 2,503 steps done). The three files of the step being written (`outdoor/farm/pick_fruits/step93`, k = 8, 2, 4) were empty (0 bytes).
     - They were moved, not deleted, to `vlia_outputs/stage14/quarantine_powerloss_20261113/`, and that step was generated again.
     - Every other stored JSON (9,457 files) was checked and loads.
   - The chain resumed on 2026-11-16 at 15:06. Finished steps were skipped, and finished rollouts were not re-run.

9. **Gate M1 log line.** The chain's status line shows the nearest-description matcher result, 18/20 PASS. `gate_m1.csv` holds the per-phrasing rows. The one-to-one mapping check in the same log also passed.

10. **Seed 1 and the token matrix stayed paused** for the whole run, so Stage 14 had the GPU alone.

11. **Within-scene 0 vs 1 is 50.0% with a zero-width CI at every p** (note, not a bug).
    - At every p, every sentence for task 0 and for task 1 is closer to the task-1 description than to the task-0 description. Task 0 is right in 0/30 demos and task 1 in 30/30.
    - The stratified bootstrap resamples within each task, so both per-task means are constant and the interval collapses to 50%.
    - The two-way choice is therefore at chance level. The matcher always picks the same side.
