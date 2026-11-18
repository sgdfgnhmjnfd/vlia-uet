# Stage 14b Part B (writer W): deviations and notes

1. **Run order.** Part C ran before Part B, because the four-part task arrived while Part C was already generating. Part B was pre-registered (commit 2c0beff) before any Part C result existed.
2. **"Trained on all k at once".** In each epoch, every EgoIntent (step, tag) pair gets one k drawn at random from {2, 4, 6, 8}. Every LIBERO (demo, p) pair with p in {10, ..., 100} gets one of its 7 sentences, drawn at random. One model per fold serves every k and every p.
3. **Data sources.** NEXT targets and step timestamps come only from the `step_label.json` files of the 27 development video directories. A check confirmed that none of those directories holds a video from outside the development split. The locked test split and its metadata file were never opened (`test_stage14b_writer.py` checks this).
4. **NEXT 5-way.** The NEXT 5-way candidates were built with `build_kway` on the NEXT texts, the same way as for WHY and WHAT. W never sees the candidates; they are used only to score its sentence afterwards.
5. **LIBERO p = 0.** The input at p = 0 is 8 copies of the first frame's embedding.
6. **human_check_w key.** The key to `human_check_w.csv` (which column holds W and which holds the reference) is stored outside the repo, at `/media/dhqg/d3/vlia_outputs/stage14b_writer/human_check_w_key.json`, so the blind check is not spoiled.
7. **F on EgoIntent** is available at every k (2, 4, 6, 8). On LIBERO, F has no p = 75 value (Stage 12 did not run p = 75), so that cell is empty.
8. **Power loss on 2026-11-16 at 19:00.** The machine shut down while `write-libero` was running (folds 0 and 1 written in memory, nothing saved).
   - All 13 checkpoints (8 EgoIntent, 5 LIBERO) were already on disk.
   - After the reboot on 2026-11-17, only `write-libero` and `unseen` were run again, from those checkpoints, with no retraining.
   - Decoding is greedy, so the rerun depends only on the checkpoints, up to GPU floating-point noise.
9. **Unseen task 7, "nearest description = task 7" column.** W was trained without any task-7 demo, and none of its task-7 sentences names the task-7 object pair: object+place = 0 at every p.
   - The non-zero values in that column (53.5% at p = 0, 37.2% at p = 75) come from the scorer, not from W.
   - SigLIP places W's task-0 sentence "I want the basket to hold the alphabet soup and the tomato sauce" nearest to the task-7 description. Task 7 ("alphabet soup and cream cheese box in the basket") shares "alphabet soup … basket" with task 0.
   - At p = 0, 23 of the 43 sentences are this task-0 sentence. The column should not be read as W recognising task 7.
10. **Task 0 on LIBERO (descriptive, scorer unchanged).** W's task-0 accuracy is 0–6% at every p, although 135 of its 198 task-0 sentences name both task-0 objects and the basket (object+place correct). 128 of those 135 are placed nearest to the task-7 description by the pre-registered SigLIP scorer. The scorer was not changed after seeing this. `object_place` is reported beside accuracy for this reason.
