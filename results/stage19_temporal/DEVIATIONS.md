# Stage 19 deviations and notes

1. **Method B only (Trang, STOP A).** Method A (temporal conv on SigLIP frames) not run unless time remains.
2. **Order changed (Trang):** speed test on 5 steps -> own videos p25/50/100 -> EgoIntent p50 -> EgoIntent p100 only
   if still inside the 4-hour budget (started 2026-11-24 16:39).
3. **EgoIntent p50 first; p100 only if time remains** (Trang). If p100 is missing, the EgoIntent M/S+M p100 cells
   are "-".
4. **Model download** (Trang ok): `MCG-NJU/videomae-base-finetuned-ssv2` into HF_HOME=/media/dhqg/d3/hf_cache,
   HF_HUB_OFFLINE=0 only for that command. The repo has no safetensors; `pytorch_model.bin` (330 MB) was fetched.
5. **transformers 5.5 loading bug:** the checkpoint stores `q_bias` / `v_bias` (VideoMAE qkv_bias with key bias 0);
   transformers reports them UNEXPECTED and leaves `query/key/value.bias` randomly initialised (MISSING). Fixed by
   copying them by hand (key bias = 0). Checked before any accuracy: SSv2 top-1 on own clips "Putting [something]
   into [something]" (task: put items in the basket), EgoIntent "hold ..." -> "Holding [something]". The speed test
   ran before the fix (same compute cost).
6. **EgoIntent window:** "p50" is the first 3/7 of the step (up to the 4th of the 8 Stage 18 frame positions), so
   that B observes exactly what Stage 18 observed; p100 is the whole step.
7. **Own videos, extraction** also covered the 9 table-E clips (single-task table, not evaluated), as in Stage 16.
8. **CI types in bang_so_sanh.md:** Stage 16 rows are Stage 16's Wilson intervals (copied); Stage 19 own-video rows
   use the pre-registered bootstrap of clips within table. R4 uses a paired bootstrap on the same clips.
9. **Stage 16 F+ p50 recomputed** per clip from the Stage 16 SigLIP cache with Stage 16's unweighted LogReg
   (for the paired R4): 0.741, identical to the committed Stage 16 number.
10. **Shuffled-label check, own videos:** 38.3% vs pooled chance 44.4% (within-table LOPO with permuted labels can
    fall below chance; anti-learning on small tables). Near chance, not above it: no sign of leakage.
11. **Post-hoc check (NOT pre-registered; asked by Trang after seeing the own-video results).** Own videos, p100:
    B (VideoMAE-SSv2) S+M 91.4% vs Stage 16 F+ 72.8% (recomputed per clip from the Stage 16 SigLIP cache with the
    Stage 16 unweighted LogReg; identical to the committed 0.728). Paired on the same 81 clips, bootstrap of clips
    within table (10,000, seed 0): **+18.5 points [+9.9, +28.4]**. Clips: 57 both right, 17 only B right, 2 only
    Stage 16 right, 5 both wrong. Command `stage19_temporal.py posthoc_p100`; output
    /media/dhqg/d3/vlia_outputs/stage19/posthoc_p100.json. Because p100 was chosen after seeing the table, this is
    descriptive evidence, not a confirmatory test; the pre-registered R4 (p50) is +1.2 [-6.2, +8.6], FAIL.
12. **Waiter for EgoIntent p100.** Two first attempts to chain p100 after p50 matched their own command line in
    `pgrep` and would have waited forever; both were stopped (only Claude's own waiter shells) and replaced by a
    wait on the p50 chain PID. No running job of anyone else was touched.
13. **Kiểm phụ sau khi đã nhìn dữ liệu (post-hoc, NOT pre-registered; asked by Trang).** Own videos, p25, same
    method as item 11 (paired on the same 81 clips, bootstrap of clips within table, 10,000, seed 0; Stage 16
    recomputed per clip from its SigLIP cache with its unweighted LogReg, both identical to the committed numbers):
    - S+M: B (VideoMAE-SSv2) 79.0% vs Stage 16 F+ 71.6%: **+7.4 points [-2.5, +17.3]** (CI contains 0).
      Clips: 52 both right, 12 only B right, 6 only Stage 16 right, 11 both wrong.
    - M: B 77.8% vs Stage 16 M 75.3%: **+2.5 points [-7.4, +12.3]** (CI contains 0).
      Clips: 53 both right, 10 only B right, 8 only Stage 16 right, 10 both wrong.
    Command `stage19_temporal.py posthoc_p25`; output /media/dhqg/d3/vlia_outputs/stage19/posthoc_p25.json.
    Descriptive only.
14. **EgoIntent p100 was run** (Trang ok; extraction 18:16-18:53). The rules are at p50; p100 cells are descriptive:
    B M p100 17.1 [14.4, 20.8], S+M p100 17.8 [14.0, 22.5] (balanced accuracy, chance 10), vs Stage 18 S+M p100
    12.5 [9.9, 15.5]. No paired test computed for p100.
15. **Method A not run** (B finished at 18:56, 2 h 16 min into the 4-hour budget; stopped at STOP B to ask Trang).
16. **C at grid edge** often on EgoIntent (mostly 0.01 -> 0.001), extended once as pre-registered; listed in
    /media/dhqg/d3/vlia_outputs/stage19/eval_ego.json (`ext`).
