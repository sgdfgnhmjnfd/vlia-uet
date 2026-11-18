# Stage 14b Part B pre-registration: our own trained model that writes the intention ("W")

Written and committed on 2026-11-16, before W was trained or run. Part C (Qwen prompt variants) was already running on the GPU when this task arrived. Its pre-registration (`prereg_stage14b_generate_variants.md`) was committed before any Part C sentence was generated. No Part C or Part B number had been seen when this file was written.

## W

- **Input.** For each unit there are 8 frames, each a frozen SigLIP so400m image embedding (1152-d, normalised).
  - EgoIntent: the ladder cache `vlia_outputs/ladder/siglip-so400m-patch14-384/frame_emb.npy`.
  - LIBERO: the Stage 15 agent-view feature cache, read with `run_stage15_supervised.py` (taken from `origin/claude/stage15-supervised-early`). The 8 frames are chosen by its `sample8`.
- **Prefix.** With k frames seen, the prefix has k visual tokens; unseen frames are left out.
  - On LIBERO, p is the prefix and Stage 15's `sample8` gives 8 frames over it. At p = 0 these are 8 copies of the first frame, as in Stage 15 and Stage 14.
- **Connector** (trained):
  - frame embedding + a learned position embedding (1152-d) for the frame index 0..7;
  - Linear 1152→960, GELU, Linear 960→960;
  - the result is one input embedding per frame.
- **Text decoder.** The `LlamaModel` text model and `lm_head` of the cached `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (d = 960, 32 blocks). It is loaded with its own tokenizer, and the vision tower is not used.
  - Trained: the connector, the last 4 decoder blocks (28–31) and the final norm. Everything else is frozen.
  - Loading was checked on CPU before this file was written and works, so the from-scratch fallback is not used.
- **Sequence.** `[visual tokens] + tokens(" {tag}") + tokens(" {target}") + <end_of_utterance>`.
  - Tags: `why:`, `what:`, `next:` (EgoIntent) and `goal:` (LIBERO).
  - Loss: cross-entropy on the target tokens and the end token only.
- **Decoding.** Greedy, at most 24 new tokens, cut at `<end_of_utterance>`. The output is stripped of whitespace.
- **Training.**
  - AdamW: lr 1e-3 for the connector and 2e-5 for the decoder blocks and norm; weight decay 0.01; batch 32.
  - At most 30 epochs. Early stopping on the inner validation loss with patience 3; the best epoch is kept.
  - bf16 autocast on the GPU. Seed 0.
  - Inner validation: 10% of the training videos (EgoIntent: by `video_uid`; LIBERO: by demo, stratified by task, Stage 15 `val_split`, seed 0).
  - One model per outer fold.
- **Each epoch.**
  - EgoIntent: every (step, tag) pair gives one example, with k drawn uniformly from {2, 4, 6, 8}. That is how "trained on all four k at once" is carried out.
  - LIBERO: every demo gives one example per p ∈ {10, 25, 50, 75, 100}, with the target sentence drawn at random from the 7 sentences of its task (the description plus the 6 appendix sentences of the task text).
  - Validation uses fixed draws (seed 0).
- **W never receives a candidate list.** Candidates and descriptions are used only by the scorers, after writing.

### Controls

- **(a) blank:** the same trained W, with every visual token input replaced by the mean training frame embedding (the position embeddings are kept).
- **(b) most frequent:** the most frequent training target sentence of the fold, per tag. Ties go to the alphabetically first.
- **(c) nearest training sentence:** 1-NN by cosine of the mean of the seen frame embeddings, among the training units at the same k (or p). It outputs that unit's target sentence; LIBERO uses the task description.

## B1. EgoIntent

- **Data.** The 2,503 development steps of `results/hoi_detection_v1/manifest.jsonl` (27 videos, 2,032 distinct WHY sentences).
  - Targets: WHY = `procedural_intent` (the manifest's `why`), WHAT = `local_intent`, NEXT = `observed_next_step`.
  - NEXT and the timestamps are read only from the `step_label.json` files of the 27 development event directories. Those directories hold only development steps, which is checked: no extra mp4 files.
  - The test directories and `egointent_full_split_v1/test.jsonl` are never opened. A test asserts that every loaded id is in the manifest.
- **Folds.** Scene-held-out, `run_scene_cv.assign_folds(scenes, 8)`, as in F's cross-validation: train on 7 groups and write for the 8th.
- **Seconds.**
  - The 8 frames are spread evenly over [start_time, obs_end_time] of the step (Stage 3 `sample_exact_video_frames`), so frame j (0..7) is at start + j/7·dur.
  - Seen at k: (k−1)/7·dur. Remaining until the end of the step: dur − seen.
  - The step end is `obs_end_time`, the pre-outcome cutoff; the annotation has no later end time.
  - For `next:`: the mean time from the last seen frame to the start of the following step of the same video (steps with no following step are left out of that mean).
- **Scores**, per k ∈ {2, 4, 6, 8} and tag:
  1. **Open-ended 5-way.** The Stage 14 A1 scorer: SigLIP text, plain, strict top-1 among the 5 candidates of `run_scene_cv.build_kway(vids, texts, sid, 5)`. For `next:` the same construction is applied to the NEXT texts. Chance is 20%.
  2. **Same-video MRR** among all distinct sentences of that kind in the same video (Stage 14 `rank_of_true`).
  3. **Token F1 and ROUGE-L (F)**: lower-cased, punctuation removed, whitespace tokens, written in the script.
  4. **Exact verb / exact object**, by a rule with no tagger:
     - the verb is the first token, after dropping a leading "i want to" or "to";
     - the object is the last token of the span after the verb that ends at the first word of the stop list {for, to, with, in, into, on, onto, from, of, before, after, and, or, by, at, using, while, so, that, then, as, through, under, over, near, inside}.
     - If that span is empty, the object comes from the next span, i.e. after the stop word. When the stop word is "to", the first token after it is skipped as a verb: "prepare to mask painting edges" gives "edges".
     - 20 development WHY sentences (the first 20 by sample id) are hand-labelled in `test_stage14b_writer.py`, and the rule must get all 20 right.
  - `what:` is reported descriptively only.
- **Rules**: scene-clustered bootstrap (clusters = scene), B = 10,000, seed 0, steps with valid candidates as in Stage 14 A1.
  - **W-1.** 5-way for `why:` at k = 8: CI lower bound > 20%.
  - **W-2.** W − blank (`why:`, k = 8, 5-way, paired): CI lower bound > 0.
  - **W-3.** W − nearest training sentence (`why:`, k = 8, 5-way, paired): CI lower bound > 0.
  - **Descriptive:**
    - W − Qwen (Stage 14 A1 sentences at k = 2, 4, 8 on the same steps, paired);
    - W next to F (Stage 14 table, same steps);
    - the same scores for `next:` (Qwen and F did not write or choose NEXT, so W stands alone there);
    - the curve over k.
- **`human_check_w.csv`.** 60 steps drawn with seed 0, stratified by scene group (7 or 8 per group). The file gives the true WHY and sentences A and B: W and Qwen at k = 8, in an order shuffled with seed 0. Grading columns are left empty.
  - The key (which sentence is W) is stored outside the repository, in `vlia_outputs/stage14b_writer/human_check_w_key.json`, and is committed only after grading.

## B2. LIBERO-10

- **Data.** The 379 demonstrations, agent view, the Stage 15 outer folds (`libero_outer`, 5 folds, seed 0).
- **Seconds.** Duration = time of the last frame of the 4 fps grid (10 fps dataset).
  - Seen at p: p%·duration. Remaining: (1 − p%)·duration.
  - Reported overall and per task. `data_summary.md` gives the frame rate and the mean, minimum and maximum duration per task.
- **Scores**, per p ∈ {0, 10, 25, 50, 75, 100}:
  1. The Stage 14 A3 scorer (nearest of the 10 descriptions, plain SigLIP, strict top-1; chance 10%).
  2. Within-scene, tasks 0 against 1.
  3. The share of written sentences not character-identical to any of the 70 training sentences.
  4. **Object-and-place check.** All keyword groups of the true task appear in the lower-cased sentence; a group is met when any of its strings appears:
     - 0: {alphabet soup, soup} {tomato sauce} {basket}
     - 1: {cream cheese} {butter} {basket}
     - 2: {moka pot, coffee pot} {stove}
     - 3: {bowl} {drawer}
     - 4: {white mug, white cup} {yellow and white mug, yellow and white cup} {left} {right} {plate}
     - 5: {book} {caddy}
     - 6: {white mug, white cup} {pudding} {plate} {right}
     - 7: {alphabet soup, soup} {cream cheese} {basket}
     - 8: {moka pots, coffee pots} {stove}
     - 9: {yellow and white mug, yellow and white cup} {microwave}
- **Unseen task (descriptive).** One extra W is trained on all demos of the 9 tasks other than 7 (10% of them as validation), with the same recipe.
  - It writes for all 43 task-7 demos at every p.
  - Reported: the object-and-place check for task 7, and the nearest of all 10 descriptions.
  - A 9-label classifier scores 0% on task 7 by construction.
- **Rules**, bootstrap over demos (B = 10,000, seed 0):
  - **W-4.** W accuracy at p = 25 on the 379 demos: CI lower bound > 50%.
  - **W-5.** W − the best Qwen variant of Part C at p = 25, paired on the 299 units: CI lower bound > 0. "Best" is the Part C variant (G0, chosen G or O1) with the highest p = 25 test accuracy. O1 knows the object names and is used only if it is the highest; the comparison with the chosen G is also given.
  - **Descriptive:** W next to C (Stage 15, 99.7% at p = 25), F (69.9%) and Qwen.
- **Tokens.** `w_tokens_p25.json` and `w_tokens_p50.json` in the `e2e_gen_tokens.json` format: `{"tokens": {"E2E-W-p25": {"t,e": sentence}}, "new": {...}}` for the 299 units, each from the fold that did not see that demo. No rollouts.

## Outputs

`results/stage14b_writer/`, as listed in the task: `data_summary.md`, `ego_writer.csv`/`.md`, `libero_writer.csv`/`.md`, `unseen_task7.md`, `sentences_ego_k8.csv`, `sentences_libero.csv`, `human_check_w.csv`, `w_tokens_p25.json`, `w_tokens_p50.json`, `config.json`, `DEVIATIONS.md`.

Code: `scripts/run_stage14b_writer.py` and `scripts/test_stage14b_writer.py`.
