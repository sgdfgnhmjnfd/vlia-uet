# Preregistration v2: freeze, locked test, S1b, adapter dims, LIBERO goal ablation

**Committed before any analysis under this preregistration.** No number of Stages 1–4 below exists yet. The EgoIntent `test` split is not read, decoded, embedded or scored in the session that writes this file. Rules cannot change after numbers are seen; any deviation is reported as a deviation.

Prior results this builds on: `results/ladder_v1/` (nested selection chose `siglip-so400m-patch14-384 | A0 | C2 | T1` in 8/8 folds; N1 FAIL, N2 PASS), `results/knn_cache_v1/` (K1 FAIL), `results/selective_e1_v1/` (S1 PASS).

---

## Rules (verbatim from the task specification)

### Stage 1: freeze + locked-test evaluator (CPU, no test data)
a. Write `results/frozen_pipeline_v1/FROZEN.json` with the ladder winner, selected in 8/8 folds: `encoder_id = google/siglip-so400m-patch14-384`, `aggregator = A0` (mean), `centering = C2` (subtract mean of all frames of strictly earlier steps of the same video; fall back to C1 when none, C1 means = the ones computed on ALL 2503 train+val queries), `template = T1` (`a first-person photo of a person doing: {why}`), `beta = 0` (kNN rejected, K1 FAIL), frame dtype as in the ladder cache, `preregistration_sha256` of `results/prereg_v2.md`, `git_commit`. Also store the comparison pipeline `baseline = siglip-base-patch16-224|A0|C0|T0` (float32).
b. Implement `evaluate_frozen_test(frozen, test_manifest, output_dir)` in a new module `scripts/frozen_eval_lib.py` and fix the import in `scripts/final_frozen_eval.py`. Test set = exactly the 407 samples / 4 UIDs of `results/test_protocol_full_v1/candidate_test_manifest.csv` (excluded UID stays excluded). The function must: decode frames with the same sampler as the ladder cache; extract embeddings for the frozen encoder and the baseline only; apply C2 using earlier steps of the same TEST video (never train/val frames); compute for k in {2,4,6,8}: gallery MRR/Top-1/Top-3 with exact chance (gallery = all 407 test samples), K=5 and K=10 accuracy with `build_kway` candidates built inside the test set, lifts, paired frozen − baseline deltas, video bootstrap over the 4 test videos (B=10000, seed 0, label "exploratory: 4 clusters"), plus the mismatched-video control (every 8th query, k=8, frames from the other scene). Write `summary.csv`, `summary.md`, `per_example.csv`, `config.json` (with input hashes) and refuse to run if `output_dir` exists.
c. Verdicts written into the function (pre-registered):
   - **T1 (frozen pipeline above chance on test):** 5-way lift CI lower bound > 1 at >= 3 of 4 prefixes.
   - **T2 (frozen pipeline beats baseline on test):** paired 5-way delta CI lower bound > 0 at >= 2 of 4 prefixes. With 4 clusters a FAIL is expected to be uninformative; report it as is.
d. Unit-test on a synthetic manifest + fake frames in `/media/dhqg/d3/tmp` (assert the test loader refuses train/val ids, C2 never uses a same-or-later step or another video, chance values match `gallery_chance`). Do NOT execute on real test data.

### Stage 2: selective prediction on the frozen pipeline (CPU, train/val out-of-fold)
Reuse `scripts/run_selective_e1.py` logic on the out-of-fold K=5 scores of the nested-selected pipeline from `results/ladder_v1` (recompute scores from the cached embeddings if they were not saved; assert they reproduce the reported 5-way accuracies exactly at k = 2,4,6,8, else stop this stage). Primary score = top-1 minus top-2 margin. Output `results/selective_frozen_v1/`.
- **S1b PASS** iff the video-bootstrap (B=2000, seed 0) CI lower bound of acc@50% coverage − acc@100% is > 0 at >= 2 of 4 prefixes. Also report AURC vs random rejection and the leakage-free per-fold threshold results at coverage 0.8/0.5/0.2, exactly as in S1.

### Stage 3: Intention Adapter dimension tests (CPU)
Run the existing adapter/prefix/injection tests with intention dims d in {256, 768, 1152} (1152 = so400m). Pass criteria per d: finite loss, non-zero adapter gradient, prefix length +1, intention token right before the state token, present in KV cache at inference, flag-off output identical to original SmolVLA (max abs diff <= 1e-6), checkpoint round-trip. Write `results/adapter_dims_v1/summary.md` (table d × test → PASS/FAIL). This proves implementation correctness only.

### Stage 4: LIBERO goal-information ablation (GPU, after Stage 3)
Question: how much does SmolVLA's success depend on the goal information it receives? This motivates (does not prove) an intention signal.
- Checkpoint: the valid 7-D SmolVLA LIBERO-10 run that gave 24% over 100 episodes (find its path from logs/configs; assert `action` dim 7 and `empty_cameras = 1` in the saved config; if not found or invalid, skip this stage and report).
- Conditions, same policy, same per-episode seeds and initial states across conditions: **C-correct** (original instruction), **C-empty** (empty string), **C-wrong** (instruction of another LIBERO-10 task, fixed derangement task i -> task (i+5) mod 10).
- Ladder: measure seconds per episode on 3 episodes. N = 10 episodes per task per condition (300 total) if ETA <= 8 h; else 5 per task (150); else 3 per task (90). Record the level.
- Verdicts (paired by task + seed; bootstrap over episodes B=10000):
  - **L1 (goal information matters):** success(C-correct) − success(C-empty) CI lower bound > 0.
  - **L2 (wrong goal hurts beyond no goal):** success(C-empty) − success(C-wrong) CI lower bound > 0.
  Report per-task success, Wilson intervals per condition, and videos of 1 success + 1 failure per condition if the env supports recording (on d3).
- Output `results/libero_goal_ablation_v1/`. Never compare with the old invalid 25k runs.

---

## Implementation details fixed now (not specified above)

General: K-way hits are strict (positive > every distractor) as in E1/ladder; invalid K-way rows count as misses. Bootstrap = `run_scene_cv.video_bootstrap` (percentile 2.5/97.5).

Stage 1:
- C1 means for the frozen pipeline: mean so400m frame embedding over all 8 frames of all 2503 train/val queries (ladder cache, float16 → float32), and mean T1 text embedding over the 2503 queries' WHY targets. Saved as `results/frozen_pipeline_v1/c1_means.npz`; its SHA256 is in FROZEN.json. C2 on test, as in the ladder: frames only are context-centred and texts stay raw; a query with no earlier step uses C1 for frames and texts.
- Test step order = integer of the `stepN` suffix of `sample_id`. Video file = `<dataset_root>/<sample_id>.mp4`, the same layout as the train/val manifest. Test rows come from the candidate manifest CSV only.
- Frozen-encoder frame embeddings are cast to float16, as in the ladder cache; baseline frame embeddings stay float32. Text embeddings are float32. Image preprocessing and tokenization are identical to `scripts/extract_ladder_embeddings.py`.
- Control donor: deterministic (crc32 of `sample_id|control`), drawn from test samples of a different scene. If no other scene exists in the test set, the donor is drawn from a different video, and this fallback is recorded.
- MRR and K-way paired deltas use the same per-query values for both pipelines.

Stage 2:
- The out-of-fold scores are recomputed with `run_ladder_v1` functions (per-fold C1 means from the training folds, C2 with fallback). They must reproduce `results/ladder_v1/summary.csv` nested_selected `5way_hits` exactly at every k, else the stage stops.
- Scores, coverage levels, B=2000 bootstrap, fold-quantile thresholds and K=10 secondary analysis are the same code as S1 (`run_selective_e1.py`).

Stage 3:
- Tests run on CPU in fp32 with a randomly initialised SmolVLA (`load_vlm_weights=False`, same architecture as the checkpoint config); seed 0.
- "Flag off identical" compares `VLIAFlowMatching(use_intention_token=False)` with an upstream `VLAFlowMatching` that shares the same weights. Both prefix embeddings and `sample_actions` (same noise) are compared, and must also match when the flag is on with `z_int=None`.
- "Present in KV cache" means the cached key length after the prefix pass equals the prefix length with the intention token (base + 1), and the cached keys differ when `z_int` changes.
- "Checkpoint round-trip" means `state_dict` save → load into a fresh model gives identical `sample_actions` (max abs diff ≤ 1e-6) and identical adapter weights.

Stage 4:
- Episode e of task t (e = 0..N−1) uses LIBERO init state index e and seed 1000 + e in every condition. Success = any success flag during the rollout (lerobot `rollout`).
- The instruction is overridden on the sub-env (`task_description`); nothing else changes. The checkpoint, env settings and rename map are the same as the 24% eval.
- Paired bootstrap resamples the 10 × N (task, episode) pairs with replacement (B=10000, seed 0) on per-pair differences. Wilson intervals are 95%.
- ETA for the ladder = measured seconds per episode × episodes × 3 conditions.
