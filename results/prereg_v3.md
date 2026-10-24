# Preregistration v3: aggregation/fusion/adapter/MLLM ladder + LIBERO intention-only

**Committed before any analysis under this preregistration.** No Stage 1–5 number exists yet. The EgoIntent locked test has been used once (`results/locked_test_final_v1/`) and is never read, decoded, embedded or scored again. Rules cannot change after numbers are seen; any deviation is reported as a deviation.

Facts fixed before writing this file (no new result):
- `Qwen/Qwen2.5-VL-7B-Instruct` safetensors total 16,584,414,560 bytes in bf16, which exceeds the 16 GB GPU (RTX 2000 Ada, 16,380 MiB) before activations. Stage 3 therefore uses `Qwen/Qwen2.5-VL-3B-Instruct`.
- No own-dataset annotation exists (only `docs/own_dataset_example_FAKE.csv`), so Stage 5 is skipped.
- The 7-D baseline checkpoint was produced by `scripts/train_matched_oracle_LIBERO7D_FIXED.py --condition baseline` (protocol: `lerobot/libero` canonical 88302 frames / 335 episodes, pretrained `lerobot/smolvla_base`, 16 VLM layers, 512×512, chunk 50, seed 0, 25000 steps, batch 32, AdamW lr 1e-4, wd 1e-4, linear warmup 1000 + cosine decay to 2.5e-6, semantic action padding). Its 25k-step history reports `elapsed_s` = 46129 s.

---

## Rules (verbatim from the task specification)

Reference pipeline everywhere in Track A: **F = `siglip-so400m-patch14-384 | A0 | C2 | T1`** (the frozen config). Its out-of-fold 5-way accuracies must be reproduced exactly (853/874/907/969 hits at k = 2,4,6,8) before any Track A stage runs; otherwise stop Track A.

Common nested-selection rule (Track A): in each outer fold choose the grid value by mean K=5 accuracy over k ∈ {2,4,6,8} on training-fold queries only (inner leave-one-scene-out where a parameter is fitted), ties to the lowest grid index; apply to the held-out fold; pool. Report selected values per fold. Paired deltas vs F with video bootstrap B=10000, seed 0, for 5-way, 10-way and gallery MRR. **"Improves on F"** = 5-way delta CI lower bound > 0 at >= 2 of 4 prefixes AND gallery-MRR delta CI upper bound > 0 at all 4 prefixes (i.e. MRR not clearly worse).

### Stage 1 (CPU, cached embeddings): aggregation + context fusion
- Aggregators (frames 1..k only): A0 mean (F); A3 last frame only (frame k); A4 exponential recency weights w_j ∝ exp(−(k−j)/h), h ∈ {1, 2, 4}.
- Context fusion: score s = cos_C0 + λ · cos_C2 (both with T1, so400m), λ ∈ {0, 0.25, 0.5, 1, 2, ∞}; λ = ∞ means F.
- Joint grid = aggregator × λ (5 × 6 = 30 configs), nested selection as above.
- **V1:** report "improves on F" PASS/FAIL. Output `results/v3_stage1_aggfusion/` (summary, per-fold selection, paired deltas, INTERPRETATION.md facts only).

### Stage 2 (GPU light): hard-negative residual adapter on frozen so400m embeddings
Image side: z' = norm(z + W z), W ∈ R^{1152×1152} zero-initialized (identity at start); text side frozen. Input = the best Stage-1 aggregation of each fold if V1 PASS, else F's.
- Loss per query: InfoNCE over {positive WHY} ∪ {all other distinct WHY texts of the same video} (hard negatives) ∪ 64 random in-batch WHY texts from other videos, τ = 0.05.
- AdamW lr 1e-3, weight decay grid {1e-2, 1e-1, 1}, epochs grid {5, 20}; select by inner leave-one-scene-out on training scenes; 3 seeds; never touch held-out-fold labels (assert).
- Also report the same adapter trained without same-video negatives (random negatives only) as a descriptive control.
- **V2:** "improves on F" PASS/FAIL. Output `results/v3_stage2_hn_adapter/`.

### Stage 3 (GPU): MLLM multiple-choice baseline + fusion
Use `Qwen/Qwen2.5-VL-7B-Instruct` in bf16 if it fits in GPU memory with 8 frames at 448 px; else `Qwen/Qwen2.5-VL-3B-Instruct`; record which. No fine-tuning, no prompt search.
- Prompt: the k prefix frames as one video/multi-image message + "These are the first frames of a first-person video. What is the person's goal in this step? Options: A. … E. … Answer with one letter." Candidates = the canonical `build_kway` K=5 set.
- Score = log-prob of each option letter as the first answer token. Run each query with 2 option orders (canonical and reversed) and average per-candidate log-probs to cancel position bias. k ∈ {2, 8}, all 2503 queries; measure throughput on 30 queries, if ETA > 8 h use every 3rd query by sorted sample_id for both k and record it.
- Report 5-way accuracy, lift CI, paired delta vs F on identical queries, and the mismatched-video control (every 8th query, k=8, frames from another scene).
- Fusion (secondary, nested): z-scored score s = s_F + γ · s_MLLM, γ ∈ {0, 0.25, 0.5, 1, 2}, chosen per outer fold on training-fold queries.
- **V3:** MLLM 5-way lift CI lower bound > 1 at both k (above chance). **V4:** fusion "improves on F" on the 5-way part of the rule at both k. Output `results/v3_stage3_mllm/`.

### Stage 4 (GPU, start only after Stage 3): LIBERO — can the intention token carry the goal?
- Model: VLIA SmolVLA with `use_intention_token = True`, `intention_dim = 1152`, initialised from the same base as the 7-D baseline checkpoint, identical training config (dataset, 7-D actions, `empty_cameras = 1`, steps, batch, lr, seed 0). During training and evaluation the **language string is empty** and z_int = L2-normalised so400m text embedding of the T1-templated task instruction (cache per task; 10 vectors). The adapter is trained jointly with the parts the baseline trained.
- Compute ladder: measure seconds/step on 200 steps. If 25k steps (matched) ETA <= 40 h, train 25k. Else STOP this stage after the probe and report the ETA (the user decides); do not train a shorter unmatched run.
- Evaluate with the exact `libero_goal_ablation_v1` protocol (10 tasks × 10 episodes, init state e, seed 1000+e) in two conditions: **I-only** (empty language + correct-task z_int) and **I-wrong** (empty language + z_int of task (t+5) mod 10).
- **B1 (token carries goal):** success(I-only) − success(empty, from v1) paired CI lower bound > 0. **B2 (token is goal-specific):** success(I-only) − success(I-wrong) paired CI lower bound > 0. Report I-only vs correct-language (19/100) descriptively; no superiority claim. Output `results/libero_intention_only_v1/` with videos on d3.

### Stage 5 (CPU, optional): own dataset as the new independent test
If `docs/own_dataset_protocol.md`-conformant annotations exist and `scripts/validate_own_dataset.py` passes: score F (with C1 fallback since clips are single steps unless earlier steps exist) and, only if V1/V2/V4 passed, the selected variant, using K-way candidates within the same scene card set. Report counts and accuracy with Wilson intervals only; label "small independent set". If no data, skip.

---

## Implementation details fixed now

General (Track A):
- Data: the ladder cache. so400m frames are float16 → float32, T1 text float32. Folds come from `assign_folds(scenes, 8)`, and C1/C2 come from `run_ladder_v1` exactly as in the ladder (C1 means from training-fold queries; C2 context = mean of strictly earlier steps of the same video, with C1 fallback).
- The F reproduction gate uses the `run_selective_frozen.py` code path (per-fold selected config from `results/ladder_v1/selected_configs.csv`).
- Hits are strict (positive > all distractors), as in the ladder. Gallery MRR is over the held-out-fold gallery, with exact-text positives. Bootstrap = `run_scene_cv.video_bootstrap` (B=10000, seed 0; per-query deltas, denominator 1).
- "Improves on F", for every Stage: 5-way delta CI lower bound > 0 at ≥ 2 of 4 prefixes AND MRR delta CI upper bound > 0 at all 4 prefixes. For V4 only the 5-way part applies, at both k ∈ {2, 8}: 5-way delta CI lower bound > 0 at k=2 and at k=8.

Stage 1:
- Grid order (index): aggregator outer [A0, A3, A4h1, A4h2, A4h4], λ inner [0, 0.25, 0.5, 1, 2, ∞]. Index 5 = (A0, ∞) = F.
- The aggregator is applied identically to the C0 frames and the C2 frames. A4 uses re-normalised weighted mean frames with j = 1..k (w_k = 1 is the largest). A3 = frame k.
- cos_C0 uses raw frames and raw T1 texts. cos_C2 uses F's processing (C2 frames; texts raw, or C1 texts on fallback). λ = ∞ scores with cos_C2 alone.
- Selection: summed training-fold K=5 hits over k (integers) per config; `argmax` → lowest index on ties. No parameter is fitted, so no inner loop is needed (C1 means are training-fold means, as in F).
- Gallery MRR uses the same score s over the held-out gallery.

Stage 2:
- z = the aggregated, re-normalised query embedding of the selected (or F's) aggregation at each k, after F's C2 / C1-fallback processing. Training examples = every training query × k ∈ {2,4,6,8}. Candidate text vectors are the pipeline's texts for that query (raw T1, or C1-centred T1 on fallback). λ-fusion is not used inside the adapter. If V1 passes, "best Stage-1 aggregation" means the per-fold selected aggregator, with s = cos_C2 of the adapted query, for the adapter input only.
- Batch = 64 queries. The negatives of each query are all distinct same-video WHY texts other than its own (hard), plus the WHY texts of the other queries in the batch that come from other videos, capped at 64 (random). The control drops the hard negatives and samples 64 other-video texts.
- Inner selection: for each of the 6 (wd, epochs) configs, leave-one-scene-out over the training scenes, trained with seed 0; criterion = summed inner held-out K=5 hits over k; ties → lowest index (wd outer [1e-2, 1e-1, 1], epochs inner [5, 20]).
- Final: train on all training-fold queries with the selected config for seeds 0, 1, 2. Per-query held-out metrics (hits, RR) are averaged over the 3 seeds before the paired bootstrap.
- Assert: the held-out fold's indices never enter any training batch or inner selection.

Stage 3:
- Qwen2.5-VL-3B-Instruct, bf16, eager or sdpa attention, frames resized so the longer side is 448 px, given as k images in one user message (chat template with `add_generation_prompt`). Options are written as "A. <why>" ... "E. <why>".
- Order 1 = canonical `build_kway` order (positive at index 0); order 2 = reversed. Score per candidate = mean of its letter log-probs (log_softmax over the full vocabulary at the first answer position) across the 2 orders. Letter tokens = the tokenizer ids of "A".."E".
- Ties: fractional credit, as in E1b v2.
- Paired delta vs F: F's per-query out-of-fold strict hit vs MLLM fractional credit on the identical evaluated queries.
- Fusion: per query, z-score each method's 5 candidate scores across the candidates (std floor 1e-6). γ is chosen per outer fold by summed training-fold K=5 hits over the evaluated k; ties → lowest γ.
- Control donor: deterministic crc32 donor from a different scene and video (the ladder control rule), on every 8th evaluated query (sorted sample_id) at k=8.

Stage 4:
- Code: a copy of the training script with an `intention_only` condition (the original script is unchanged). The dataset wrapper sets `task` to "" and `intention` to the cached so400m T1 embedding of the original task string. VLIA config: `intention_dim` = 1152, everything else cloned from the baseline config, `strict=False` load from `lerobot/smolvla_base` as in the oracle condition.
- Task-embedding cache: so400m text features (same tokenisation as the ladder, max_length 64) for each unique task string in the training dataset and for the 10 LIBERO-10 instructions. It is saved to `/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings.npz`.
- Probe: 200 steps → seconds per step; ETA = s/step × 25000.
- Evaluation: the `libero_goal_ablation.py` protocol and code path, with an `intention` tensor added to the policy batch and the language set to "". Pairing with v1 is by (task, episode), using v1's `empty` per-episode results. B1/B2 use a paired bootstrap over the 100 (task, episode) units, B=10000, seed 0.
