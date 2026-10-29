# Pre-registration: Stage 9, improving the prediction part (2026-10-29)

Written and committed before any Stage 9 number was computed, with one exception, disclosed under 9b. Data: EgoIntent train/val only (`results/hoi_detection_v1/manifest.jsonl`, 2503 steps, 27 videos). The locked test split is never read. F = SigLIP so400m | A0 (mean of frames) | C2 | T1.

## General rules (all parts)

- Accuracy is strict top-1 among 5 same-video candidates (a tie counts as wrong) unless stated otherwise.
- Compare methods with paired per-query differences.
- 95% CIs come from a video-level bootstrap (B=10000; B=2000 is allowed for the 9a grid).
- A difference is "clear" only if its whole CI is on one side of 0.

Bootstrap implementation: `run_anticipation_horizon.video_bootstrap`, seed 0. Videos are resampled with replacement; the statistic is the ratio of summed values to summed counts.

## 9a. Frame count vs window length

**Question.** Does the number of frames matter, or only how many seconds the model looks back?

Implementation: subcommand `frames` of `scripts/run_time_horizon.py`. The existing `run` path is unchanged.

**Data and settings**
- 4 fps cache `/media/dhqg/d3/vlia_outputs/time_cache_4fps/`.
- Label WHAT; horizons H ∈ {0.5, 2, 4, 8} s.
- Anchors, seeded candidates and the dropped-video rule are those of `run` with its default grid (contexts up to 16 s, horizons up to 20 s). Anchors are taken every 4 frames, from frame 63 to `len − 80`, so the set of queries is identical to `results/time_horizon_v1`.

**Windows and frame counts**
- Window W ∈ {2, 4, 8} s = 4W frames ending at the anchor frame a.
- N frames, evenly spaced, always including the newest: frames a, a−s, …, a−(N−1)s with stride s = 4W/N.
- N ∈ {2, 4, 8} for W=2; {2, 4, 8, 16} for W=4; {2, 4, 8, 16, 32} for W=8.
- C2 reference: the mean of all frames before the window start a−4W+1, or the global frame mean if there are none. It is the same for every N of a given W.
- Query = unit(mean of unit(frame − reference)).
- Contiguous contexts 3 s and 6 s (12 and 24 frames) use `query_vec` exactly as `run` does.

**Gate.** With N = 4W, ours_acc5 must reproduce `results/time_horizon_v1/summary.csv` (WHAT, C = W, same H) within 1e-9. If it does not, stop.

**Outputs** in `results/stage9a_frames/`:
- `summary.csv` and `config.json`.
- `paired_N.csv`: within each (W, H), each N vs the next larger N (diff = larger − smaller). Also 12 frames (3 s) vs 8 frames (W=2, N=8, i.e. 2 s), and 24 frames (6 s) vs 16 frames (W=4, N=16, i.e. 4 s).

**Pre-registered reading.**
- "Frame count matters" if at some W, for H ≤ 2 s, a pair of consecutive N differs clearly by ≥ 1 point.
- Otherwise report "window length, not frame count, decides".

## 9b. Fair recognition metric (diagnostic, no pass/fail)

Recompute F's current-step WHAT recognition at k = 2/4/6/8. The gallery is all steps of the same video, scored out of fold exactly as in `run_literature_baselines.cmd_next`. Measures:

1. **Exact row match**: the top-1 row is the step itself.
2. **Text match, raw**: WHAT of the top-1 row == WHAT of the step.
3. **Text match, normalised**: lower-case, whitespace collapsed.
4. **Recall@5 and recall@10 by text**:
   - Rank the distinct normalised WHAT texts of the video.
   - A text's score is the maximum F score over the rows that carry it.
   - A step counts as a hit if its normalised text is among the top 5 (10).
5. **Distinct normalised WHAT texts per video**: median, min, max.

CIs: video bootstrap, B=10000.

**Correction to the task text (disclosed).** The task described the stored numbers 0.040 / 0.048 / 0.053 / 0.062 (`results/lit_baselines_next/config.json`, `what_recognition_top1_in_video`) as exact row matches. The code that produced them (`cmd_next`) actually computes raw WHAT text equality.

A quick check made while writing this file, from the saved `recognized_what.json`:
- row match: 0.0376 / 0.0463 / 0.0503 / 0.0583;
- raw and normalised text match: 0.0400 / 0.0483 / 0.0527 / 0.0619.

Gate for 9b: measure 2 (raw text match) must reproduce the stored numbers within 1e-9. Row match is reported as measure 1 only.

## 9c. Better recognition, then AntGPT

**Shortlist**
- For every step (all 2503), take F's top-5 distinct normalised WHAT texts among all steps of the same video, scored as in 9b.
- Use k=8 for every step, and also k=2.
- If a video has fewer than 5 distinct texts, the shortlist holds all of them (fewer letters).
- Each shortlisted text is shown in the original wording of its highest-scoring row.
- The shortlist order is F's ranking.
- If the true text is not in the top 5, Qwen cannot fix it. That ceiling equals recall@5 from 9b.

**Re-rank**
- Qwen2.5-VL-3B-Instruct, bf16, sdpa, as in `v3_stage3_mllm.py` (`origin/main`).
- Frames: the 8 ladder frames of the step (`hoi_utils.sample_exact_video_frames`), resized so the longer side is 448 px. Use the first k frames.
- Score: letter log-prob at the answer position, averaged over the canonical (F-ranked) and reversed option orders.
- Fixed question:
  "These frames are from a first-person video of one step of a task. Which action is the person doing in this step? Options: {options} Answer with one letter."
  `{options}` = "A. <text> B. <text> …".
- Resumable: one jsonl line per (sample_id, k).
- Probe 30 queries first and print seconds per query and the ETA. Then run k=8, then k=2.

**Recognised labels**
- `rec_qwen` = Qwen's top option. Ties go to the earlier option, i.e. F's ranking.
- `rec_fuse` = argmax of z(F) + γ·z(Qwen) over the shortlisted options. z-scores are taken within each step's shortlist. A text's F score is the max over its rows.
  - γ ∈ {0, 0.25, 0.5, 1, 2} is chosen per outer fold (the 8 scene-grouped folds) on the other folds. It maximises text-match recognition accuracy at that k.
  - Ties go to the smaller γ (0 first).
- F's label = shortlist position 1.
- Recognition accuracy = normalised text match, over all 2503 steps.

**Anticipation**
- New AntGPT variants: `qwen_k2`, `qwen_k8`, `fuse_k2`, `fuse_k8`.
  - The previous steps (up to HISTORY = 5) use the k=8 recognised text of each step.
  - The current step uses the k-frame recognised text.
- The text-only AntGPT scorer (`run_literature_baselines.LLM`, same prompt, same 5 NEXT candidates, same query set) is unchanged.
- AntGPT accuracy uses fractional tie credit, as in the existing `antgpt-analyze`. This is needed to reproduce the gate below.
- F on NEXT uses the strict `ours` hits of `results/lit_baselines_next/per_query.csv.gz`.
- Fusion z(F) + γ·z(LLM) uses `nested_gamma` exactly as in `cmd_fusion` (strict).

**Gates**
- F must reproduce hits {2: 853, 4: 874, 6: 907, 8: 969}.
- The saved `antgpt_scores/rec_k8.jsonl` must give 0.2917 (|x − 0.29165| < 5e-5).
- 9b's raw text-match recognition must reproduce at k = 2 and 8.

**Outputs** in `results/stage9c_qwen_recognition/`: `recognition_summary.csv`, `antgpt_summary.csv`, `fusion_summary.csv`, `config.json`, `run_info.json`, plus the score jsonl files.

**Pass rules**
- **R1 (recognition):** `rec_fuse` text-match recognition − F's text-match recognition > 0 clearly at k=8.
- **A1 (anticipation):** AntGPT with `fuse_k8`, or with `qwen_k8` if `rec_qwen` has the higher k=8 recognition point estimate in R1's numbers (tie → `fuse_k8`), minus F on NEXT > 0 clearly at k=8.
- k=2 is reported but is not part of the pass rules.

## 9d. Feasibility check only

LaViLa and EgoVLP/EgoVLPv2:
- weight source and size;
- missing packages;
- free disk space;
- whether EgoIntent videos come from Ego4D, which would be a leakage risk.

Nothing is downloaded or installed.
