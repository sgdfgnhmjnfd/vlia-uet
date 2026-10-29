# Pre-registration: Stage 10, call Qwen only when F is unsure (2026-10-29)

Written and committed before any Stage 10 number was computed. Everything comes from saved scores; no model is run.

**Data**
- EgoIntent train/val only (`results/hoi_detection_v1/manifest.jsonl`). The locked test split is never read.
- Label WHY, 5 same-video candidates.
- F = SigLIP so400m | A0 (mean of frames) | C2 | T1.
- Qwen2.5-VL-3B scores: `/media/dhqg/d3/vlia_outputs/v3_mllm/scores/*.json`, `control == false` only, k ∈ {2, 8}.

## Method

Script: `scripts/run_stage10_cascade.py`. Synthetic self-test: `scripts/test_stage10_cascade.py`.

1. **Reuse the Stage 8c analyze path exactly** (`scripts/v3_stage3_mllm.py` `analyze`, from `origin/main`):
   - F out-of-fold candidate scores (`L.candidate_scores`, A0) on the 8 scene-grouped folds;
   - the evaluated query set (queries with Qwen scores at both k);
   - `zscore` within each query's 5 candidates;
   - nested γ ∈ {0, 0.25, 0.5, 1, 2} per outer fold, chosen on the training folds' evaluated queries by summed strict hits over k = 2 and 8, ties → first (smaller) γ;
   - strict-hit scoring (`L.strict_hits`; a tie counts as wrong).
2. **F margin** of a query at k = top-1 − top-2 of F's raw 5 candidate scores (the out-of-fold scores of step 1). It uses no labels.
3. **Cascade at budget b ∈ {10%, 25%, 50%, 100%}**, per outer fold and k:
   - t_b = `np.quantile` (default linear interpolation) at b of the F margins of the training folds' evaluated queries at the same k;
   - a held-out query with margin < t_b (strict) uses z(F) + γ_fold·z(Qwen), all others use F;
   - b = 100% is defined as full fusion: every query uses the fused score (t = +∞). This avoids the edge case where a held-out margin ≥ the largest training margin would skip Qwen.
4. **For each (k, b):**
   - accuracy (strict);
   - the fraction of evaluated queries that called Qwen;
   - expected Qwen seconds per query = fraction × 2.951939 s (`run_info.seconds_per_query` in `results/v3_stage3_mllm/config.json`);
   - paired per-query differences: cascade − F and cascade − full fusion.
   - CIs: `CV.video_bootstrap` (B = 10000, seed 0), clustered by source video.
5. **Diagnostic at b = 25%** for each k: accuracy of F and of full fusion inside the low-margin group (called Qwen) and inside the high-margin group (did not), with group sizes.

## Gates (stop if any fails)

- **G1:** F's mean strict hits on the evaluated queries equals `F_5way_acc_same_queries` in `results/v3_stage3_mllm/summary.csv` at k = 2 and 8 (|diff| < 1e-12). F hits over all queries also reproduce {2: 853, 8: 969} (`V.f_gate`).
- **G2:** at b = 100%, the fused accuracy equals the `fusion(F + gamma*MLLM)` rows of that summary (|diff| < 1e-12). The γ chosen per fold and the training hit totals equal `results/v3_stage3_mllm/fusion_gamma_by_fold.csv` exactly.

## Pre-registered rules

- **C1:** at k = 8, b = 25%, cascade − F > 0 with the 95% CI lower bound > 0.
- **C2:** at k = 8, b = 25%, (cascade − F) ≥ 0.5 × (full fusion − F), as point estimates.
- k = 2 is reported but is not part of the rules.

## Outputs

`results/stage10_cascade/`: `summary.csv`, `paired.csv`, `groups.csv`, `config.json`.
