# Pre-registration: v3 supplementary experiments (written 2026-10-27, before any number)

Script: `scripts/run_v3_supplementary.py`. Data: EgoIntent train/val ladder cache only (2503 queries,
27 videos, scene-grouped 8 folds as in `results/ladder_v1`). The locked test split is never read.

Reference F = `siglip-so400m-patch14-384 | A0 | C2 | T1`. **Gate:** F must reproduce the out-of-fold
5-way hits 853 / 874 / 907 / 969 at k = 2 / 4 / 6 / 8 exactly; otherwise nothing else runs.

Selection rule (Stages 1 and 2): per outer fold, choose the grid value with the highest mean 5-way
accuracy over k in {2, 4, 6, 8}, computed on training-fold queries only (Stage 2: on an inner split of
held-out training scenes, `run_scene_cv.inner_split`); ties go to the lowest grid index. Apply to the
held-out fold and pool. Paired deltas vs F with a video bootstrap (B = 10000, seed 0) for 5-way,
10-way and gallery MRR.

**"Improves on F"** = 5-way delta CI lower bound > 0 at >= 2 of 4 prefixes AND gallery-MRR delta CI
upper bound > 0 at all 4 prefixes.

## Stage 1 — aggregation x context fusion (no training)
Aggregators over frames 1..k: A0 mean (F); A3 last frame only; A4 recency weights
w_j ∝ exp(−(k−j)/h), h in {1, 2, 4}. Fusion score s = cos_C0 + λ·cos_C2, λ in {0, 0.25, 0.5, 1, 2, ∞}
(λ = ∞ is C2 alone). Grid 5 × 6 = 30, F first. **V1** = "improves on F".

## Stage 2 — hard-negative residual adapter
z' = norm(z + W z), W in R^{1152×1152} zero-initialised (identity at start), on F's query vector
(A0, C2) for every k; text side frozen. InfoNCE (τ = 0.05) over the positive WHY, all other distinct
WHY texts of the same video (hard negatives) and 64 random WHY texts of other videos. AdamW lr 1e-3,
batch 64; grid weight decay {1e-2, 1e-1, 1} × epochs {5, 20}, selected with seed 0 on the inner split;
final adapters trained on the whole training fold with seeds {0, 1, 2}, held-out metrics averaged over
seeds. Control: same, random negatives only. **V2** = "improves on F" for the hard-negative adapter.
Deviation from `prompt_2026-10-23_improve_v3.md`: inner selection uses one held-out set of training
scenes instead of leave-one-scene-out (CPU budget), and the input is F's aggregation regardless of V1.

## NEXT — predict the observed next step
Label = `observed_next_step` of the query step (from `step_label.json`), candidates = 5 (and 10)
distinct NEXT strings of the same video (`build_kway`), T1 template, so400m text encoder.
Query = first k frames of the current step, F's processing (C2; C0 reported too). No parameter is
selected. Reported: 5-way / 10-way accuracy with lift CIs, gallery MRR, an upper bound where the query
is the following step's 8 frames, and the share of steps whose NEXT text equals the following step's
WHAT. Descriptive only; no verdict.
