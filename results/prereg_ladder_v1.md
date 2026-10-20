# Preregistration: zero-parameter improvement ladder, early decision, and cache-kNN

Committed before any new embedding extraction or analysis under this protocol. The E1b run already in progress is not a new computation under this preregistration.

## Stage 2: multi-encoder embedding cache

Decode each train/validation manifest sample's 8 frames once (same RGB sampler as the existing feature cache), run all encoders on the same frames, and save per encoder under `/media/dhqg/d3/vlia_outputs/ladder/<encoder_name>/`: `frame_emb.npy` float16 `(N, 8, D)`, `sample_ids.json`, text embeddings for both templates below, and counts of truncated texts.

Encoders, in listed order:

1. `google/siglip-base-patch16-224` (baseline). Its K=5 accuracies at k=2,4,6,8 must reproduce `results/e1_zero_shot_v1/summary.csv` to absolute tolerance `1e-9`; otherwise stop and report the mismatch.
2. `google/siglip-base-patch16-384`
3. `google/siglip2-base-patch16-224`
4. `google/siglip-so400m-patch14-384`
5. `openai/clip-vit-large-patch14`

Use public models only and no tokens. For SigLIP, tokenize with `padding="max_length"`, `max_length=64`; for CLIP use `max_length=77`; count truncations. Fixed text templates: T0 = raw WHY string; T1 = `a first-person photo of a person doing: {why}`. Measure throughput on 200 samples first. If projected total exceeds 3 hours, drop `so400m` first, then `clip-vit-large`, and record the drop. If GPU memory is insufficient, load encoders one at a time (re-decode frames) rather than failing.

## Stage 3: nested pipeline ladder

Run on cached embeddings on CPU with numpy. Grid: encoder × aggregator × centering × template.

Aggregators over the first k frames, for k in `{2,4,6,8}`:
- A0: cosine of the mean of the k frame embeddings, re-normalized, against text candidate.
- A1: maximum over frames of per-frame cosine.
- A2: log-sum-exp over frames of per-frame cosine with temperature `0.02`.

Centering:
- C0: none.
- C1: modality-gap correction: subtract mean frame embedding and mean text embedding (both computed only on training-fold queries of the current outer fold), then re-normalize.
- C2: causal video context: subtract mean of all frames from the same video at steps strictly earlier than the query's step; fall back to C1 when no earlier step exists. Parse step order numerically from `step_id`, assert earlier steps are strictly earlier, and print the fallback fraction. If order cannot be parsed reliably, skip C2 and report this.

Templates are T0 and T1 from Stage 2. Outer folds are `assign_folds(scenes, 8)`. In each outer fold, score every grid config on training-fold queries by mean K=5 accuracy across k `{2,4,6,8}`; only C1 means are fitted. Select the argmax, breaking ties by lowest grid index, apply that one config to held-out queries, and pool held-out predictions. This is the nested-selected pipeline. Baseline is `(siglip-base-patch16-224, A0, C0, T0)` with no selection. Report the selected config for every fold and selection stability.

Metrics: gallery MRR over the held-out-fold gallery with exact chance (mean `0.02977`); K=5 and K=10 accuracy (chance `0.20` and `0.10`) using `build_kway`; lift = value / chance; video bootstrap `B=10000`, seed 0. Secondary metrics, never used for a verdict: clean 5-way (distractor MiniLM cosine `<0.8`) and paraphrase-aware MRR (positives at MiniLM cosine `>=0.9`).

Control: for each config actually selected, on every 8th query sorted by `sample_id` at k=8, replace frames with frames of a different video from a different scene and report control accuracy beside real accuracy. Also evaluate the full grid on all queries descriptively, explicitly label it “not selection-corrected; no verdict.”

Verdicts:
- N1: nested-selected minus baseline, paired on identical queries and k with video bootstrap. “Selection helps” iff the CI lower bound is greater than 0 for both gallery MRR and 5-way at at least 2 of 4 prefixes.
- N2: yesterday's G1 remains FAIL and unchanged. The new practical bar passes iff nested-selected 5-way accuracy is at least `0.30` at at least 2 of 4 prefixes, and the lift CI lower bound is greater than 1 for MRR and for 5-way at at least 2 of 4 prefixes.

Outputs in `results/ladder_v1/`: `summary.csv`, `summary.md`, `selected_configs.csv`, `paired_vs_baseline.csv`, `grid_descriptive.csv`, `control.csv`, lift-vs-k plot (baseline, nested-selected, chance, with CI), `config.json`, and `INTERPRETATION.md` containing facts only and limitations: 27 videos; same-video distractors may be paraphrases; grid selection uses training scenes only; no untouched test.

`--selftest` (numpy only): aggregators on hand-made arrays; C1 means never use held-out fold; C2 never uses a same-or-later step; nested selection never reads held-out labels.

## Stage 4: evidence-accumulation early decision

Use baseline only (no selection). For each query calculate `margin_k` = top-1 minus top-2 cosine among K=5 candidates from the mean embedding of the first k frames, for k=1..8. Also calculate fixed-k accuracy at k=2..8.

Stop at the first k in 2..8 with `margin_k >= tau`; if none, decide at k=8. For each outer fold choose tau on training-fold queries so mean frames used is closest to budget `b` in `{3,4,5,6}` (candidates are observed margins), then apply to held-out fold and pool. Report realized mean frames and accuracy. Comparator: fixed-k accuracy curve over k=2..8 linearly interpolated at realized mean frames. Compute paired early-decision minus interpolated fixed-k difference with video bootstrap.

Verdict S2: early decision beats fixed-k iff the CI lower bound is greater than 0 at at least 2 of 4 budgets. Plot accuracy versus mean frames (early-decision points, fixed-k curve, chance). Outputs in `results/ead_v1/` and `INTERPRETATION.md` with facts only, stating margin is a decision-score baseline, not a validated measure of semantic evidence sufficiency.

## Stage 5: cache-kNN text-space correction

Run only after Stage 3 is done. Use baseline embeddings. For a held-out query prefix embedding x, retrieve M=20 most similar training-scene queries at the same k; weights are `softmax(cos/0.05)`, and `y_hat` is the weighted sum of those neighbors' T0 WHY text embeddings. Score each candidate with `cos(x, t_c) + beta * cos(y_hat, t_c)`. Choose beta from `{0,0.25,0.5,1,2}` per outer fold using leave-one-scene-out inside the training scenes (neighbors exclude the query's own scene), maximizing mean 5-way accuracy over k. Neighbors never come from the held-out fold.

Verdict K1: beta-selected minus baseline, paired with video bootstrap; CI lower bound greater than 0 for MRR and 5-way at at least 2 of 4 prefixes. State that this method uses training-scene labels and is a non-parametric trained method. Outputs in `results/knn_cache_v1/`.
