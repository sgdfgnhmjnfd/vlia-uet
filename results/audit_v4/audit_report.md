# Prefix ablation and selective-decision audit

## Executive result

The saved prefix artifacts are internally consistent against their raw per-run JSON and validation cache; the run JSON does not contain model weights or similarity matrices, so this audit does not reproduce training or independently reconstruct all candidate scores. The main result remains validation-only, and `best_val_epoch` is selected on the same validation examples used for reporting. The current selective summary is not a valid evidence of deferral: its 25% risk gate is infeasible on the observed data and the fallback selects the full-coverage point.

## Implementation and artifact checks

- **PASS** No train/validation video_uid overlap: overlap=[].
- **PASS** Unique train/validation sample identities: train=556, val=215.
- **PASS** Train/validation sample identities disjoint: overlap=0.
- **PASS** Config video IDs agree with loaded cache: train_uid=6, val_uid=2.
- **PASS** Per-example rank table uses cache validation identities and order: rows=215, cache_val=215.
- **PASS** Positive mask matches exact WHY equality: positive pairs=407, diagonal=True.
- **PASS** Chance baseline matches saved config: {"MRR": 0.04220487522450658, "Top1": 0.008804759329367261, "Top3": 0.026126726036073664}.
- **PASS** Gallery semantics: all val WHY targets; exact-text positives: gallery=215, unique_why=167.
- **PASS** Raw per-seed ranks and predictions exist for all conditions: combinations=48.
- **PASS** Recomputed per-seed and aggregate metrics match saved CSV: mismatches=0; Top1/3/5 use positive rank <=1/3/5.
- **PASS** Evaluation sample IDs and gallery are identical across prefixes: n_examples=215, gallery_candidates=215.
- **PASS** Positive labels are exact WHY-text matches: same construction as build_positive_mask/retrieval_metrics.
- **PASS** Top-1 label correctness agrees with rank-1 metric: disagreements=0; exact-score ties would require score vectors to resolve.
- **PASS** Saved training-length rank cross-check is within tolerance: max_abs_diff=3.9733093204485925e-09.
- **PASS** Paired rank comparisons use the same per-example identities: comparisons=16.
- **PASS** Source annotation manifests inspected: manifests=32, unique_video_uid=32.
- **PASS** All pilot source UIDs occur in full dataset inventory: pilot_uids=8, missing=[].
- **PASS** train split source paths resolve under the dataset root: resolved=2091/2091, stale_d1_paths_remapped=2091, missing=0.
- **PASS** train split sample IDs are unique and split-disjoint: duplicate_within_split=0, overlap_with_previous_splits=0.
- **PASS** train cached feature filenames match split sample IDs: cache_files=2091/2091; filenames checked without loading tensors.
- **PASS** val split source paths resolve under the dataset root: resolved=412/412, stale_d1_paths_remapped=412, missing=0.
- **PASS** val split sample IDs are unique and split-disjoint: duplicate_within_split=0, overlap_with_previous_splits=0.
- **PASS** val cached feature filenames match split sample IDs: cache_files=412/412; filenames checked without loading tensors.
- **PASS** test split source paths resolve under the dataset root: resolved=511/511, stale_d1_paths_remapped=511, missing=0.
- **PASS** test split sample IDs are unique and split-disjoint: duplicate_within_split=0, overlap_with_previous_splits=0.
- **PASS** test cached feature filenames match split sample IDs: cache_files=511/511; filenames checked without loading tensors.
- **PASS** Full-dataset source video UIDs are split-disjoint: overlapping_uids=[].

- Provenance note: `config.json` records run commit `34571c46e5b82300adf92ebfd94549fbc6007d05`, while the current repository HEAD may differ; artifacts are audited against the checked-out code and saved raw outputs, not regenerated model weights.
- Gallery consists of all 215 validation WHY-target rows; a query is positive for every gallery row whose WHY text exactly matches the query WHY. Rank-based MRR/Top-1/3/5 are recalculated from saved positive ranks.
- The raw JSON persists positive ranks and top-1 gallery indices, but not similarity/logit matrices. Thus aggregate rank metrics and selected WHY labels are rechecked; a complete independent reconstruction of every candidate ordering from raw scores is impossible. For the main train@k setting, each trained prefix is the model's own train length and the saved project-metric cross-check maximum absolute difference is `3.97e-09`. Exact-score tie behavior for prefixes without a score matrix cannot be separately audited.
- `best_val_epoch` selects the epoch with the maximum validation MRR at each model's own training prefix; this directly optimizes the metric on the same validation set and is optimistically biased. `last_epoch` is the final training epoch (no validation-based checkpoint selection), but is still evaluated on validation.
- Prefixes use the first k of eight features sampled across the entire clip, not a fixed elapsed-time cutoff. Changing prefix k does not change train/validation sample identities or gallery.
- Error counts pool 3 seeds × repeated sample IDs; the distinct validation unit remains only two `video_uid`s. No seed/sample count is interpreted as independent source-video replication.

## Main train@k/eval@k results

| Frames | MRR mean ± seed SD | Top-1 | Top-3 | Top-5 | n examples/seed | same-task share among errors |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0551 ± 0.0035 | 0.0310 | 0.0326 | 0.0481 | 215 | 0.874 |
| 4 | 0.0545 ± 0.0039 | 0.0279 | 0.0357 | 0.0496 | 215 | 0.810 |
| 6 | 0.0604 ± 0.0047 | 0.0310 | 0.0481 | 0.0667 | 215 | 0.756 |
| 8 | 0.0589 ± 0.0016 | 0.0295 | 0.0419 | 0.0558 | 215 | 0.814 |

### Paired prefix differences (same examples)

The CSV includes seed-level changes and an exploratory bootstrap over the two source-video clusters. With only two validation videos, that interval has too few clusters to support population-level uncertainty claims. The earlier published interval resampled individual examples and understates clustering uncertainty.

| Prefix pair | ΔMRR (paired) | 2-video cluster bootstrap 95% interval | Video-specific ΔMRR |
|---|---:|---:|---|
| 2→4 | -0.0005 | [-0.0012, +0.0002] | `{"02d1f024-1470-4ce6-acde-b938f7847eb6": -0.0011679256919520098, "cc883eb4-7772-4ef8-bd85-2c9cb5c6565d": 0.0001800910793326764}` |
| 4→6 | +0.0058 | [+0.0014, +0.0100] | `{"02d1f024-1470-4ce6-acde-b938f7847eb6": 0.009984571461082165, "cc883eb4-7772-4ef8-bd85-2c9cb5c6565d": 0.001407487935712451}` |
| 6→8 | -0.0015 | [-0.0060, +0.0033] | `{"02d1f024-1470-4ce6-acde-b938f7847eb6": -0.005976207122113483, "cc883eb4-7772-4ef8-bd85-2c9cb5c6565d": 0.003252543228578132}` |
| 2→8 | +0.0038 | [+0.0028, +0.0048] | `{"02d1f024-1470-4ce6-acde-b938f7847eb6": 0.002840438647016671, "cc883eb4-7772-4ef8-bd85-2c9cb5c6565d": 0.004840122243623264}` |

### Same-task goal confusions

For the main setting, 2036 of 2503 seed-example Top-1 errors (81.3% if any errors) were within the query task. This uses the selected gallery candidate's task and the same exact-WHY correctness test as the training script. Counts are repeated across seeds; `unique_error_samples` in the CSV avoids treating those repetitions as new examples.

Most frequent label-pair examples (counts are seed-example errors; text labels are not human error annotations):

| Frames | Task | Ground-truth WHY | Predicted WHY | Error seed-examples | Unique examples |
|---:|---|---|---|---:|---:|
| 8 | pick_fruits | detaching the lemon from the branch | store the harvested lemon | 13 | 7 |
| 6 | pick_fruits | detaching the lemon from the branch | harvesting the lemon | 12 | 8 |
| 2 | pick_fruits | detaching the lemon from the branch | harvesting the lemon | 10 | 7 |
| 4 | pick_fruits | detaching the lemon from the branch | harvesting the lemon | 10 | 8 |
| 8 | pick_fruits | collecting harvested lemons | store the harvested lemon | 7 | 4 |

The available task/WHY strings support descriptive confusion tables only. There are no verified human labels for visual ambiguity, action phase, or semantic evidence sufficiency, so no richer qualitative taxonomy is asserted.

## Selective-prediction audit

The raw threshold sweep contains 5760 aggregate rows across confidence, margin, and entropy. Reconstructed risk–coverage curves and explicit always-predict endpoints are in `risk_coverage.csv` and `risk_coverage_{confidence,margin,entropy}.svg`. The SVGs show score-based rejection as solid lines and a random-rejection baseline at the same coverage as dashed lines; CSV rows also contain the exact finite-population random-risk expectation and 95% hypergeometric interval (conditional on the validation run, not population-level video uncertainty). The always-predict error spans 0.963–1.000 across saved condition/prefix/seed combinations (these are separate validation runs, not independent observations).

The existing Markdown summary collapses each group to a threshold, not the full curve, and reports coverage 1.0 because no observed risk satisfies its ≤0.25 constraint; its fallback considers all rows and prefers lower risk then greater coverage. The CSV keeps threshold-sweep rows but omits the bootstrap CI fields calculated in memory by the original script. Existing plot generation only plots confidence, not margin or entropy. These validation sweeps are post-hoc diagnostics: no independent calibration split exists, and confidence, margin, or entropy are not evidence of semantic sufficiency or calibrated uncertainty.

For a compact comparison, `selective_curve_comparison.csv` reports trapezoidal risk–coverage area for each condition/prefix/policy against random rejection at matched coverage. Lower area is preferable descriptively; this is measured on the same validation examples used to generate the curves and is not an independent policy-selection result.
Favorable score-area counts (lower than random, out of condition × prefix groups): confidence 5/12; margin 7/12; entropy 12/12. These are descriptive, not inferential.

## Full-dataset split and test protocol

The full dataset is present at `/media/dhqg/d3/datasets/egointent_full`: 32 annotation manifests and 32 source video UIDs. The full split manifests are `/media/dhqg/d3/datasets/egointent_full_split_v1` and contain train=2091 samples/23 UIDs; val=412 samples/4 UIDs; test=511 samples/5 UIDs.

The pre-existing split has 1 test UID(s) that overlap pilot validation and must not be treated as held out. Excluding that UID leaves 4 candidate test UIDs and 407 samples. These are frozen in `results/test_protocol_full_v1/locked_protocol.json`; they remain a candidate video-level test only, not a pristine or scene-disjoint benchmark.

The saved `egointent_full_intention_v1/summary.json` is a pilot-cache run: its recorded Stage-A and temporal cache paths point to `/media/dhqg/d1/datasets/egointent/cache/...`, and its validation metrics are over 215 pilot examples. Searching the available repository results and `/media/dhqg/d3/vlia_outputs` found no exact references to the four candidate UIDs. This is evidence about the inspected local artifacts, not proof that no external or unrecorded run ever accessed them. Before protocol lock, the test cache was counted by filename only. Candidate tensors were subsequently accessed only after lock, as recorded in the protocol addendum.

## Full-dataset prefix validation follow-up

After verifying the full split and cache, the 2/4/6/8-frame experiment was run on the full train/validation split only. The completed run used 2,091 training samples/23 UIDs, 412 validation samples/4 UIDs, 3 seeds, 100 epochs, and the existing train@k/eval@k protocol. WHY text targets were generated on a separate derived cache for train and validation; the original temporal cache was left unchanged. The fixed 960→256 PCA was fit on the earlier pilot training split, whose six UIDs are a subset of full train; it was not refit on all 23 full-training UIDs. No test tensors were loaded.

| Frames | Validation MRR (mean ± seed SD) | Top-1 | Top-3 | Top-5 |
|---:|---:|---:|---:|---:|
| 2 | 0.0440 ± 0.0064 | 0.0162 ± 0.0037 | 0.0356 ± 0.0109 | 0.0453 ± 0.0085 |
| 4 | 0.0448 ± 0.0072 | 0.0162 ± 0.0074 | 0.0364 ± 0.0106 | 0.0526 ± 0.0051 |
| 6 | 0.0474 ± 0.0010 | 0.0218 ± 0.0024 | 0.0413 ± 0.0042 | 0.0477 ± 0.0028 |
| 8 | 0.0429 ± 0.0053 | 0.0146 ± 0.0073 | 0.0380 ± 0.0056 | 0.0469 ± 0.0028 |

Six frames had the highest validation MRR and Top-1; performance did not improve monotonically to 8 frames. Paired comparisons resampled by the four validation video UIDs all had 95% intervals spanning zero. These are descriptive validation results, not evidence of statistical superiority or generalization. The complete table, video-cluster intervals, settings, and limitations are in [`results/prefix_length_full_v1/analysis.md`](../prefix_length_full_v1/analysis.md).

## Locked candidate test result

The candidate test was evaluated using the frozen 2/4/6/8-frame mean-pool protocol, three seeds, and validation-MRR checkpoint selection. There were three failed initializations before the successful metric run; no metrics were produced by the failures. The implementation fixes and exact code hashes are documented in `results/test_protocol_full_v1/protocol_addendum.json`. The test evaluation itself was completed once; no hyperparameter or threshold was selected from its results.

| Frames | Test MRR ± seed SD | Top-1 ± seed SD | Top-3 ± seed SD | Top-5 ± seed SD |
|---:|---:|---:|---:|---:|
| 2 | 0.0290 ± 0.0028 | 0.0057 ± 0.0014 | 0.0172 ± 0.0000 | 0.0287 ± 0.0038 |
| 4 | 0.0288 ± 0.0033 | 0.0082 ± 0.0038 | 0.0147 ± 0.0043 | 0.0246 ± 0.0000 |
| 6 | 0.0294 ± 0.0045 | 0.0090 ± 0.0051 | 0.0172 ± 0.0049 | 0.0221 ± 0.0043 |
| 8 | 0.0314 ± 0.0035 | 0.0098 ± 0.0049 | 0.0197 ± 0.0025 | 0.0270 ± 0.0025 |

Exact multi-positive random-ranking chance on the candidate gallery is MRR 0.0318, Top-1 0.0071, and Top-3 0.0208. Eight frames had the highest average MRR, but it is approximately chance. Paired prefix comparisons and video-cluster bootstrap intervals are in `results/locked_test_full_v1/test_analysis.md` and `paired_test_comparisons.csv`. Four video clusters are too few for a strong inferential claim. This result does not support statistical superiority or generalization.

## Single best next experiment

Do not tune or choose thresholds using the candidate test results. The next confirmatory run should use a new, scene-disjoint source-video holdout and a training-only-fitted WHY PCA, with the entire protocol locked before test access. The current garden candidate UIDs share a scene with pilot validation, and the PCA was fit on pilot train rather than all full-train UIDs; these constraints limit the current test result.

## Audit outputs

- `per_example_errors.csv` includes correct and incorrect top-1 outputs with an `is_top1_error` flag.
- `top1_errors_by_task.csv` and `top1_errors_by_why_pair.csv` provide task/pair grouping.
- `prefix_paired_comparison.csv` reports paired deltas with explicit two-video cluster limitations.
- `risk_coverage.csv` and three SVGs include threshold curves and always-predict endpoints.
- `selective_curve_comparison.csv` compares curve area against matched-coverage random rejection.
- `source_video_inventory.csv` records source UID/scene/event/step counts without modifying the pilot split.
- `full_split_inventory.csv` records full split sizes, video path resolution, and cache filename coverage.
- `candidate_test_uid_manifest.csv` lists candidate UIDs; the formal candidate lock and hash record are in `results/test_protocol_full_v1/`.
- `results/locked_test_full_v1/` contains the single successful candidate test evaluation and its post-hoc paired analysis.
