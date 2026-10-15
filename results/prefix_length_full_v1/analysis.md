# Full-dataset prefix-length validation analysis

## Execution and protocol

The full-dataset run completed with the existing prefix-ablation script and its 3 seeds (0, 1, 2), 100 epochs, and 2/4/6/8-frame prefixes. The main `train_k_eval_k` condition trains and evaluates each model at the same prefix length. The gallery is the 412 validation WHY targets; exact WHY text matches are positives. Training and validation contain 2,091 samples/23 video UIDs and 412 samples/4 video UIDs respectively.

Frame features are the existing cached 8×960 frozen visual features. Compatible text targets were generated only for train and validation in `/media/dhqg/d3/vlia_outputs/egointent_full_stage_a_v1`; the original full temporal cache remains unchanged and the derived cache contains no test split. The existing frozen 960→256 WHY PCA was used. Its fitting code uses the pilot training split only; those six source UIDs are a subset of the full training split, so the transform does not use full validation or test labels, but it was not refit on all 23 full-training UIDs. Test cache tensors were not loaded, and no test metrics were computed.

## Main results: train@k/eval@k, best validation epoch

| Frames | MRR (mean ± seed SD) | Top-1 | Top-3 | Top-5 | Examples/seed | Same-task share of Top-1 errors |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0440 ± 0.0064 | 0.0162 ± 0.0037 | 0.0356 ± 0.0109 | 0.0453 ± 0.0085 | 412 | 0.470 |
| 4 | 0.0448 ± 0.0072 | 0.0162 ± 0.0074 | 0.0364 ± 0.0106 | 0.0526 ± 0.0051 | 412 | 0.471 |
| 6 | 0.0474 ± 0.0010 | 0.0218 ± 0.0024 | 0.0413 ± 0.0042 | 0.0477 ± 0.0028 | 412 | 0.460 |
| 8 | 0.0429 ± 0.0053 | 0.0146 ± 0.0073 | 0.0380 ± 0.0056 | 0.0469 ± 0.0028 | 412 | 0.418 |

Exact multi-positive random-ranking chance on this validation gallery is MRR 0.0249, Top-1 0.0047, and Top-3 0.0140. Results remain validation-only; `best_val_epoch` was chosen by validation MRR on these same examples.

## Paired comparisons and source-video uncertainty

Each prefix comparison uses the same 412 validation samples and averages reciprocal-rank changes over the three seeds. The intervals below resample the four source-video clusters, then average over all examples in each resampled draw to preserve the pooled-example estimand. With only four clusters, these intervals are exploratory and do not establish population-level uncertainty.

| Prefix pair | Paired ΔMRR | 4-video cluster bootstrap 95% interval | Top-1 gains/losses (seed-example counts) |
|---|---:|---:|---:|
| 2→4 | +0.0008 | [-0.0025, +0.0044] | 6 / 6 |
| 4→6 | +0.0025 | [-0.0076, +0.0122] | 10 / 3 |
| 6→8 | -0.0045 | [-0.0110, +0.0006] | 6 / 15 |
| 2→8 | -0.0011 | [-0.0032, +0.0016] | 10 / 12 |

The script's example-level bootstrap intervals in `paired_differences.csv` do not account for video clustering. The cluster-aware intervals above were recomputed from saved per-seed ranks and full validation cache identities; validation sample counts by video are 111, 103, 101, and 97. Exact intervals and per-video deltas are in `video_cluster_paired_differences.csv`.

## Interpretation and limitations

On this validation split, 6 frames had the highest mean MRR and Top-1, while 8 frames were lower; gains are therefore not monotonic and do not support a saturation claim. Paired differences are small relative to their uncertainty across only four validation videos. Top-1 errors remain substantial, and roughly 42–47% of the wrong predictions in the main condition are same-task confusions.

The full split's test contains one video UID previously used in pilot validation. Excluding it leaves four candidate UIDs/407 examples, listed as candidates (not a locked test) in `results/audit_v4/candidate_test_uid_manifest.csv`. The remaining UIDs had no exact references in the inspected local results, but external/unrecorded use cannot be ruled out. The candidate set is video-disjoint from the pilot, not scene-disjoint: the garden scene was already present in pilot validation. No test result is reported here.

Artifacts: `summary.csv`, `paired_differences.csv`, `video_cluster_paired_differences.csv`, `per_example_ranks.csv`, `config.json`, and `prefix_length_vs_ranking.svg`/`.png`.
