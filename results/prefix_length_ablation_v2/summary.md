# Prefix-length goal-disambiguation ablation (validation only)

Seeds: [0, 1, 2]; epochs: 100; gallery = all validation WHY targets; positives = rows with identical WHY text. Values are mean ± sample std over seeds. Chance (exact, multi-positive): MRR 0.0422, Top-1 0.0088, Top-3 0.0261.

## train_k_eval_k / best_val_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0551 ± 0.0035 | 0.0310 ± 0.0027 | 0.0326 ± 0.0047 | 215 | 0.874 | 20 / 546 / 79 |
| 4 | 0.0545 ± 0.0039 | 0.0279 ± 0.0047 | 0.0357 ± 0.0027 | 215 | 0.810 | 18 / 508 / 119 |
| 6 | 0.0604 ± 0.0047 | 0.0310 ± 0.0071 | 0.0481 ± 0.0027 | 215 | 0.756 | 20 / 472 / 153 |
| 8 | 0.0589 ± 0.0016 | 0.0295 ± 0.0071 | 0.0419 ± 0.0047 | 215 | 0.814 | 19 / 510 / 116 |

## train_k_eval_k / last_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0360 ± 0.0042 | 0.0093 ± 0.0047 | 0.0186 ± 0.0047 | 215 | 0.495 | 6 / 316 / 323 |
| 4 | 0.0379 ± 0.0092 | 0.0093 ± 0.0093 | 0.0186 ± 0.0047 | 215 | 0.432 | 6 / 276 / 363 |
| 6 | 0.0434 ± 0.0023 | 0.0109 ± 0.0027 | 0.0326 ± 0.0000 | 215 | 0.574 | 7 / 366 / 272 |
| 8 | 0.0495 ± 0.0036 | 0.0171 ± 0.0054 | 0.0326 ± 0.0093 | 215 | 0.773 | 11 / 490 / 144 |

## train_full_eval_k / best_val_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0521 ± 0.0030 | 0.0233 ± 0.0047 | 0.0326 ± 0.0047 | 215 | 0.781 | 15 / 492 / 138 |
| 4 | 0.0550 ± 0.0022 | 0.0264 ± 0.0027 | 0.0357 ± 0.0027 | 215 | 0.771 | 17 / 484 / 144 |
| 6 | 0.0556 ± 0.0070 | 0.0248 ± 0.0071 | 0.0388 ± 0.0054 | 215 | 0.783 | 16 / 492 / 137 |
| 8 | 0.0589 ± 0.0016 | 0.0295 ± 0.0071 | 0.0419 ± 0.0047 | 215 | 0.814 | 19 / 510 / 116 |

## train_full_eval_k / last_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0447 ± 0.0036 | 0.0155 ± 0.0071 | 0.0248 ± 0.0071 | 215 | 0.747 | 10 / 475 / 160 |
| 4 | 0.0446 ± 0.0038 | 0.0140 ± 0.0081 | 0.0264 ± 0.0054 | 215 | 0.764 | 9 / 486 / 150 |
| 6 | 0.0471 ± 0.0034 | 0.0155 ± 0.0071 | 0.0279 ± 0.0047 | 215 | 0.763 | 10 / 485 / 150 |
| 8 | 0.0495 ± 0.0036 | 0.0171 ± 0.0054 | 0.0326 ± 0.0093 | 215 | 0.773 | 11 / 490 / 144 |

## Paired differences (to_k minus from_k, same examples)

Example-level bootstrap CI on the seed-averaged reciprocal rank. It ignores clustering by source video (the validation set has few videos), so it understates uncertainty: descriptive only.

| setting / selection | from→to | ΔMRR [95% CI] | seeds with Δ>0 | Top-1 gains / losses (pooled) |
|---|---|---|---:|---|
| train_k_eval_k / best_val_epoch | 2→4 | -0.0005 [-0.0102, +0.0090] | 1/3 | 4 / 6 |
| train_k_eval_k / best_val_epoch | 4→6 | +0.0058 [-0.0055, +0.0169] | 2/3 | 9 / 7 |
| train_k_eval_k / best_val_epoch | 6→8 | -0.0015 [-0.0154, +0.0124] | 2/3 | 12 / 13 |
| train_k_eval_k / best_val_epoch | 2→8 | +0.0038 [-0.0136, +0.0208] | 2/3 | 12 / 13 |
| train_k_eval_k / last_epoch | 2→4 | +0.0019 [-0.0104, +0.0134] | 2/3 | 6 / 6 |
| train_k_eval_k / last_epoch | 4→6 | +0.0056 [-0.0019, +0.0135] | 2/3 | 4 / 3 |
| train_k_eval_k / last_epoch | 6→8 | +0.0061 [-0.0031, +0.0165] | 3/3 | 8 / 4 |
| train_k_eval_k / last_epoch | 2→8 | +0.0135 [-0.0023, +0.0290] | 3/3 | 10 / 5 |
| train_full_eval_k / best_val_epoch | 2→4 | +0.0029 [+0.0002, +0.0061] | 3/3 | 2 / 0 |
| train_full_eval_k / best_val_epoch | 4→6 | +0.0006 [-0.0030, +0.0041] | 2/3 | 1 / 2 |
| train_full_eval_k / best_val_epoch | 6→8 | +0.0033 [-0.0011, +0.0081] | 2/3 | 5 / 2 |
| train_full_eval_k / best_val_epoch | 2→8 | +0.0068 [+0.0008, +0.0136] | 3/3 | 5 / 1 |
| train_full_eval_k / last_epoch | 2→4 | -0.0001 [-0.0038, +0.0040] | 2/3 | 1 / 2 |
| train_full_eval_k / last_epoch | 4→6 | +0.0024 [-0.0006, +0.0062] | 3/3 | 1 / 0 |
| train_full_eval_k / last_epoch | 6→8 | +0.0024 [-0.0009, +0.0065] | 3/3 | 2 / 1 |
| train_full_eval_k / last_epoch | 2→8 | +0.0048 [-0.0017, +0.0114] | 3/3 | 4 / 3 |

## Limitations

- Validation-only on a small pilot split (no untouched test set); no significance or generalisation claim.
- `best_val_epoch` selects on the same validation set it reports on (optimistic); `last_epoch` does not.
- Prefix k = first k of the 8 cached frames, sampled uniformly over the whole clip; its share of wall-clock time depends on clip length.
- `train_full_eval_k` is a train/test length-mismatch diagnostic, not the main comparison.
- Confidence/margin/entropy decision analysis is a separate step.
