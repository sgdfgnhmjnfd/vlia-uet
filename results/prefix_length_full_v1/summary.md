# Prefix-length goal-disambiguation ablation (validation only)

Seeds: [0, 1, 2]; epochs: 100; gallery = all validation WHY targets; positives = rows with identical WHY text. Values are mean ± sample std over seeds. Chance (exact, multi-positive): MRR 0.0249, Top-1 0.0047, Top-3 0.0140.

Validation contains 412 examples from 4 source-video UIDs. No test features were loaded and no test score is reported.

## train_k_eval_k / best_val_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0440 ± 0.0064 | 0.0162 ± 0.0037 | 0.0356 ± 0.0109 | 412 | 0.470 | 20 / 571 / 645 |
| 4 | 0.0448 ± 0.0072 | 0.0162 ± 0.0074 | 0.0364 ± 0.0106 | 412 | 0.471 | 20 / 573 / 643 |
| 6 | 0.0474 ± 0.0010 | 0.0218 ± 0.0024 | 0.0413 ± 0.0042 | 412 | 0.460 | 27 / 556 / 653 |
| 8 | 0.0429 ± 0.0053 | 0.0146 ± 0.0073 | 0.0380 ± 0.0056 | 412 | 0.418 | 18 / 509 / 709 |

## train_k_eval_k / last_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0319 ± 0.0020 | 0.0057 ± 0.0014 | 0.0186 ± 0.0014 | 412 | 0.492 | 7 / 604 / 625 |
| 4 | 0.0279 ± 0.0042 | 0.0024 ± 0.0024 | 0.0138 ± 0.0056 | 412 | 0.459 | 3 / 566 / 667 |
| 6 | 0.0312 ± 0.0020 | 0.0049 ± 0.0024 | 0.0162 ± 0.0037 | 412 | 0.509 | 6 / 626 / 604 |
| 8 | 0.0307 ± 0.0025 | 0.0032 ± 0.0014 | 0.0186 ± 0.0051 | 412 | 0.445 | 4 / 548 / 684 |

## train_full_eval_k / best_val_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0413 ± 0.0058 | 0.0146 ± 0.0106 | 0.0364 ± 0.0000 | 412 | 0.424 | 18 / 516 / 702 |
| 4 | 0.0413 ± 0.0072 | 0.0146 ± 0.0097 | 0.0340 ± 0.0042 | 412 | 0.422 | 18 / 513 / 705 |
| 6 | 0.0417 ± 0.0074 | 0.0138 ± 0.0109 | 0.0348 ± 0.0037 | 412 | 0.419 | 17 / 510 / 709 |
| 8 | 0.0429 ± 0.0053 | 0.0146 ± 0.0073 | 0.0380 ± 0.0056 | 412 | 0.418 | 18 / 509 / 709 |

## train_full_eval_k / last_epoch

| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |
|---:|---|---|---|---:|---:|---|
| 2 | 0.0261 ± 0.0022 | 0.0016 ± 0.0014 | 0.0097 ± 0.0024 | 412 | 0.439 | 2 / 542 / 692 |
| 4 | 0.0268 ± 0.0007 | 0.0016 ± 0.0014 | 0.0097 ± 0.0024 | 412 | 0.448 | 2 / 553 / 681 |
| 6 | 0.0299 ± 0.0022 | 0.0049 ± 0.0024 | 0.0146 ± 0.0024 | 412 | 0.445 | 6 / 548 / 682 |
| 8 | 0.0307 ± 0.0025 | 0.0032 ± 0.0014 | 0.0186 ± 0.0051 | 412 | 0.445 | 4 / 548 / 684 |

## Paired differences (to_k minus from_k, same examples)

Example-level bootstrap CI on the seed-averaged reciprocal rank. It ignores clustering by source video (the validation set has few videos), so it understates uncertainty: descriptive only.

| setting / selection | from→to | ΔMRR [95% CI] | seeds with Δ>0 | Top-1 gains / losses (pooled) |
|---|---|---|---:|---|
| train_k_eval_k / best_val_epoch | 2→4 | +0.0008 [-0.0037, +0.0055] | 2/3 | 6 / 6 |
| train_k_eval_k / best_val_epoch | 4→6 | +0.0025 [-0.0027, +0.0082] | 2/3 | 10 / 3 |
| train_k_eval_k / best_val_epoch | 6→8 | -0.0045 [-0.0099, +0.0006] | 1/3 | 6 / 15 |
| train_k_eval_k / best_val_epoch | 2→8 | -0.0011 [-0.0075, +0.0058] | 2/3 | 10 / 12 |
| train_k_eval_k / last_epoch | 2→4 | -0.0040 [-0.0087, +0.0008] | 0/3 | 2 / 6 |
| train_k_eval_k / last_epoch | 4→6 | +0.0033 [-0.0009, +0.0076] | 2/3 | 4 / 1 |
| train_k_eval_k / last_epoch | 6→8 | -0.0004 [-0.0032, +0.0024] | 1/3 | 0 / 2 |
| train_k_eval_k / last_epoch | 2→8 | -0.0011 [-0.0067, +0.0043] | 1/3 | 4 / 7 |
| train_full_eval_k / best_val_epoch | 2→4 | -0.0000 [-0.0024, +0.0022] | 1/3 | 2 / 2 |
| train_full_eval_k / best_val_epoch | 4→6 | +0.0004 [-0.0021, +0.0031] | 3/3 | 1 / 2 |
| train_full_eval_k / best_val_epoch | 6→8 | +0.0011 [-0.0012, +0.0039] | 2/3 | 2 / 1 |
| train_full_eval_k / best_val_epoch | 2→8 | +0.0015 [-0.0030, +0.0064] | 3/3 | 4 / 4 |
| train_full_eval_k / last_epoch | 2→4 | +0.0008 [-0.0014, +0.0036] | 2/3 | 1 / 1 |
| train_full_eval_k / last_epoch | 4→6 | +0.0031 [+0.0003, +0.0065] | 3/3 | 4 / 0 |
| train_full_eval_k / last_epoch | 6→8 | +0.0008 [-0.0017, +0.0031] | 3/3 | 0 / 2 |
| train_full_eval_k / last_epoch | 2→8 | +0.0046 [+0.0003, +0.0097] | 3/3 | 4 / 2 |

## Limitations

- Validation-only; no significance or generalisation claim. Best-epoch selection and reporting use the same validation split.
- `best_val_epoch` selects on the same validation set it reports on (optimistic); `last_epoch` does not.
- Prefix k = first k of the 8 cached frames, sampled uniformly over the whole clip; its share of wall-clock time depends on clip length.
- `train_full_eval_k` is a training/evaluation length-mismatch diagnostic, not the main comparison.
- Confidence/margin/entropy decision analysis is a separate step.
