# Pre-registration: VLIA seed 1 (2026-11-10)

The plan was written on 2026-10-31 and is copied verbatim below. It was committed before the seed-1 training and before any seed-1 evaluation. Nothing else changes from the seed-0 run (Stage 4).

## Training

- Same command as Stage 4 in `scripts/run_v3_chain.sh`, with two changes:
  - `--seed 1`;
  - `--save-freq 5000`, which saves disk space and does not change training.
- Steps 25,000, batch size 32, output root `$OUT/train`. The checkpoint lands in `$OUT/train/intention_only/seed_1/`.
- Resume from the latest checkpoint after an interruption, like the retry loop in `run_v3_chain.sh`.
- Use the same token table (`task_embeddings_v2.npz` for evaluation) and the same processors (`copy_processors`).

## Evaluation

The checkpoint is `seed_1/checkpoint_025000/pretrained_model`. Output goes to `$OUT/eval_seed1`, with an empty instruction, e = 0..29 (`--episodes-per-task 30`), on all 10 tasks. Episode e uses `init_state_id = e` and seed `1000 + e`, as before.

| condition | token | units |
|---|---|---|
| `I-only` | correct task description | 300 |
| `I-wrong` | description of task (t+5) mod 10 | 300 |
| `E2E` | the Stage 12 predictions, `results/stage12_end_to_end/e2e_tokens.json` | 299 |

## Pre-registered rules

Paired bootstrap over units (`paired_bootstrap`, B = 10,000, seed 0).

- **S1 (the intention token works in a second training run).** Seed-1 `I-only` − seed-1 `I-wrong` has CI lower bound > 0.
- **S2 (the end-to-end loop works in a second training run).** Seed-1 `E2E` − seed-1 `I-wrong`, over the 299 units, has CI lower bound > 0.
- **Descriptive**, no pass/fail:
  - successes and Wilson CI for each condition and seed;
  - seed-1 minus seed-0 for each condition, paired over the same units;
  - the two-seed pooled rate for each condition, with a bootstrap that resamples units and keeps both seeds of a unit together;
  - per-task successes for both seeds.

## Gate S0 (before the full run)

Run training for 2 steps with seed 1 into `/media/dhqg/d3/tmp/seed1_check`, and do the same with seed 0. Show that the two runs differ: a hash of the newly initialised adapter weights, or the first batch indices. If they are identical, the seed does nothing; stop and report.

Implementation detail fixed here:
- Each check run uses `--steps 2 --batch-size 32 --save-freq 100000 --log-freq 1`; the script saves at the last step.
- The evidence is the SHA-256 of the `intention_adapter` tensors in each `checkpoint_000002/pretrained_model/model.safetensors`, plus the logged step-1 loss.
- Seed 0 is run twice. If its two runs agree while seed 1 differs, the difference comes from the seed and not from GPU non-determinism.

## Disk check

seed_0 holds 25 checkpoints in 32 GB, i.e. 1.3 GB per checkpoint. 5 checkpoints ≈ 6.5 GB, and 1.5 × that ≈ 10 GB. `/media/dhqg/d3` has 1.5 TB free, so the check passes.
