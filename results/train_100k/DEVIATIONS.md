# 100k training: deviations and notes

1. **Resume continues the data order (100k scripts differ from the 25k scripts here).**
   - **The problem.** Gate T0 on 2026-11-17 (VLIA, lab) found that a resumed run did not continue the data order. The 25k scripts build `DataLoader(shuffle=True, generator=g)` with `g` seeded 0 at every start, so after a resume the shuffle starts again from the first permutation.
     - Loss at steps 21–40 differed by up to 0.81 between "40 steps in one go" and "20 + resume + 20", while two uninterrupted runs were bit-identical.
     - The lr schedule and the per-step model RNG (`seed + step + 100000`) were already restored correctly.
   - **The fix** (Trang, option B). `vlia_data/resumable_sampler.py` wraps torch's `RandomSampler` with the same generator. Both training scripts (`train_intention_only_LIBERO7D.py`, `train_matched_oracle_LIBERO7D_FIXED.py`) use it as `sampler=` instead of `shuffle=True`. On resume, the finished epochs are replayed on indices only and the batches already used in the current epoch are skipped.
   - **Checks, all passed before the real run:**
     - (1) T0 again: max |loss difference| at steps 21–40 is 0, and lr is equal;
     - (2) `scripts/test_train100k.py`: the batch sequence after a resume at step 2,770 (just past epoch 1, which has 2,759 steps) is identical to the uninterrupted run. This also holds at steps 20, 2,759 and 5,523. The uninterrupted order is identical to the 25k `DataLoader(shuffle=True)` order;
     - (3) loss at step 1 = 3.584585, as at 25k.
   - **The 25k runs, described only and not rerun.** Each was resumed once: VLIA from step 12,000 (resume source `checkpoint_012000`) and the baseline from step 24,000. At each of those resumes the shuffle restarted from the first permutation, so the 25k data order differs from an uninterrupted run after those steps.
2. **Offline dataset load.** With `HF_HUB_OFFLINE=1`, the scripts fail when they ask the Hub for the dataset version (`list_repo_refs`). The 25k runs were online.
   - `HF_LEROBOT_HOME/lerobot/libero` (`/home/dhqg/.cache/huggingface/lerobot/lerobot/libero`) is now a symlink to the cached snapshot `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`. That is ref v3.0, the only revision in the cache. No code changed.
   - Loss at step 1 equals the 25k probe. `restore.sh` of the pack does the same on the rented machine.
3. **Different GPUs.** The baseline and VLIA train on different GPUs. Within-model comparisons are unaffected; cross-model differences include hardware nondeterminism, of the same order as a seed change.
4. **Output location.** All 100k outputs are under `/media/dhqg/d3/vlia_outputs/train_100k/`. Nothing is written to `/media/dhqg/d1`, which sits on the 95%-full `/` disk. The 25k baseline checkpoint stays where it is.
5. **No baseline 100k run (Trang, 19/11).** No GPU was rented, so the SmolVLA baseline is not trained to 100k. Item 3 above (different GPUs) no longer applies. The freeze rule "common milestone of both models" is replaced by "the last VLIA milestone actually reached before 22:00 on 22/11".
6. **Old SmolVLA run checked and excluded.** `/media/dhqg/d3/lerobot_outputs/smolvla_libero` (checkpoints 22k–50k, last resumed at 43k on 27/08) was checked on 19/11, read only, and is not used.
   - **Dataset:** it trains on `HuggingFaceVLA/libero` (all four LIBERO suites, 40 tasks), not the LIBERO-10 subset (335 episodes).
   - **Learning rate:** its cosine schedule decays over 30k steps (lr = 2.5e-6 from about 31k on) instead of over the full run.
   - **Optimizer and seed:** weight decay 1e-10 (ours 1e-4), seed 1000 (ours 0), AdamW betas (0.9, 0.95) with gradient clip 10.
   - **Frozen parts:** vision encoder frozen, expert-only training.
   - **Evaluation:** its only LIBERO-10 evaluation (`/home/dhqg/lerobot/eval_logs/eval_info.json`, 50/100) used the standard `lerobot_eval` protocol, not our units and seeds.
7. **New evaluation plan (Trang, 19/11), written before any evaluation episode.**
   - **(1) Token vs instruction, at 25k.**
     - The baseline 25k checkpoint runs L-para, L-goal, L-E2E, L-sup-p25, L-gen-p25.
     - The VLIA 25k checkpoint runs E2E-gen-p25.
     - These are paired, on the same units, with the 25k rows that already exist: I-para, I-goal, E2E, E2E-sup-p25, and the new E2E-gen-p25.
     - Reused units take the 25k `correct` / `I-only` episodes.
   - **(2) VLIA at its final milestone** (the last one reached before 22:00 on 22/11).
     - All VLIA rows are run: I-only, I-wrong, I-para, I-goal, E2E, E2E-sup-p25, E2E-gen-p25.
     - Each is compared with the same VLIA row at 25k on the same units.
     - Baseline rows (correct, empty) are taken from 25k and labelled as a different step count. T1 (I-only − empty) therefore compares VLIA at the final milestone with the baseline at 25k.
   - **(3) Learning curve:** VLIA only (I-only, 10 episodes per task at 25k / 50k / 75k).
   - **GPU timing:** only after VLIA training has finished. Training is never paused for evaluation.
8. **Harness, beyond the plan:** episode files are written atomically (tmp then rename). The VLIA token table for the evaluation is `eval_token_table.npz` (`task_embeddings_v2` + 144 G2 sentences; descriptions re-encoded with cosine 1.000000).
9. **September baseline checked and excluded (19/11).** `/media/dhqg/d3/vlia_outputs/oracle_matched_full_INVALID_6D/baseline/seed_0` (checkpoints 1k–25k, 15/09–01/10, resumed at 23k) has the same data and hyper-parameters as our baseline. Its policy action output is **6-D**, not the 7-D LIBERO action. That is the reason for the INVALID_6D name, and the run was replaced by `oracle_matched_LIBERO7D_FIXED/baseline` (7-D, 02/10–06/10). That replacement is the 25k baseline used here (correct 55/300). No other baseline run reaches more than 25k on the lab disks.
