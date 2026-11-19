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
