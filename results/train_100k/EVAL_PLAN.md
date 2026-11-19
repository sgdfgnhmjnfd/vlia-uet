# Final evaluation plan (rewritten 19/11/2026, Trang: no rented GPU, no baseline 100k)

**Rules:**
- GPU only after VLIA training has finished. Training is never paused for evaluation; `run_final_eval_100k.sh` refuses while a process is on the GPU.
- Reused units (`unit_sentences.json`) are not rolled out; they take the 25k `correct` (baseline) or `I-only` (VLIA) episode of the same unit.
- **Final VLIA milestone:** the last checkpoint reached before 22:00 on 30/11 (freeze moved from 22/11 on 19/11) (100k if training finishes, otherwise 75k / 50k). It is chosen by this rule only.

## (1) Token vs instruction at 25k

| checkpoint | rows | episodes |
|---|---|---:|
| baseline `checkpoint_025000` | L-para 300, L-goal 300, L-E2E 90, L-sup-p25 1, L-gen-p25 299 | 990 (≈ 2.8 h) |
| VLIA `checkpoint_025000` | E2E-gen-p25 299 | 299 (≈ 0.8 h) |

Pairs are on the same units, with the 25k rows that already exist:

| token row (VLIA 25k) | instruction row (baseline 25k) | units |
|---|---|---:|
| I-para | L-para | 300 |
| I-goal | L-goal | 300 |
| E2E | L-E2E | 299 |
| E2E-sup-p25 | L-sup-p25 | 299 |
| E2E-gen-p25 (new) | L-gen-p25 | 299 |

## (2) VLIA at the final milestone

| rows | episodes |
|---|---:|
| I-only 300, I-wrong 300, I-para 300, I-goal 300, E2E 299, E2E-sup-p25 1, E2E-gen-p25 299 | 1,799 (≈ 5.0 h) |

- Each row is compared with the same VLIA row at 25k, on the same units (paired bootstrap).
- Rules T1–T5 are computed at the final milestone. T1 uses the baseline `empty` row at 25k, labelled "baseline 25k" because the step count differs.
- `correct` / `empty` are not rerun.

## (3) Learning curve (VLIA only)

- I-only, 10 episodes per task at 50k and 75k: 200 episodes (≈ 0.6 h).
- The 25k point is the existing I-only e0–9 (20/100); the final point is I-only e0–9 from (2).

## Schedule (times, not results)

- **VLIA training:** 100k at about 04:00 on 21/11 if run continuously (1.85 s/step). With about 10 h off each night, about 10:00 on 22/11.
- **Evaluation:** about 9.2 h of GPU time in all, in this order: (1) baseline 25k, (1) VLIA 25k, (2) VLIA final, (3) curve. If training finishes at about 04:00 on 21/11, the tables are ready at about 13:30 on 21/11.
- **If time runs short:** stop only at the end of a row and cut from the end of the order, never by results.

## Commands (run only after Trang says go)

```
cd /media/dhqg/d3/vlia-100k
bash scripts/run_final_eval_100k.sh baseline /media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000 L-para,L-goal,L-E2E,L-sup-p25,L-gen-p25
bash scripts/run_final_eval_100k.sh vlia /media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000 E2E-gen-p25
bash scripts/run_final_eval_100k.sh vlia /media/dhqg/d3/vlia_outputs/train_100k/vlia/intention_only/seed_0/checkpoint_<FINAL>
bash scripts/run_final_eval_100k.sh vlia /media/dhqg/d3/vlia_outputs/train_100k/vlia/intention_only/seed_0/checkpoint_050000 --curve
bash scripts/run_final_eval_100k.sh vlia /media/dhqg/d3/vlia_outputs/train_100k/vlia/intention_only/seed_0/checkpoint_075000 --curve
```

**Analysis:** `scripts/analyze_train100k.py`
- (1): `--vlia` = 25k VLIA episodes + `eval/vlia_25000`; `--base` = 25k baseline episodes + `eval/baseline_25000`.
- (2): `--vlia eval/vlia_<FINAL>`, `--vlia-25k` = the 25k folders, `--base` = 25k baseline episodes (labelled as 25k).

## Update 19/11 10:05 (Trang): submission deadline 6/12, baseline 100k only if time remains

- **Baseline 25k is kept** for all comparisons above (1)–(3).
- **VLIA 100k continues.**
- **(4) Optional, only after VLIA 100k and (1)–(3) are done:** train the SmolVLA baseline to 100k on the lab GPU, then evaluate it.
  - The run is started fresh, `train_matched_oracle_LIBERO7D_FIXED.py --condition baseline`, same arguments as VLIA 100k, with resumable data order. The run is preceded by gate T0 for the baseline (about 5 min of GPU).
  - Baseline rows at 100k: correct, empty, L-para, L-goal, L-E2E, L-sup-p25, L-gen-p25 (1,590 episodes, about 4.4 h). Baseline learning curve: `correct` at 25k / 50k / 75k (about 1 h).
  - After that, T1 and the token-vs-instruction pairs can also be reported at 100k vs 100k.
- **Schedule (continuous running, 1.85 s/step):**

  | step | when |
  |---|---|
  | VLIA 100k done | about 04:00 on 21/11 |
  | evaluations (1)–(3) | about 04:00–13:30 on 21/11 |
  | baseline T0 + 100k training (51 h) | about 13:30 on 21/11 → 16:30 on 23/11 |
  | baseline 100k evaluation + curve | about 23:00 on 23/11 |

  With about 10 h off each night, baseline training ends about 25/11 and its evaluation about 26/11. Both are well before 6/12.
- **Needs Trang's decision first:** `run_train100k_chain.sh` has the freeze hard-coded at 22:00 on 22/11. A baseline run started on 21/11 would be stopped by it, so the freeze must be moved (proposed: 22:00 on 30/11) before the baseline is started.
