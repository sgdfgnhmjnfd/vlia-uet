# E1b v2: SmolVLM2-500M-Video-Instruct zero-shot K=5 goal scoring: interpretation (facts only)

This run uses the corrected scorer (`scripts/run_e1b_smolvlm2_v2.py`); v1 is invalid (`results/e1b_smolvlm2_v1/INVALID_20261021.md`). Scope: all 2503 train/val queries, 27 videos, prefixes k ∈ {2, 8}, canonical `build_kway` K=5 candidates, bf16 on GPU. The locked test split was not read. The run was interrupted once (machine shutdown) and resumed from the per-(sample, k) files; every query has exactly one record per k.

## Scorer checks

- For every image pass, the expanded prefix length is greater than the unexpanded tokenizer length (k=2 smoke example: 571 vs 43 tokens). The number of scored positions equals the candidate token count for every candidate in both passes (assertions in the scorer).
- Scoring a greedy generation reproduces its generation log-probability: −6.02 vs −6.09 summed over 12 tokens; the gap is bf16 batch numerics.
- Full run diagnostics: the within-query std ratio lp_null / lp_img is 1.22 (k=2) and 1.23 (k=8); v1 was 17×. PMI is > 0 for 99.4% of candidates. No query has a tied maximum PMI.

## Verdict E1b-A (primary: per-token-mean PMI, fractional tie credit)

**E1b-A: PASS.** The 5-way lift CI lower bound is > 1 at both evaluated prefixes (video bootstrap B=10000, seed 0):

| k | 5-way acc | lift [95% CI] |
|---:|---:|---|
| 2 | 0.2353 | 1.177 [1.074, 1.285] |
| 8 | 0.2461 | 1.231 [1.135, 1.333] |

Secondary scores (sum PMI, raw image log-prob mean/sum) give 5-way accuracy 0.231 to 0.254, with lift CI lower bounds > 1 for all of them (`summary.csv`).

## E1b-B (descriptive): SmolVLM2 minus SigLIP float32 baseline, identical queries

| k | SmolVLM2 (pmi_mean) | SigLIP | delta [95% CI] |
|---:|---:|---:|---|
| 2 | 0.2353 | 0.2773 | −0.0419 [−0.0753, −0.0105] |
| 8 | 0.2461 | 0.2885 | −0.0423 [−0.0746, −0.0126] |

SmolVLM2 is lower than SigLIP at both prefixes for every SmolVLM2 score. The CI excludes 0 except for `lp_img_sum` at k=2 (`paired_vs_siglip.csv`).

## Mismatched-video control (every 8th query by sample_id, k=8, donor from a different scene)

For pmi_mean (n=313), real-frame accuracy is 0.2428 and donor-frame accuracy is 0.2077. The control lift is 1.038 [0.794, 1.296], so the CI includes chance. All four scores show a control CI that includes 1 (`control.csv`).

## Limitations

- 27 source videos; same-video distractors may be paraphrases; no untouched test.
- Only k=2 and k=8 were evaluated (the prereg-specified scale for this redo).
- The null prompt contains k in its wording, so the null score is cached per (candidate text, k), not per text alone.
- Candidates are tokenized with a leading space after `Assistant:`. Absolute log-probs are low (the model prefers answers of the form "The person's goal is …"), which affects the raw scores but not the within-query ranking logic.
- This addendum's thresholds were reconstructed after the original E1b started (`results/prereg_daily_v1_addendum.md`). The E1b-A rule applied here is the one stated in the 2026-10-21 task, fixed before any v2 number existed.
