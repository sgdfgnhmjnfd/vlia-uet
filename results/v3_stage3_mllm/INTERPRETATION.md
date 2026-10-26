# v3 Stage 3: MLLM multiple-choice baseline + fusion: interpretation (facts only)

Model: `Qwen/Qwen2.5-VL-3B-Instruct`, bf16. The 7B model was not used because its bf16 weights (16.58 GB) exceed the 16 GB GPU. There was no fine-tuning and no prompt search. Frames were resized to 448 px on the long side. Each query was scored with 2 option orders (canonical and reversed), averaging the letter log-probs per candidate; ties get fractional credit. All 2503 train/val queries were scored at k = 2 and k = 8 (throughput probe 3.2 s/query, ETA 2.2 h ≤ 8 h, so stride 1). The run was interrupted by OOM kills and a power cut and resumed from the per-query files. The test split was not read.

## V3: PASS (MLLM above chance at both k)

| k | MLLM 5-way | lift [95% CI] | F 5-way (same queries) | MLLM − F [95% CI] |
|---:|---:|---|---:|---|
| 2 | 0.3080 | 1.540 [1.358, 1.730] | 0.3408 | −0.0328 [−0.0600, −0.0033] |
| 8 | 0.3981 | 1.991 [1.767, 2.216] | 0.3871 | +0.0110 [−0.0260, +0.0508] |

## V4: FAIL (nested fusion improves on F at both k required)

γ was selected per fold on training-fold queries: 0.5 in 7 folds, 1.0 in 1 fold. Fusion − F, 5-way:
- k=2: +0.0012 [−0.0183, +0.0218]
- k=8: +0.0220 [+0.0008, +0.0439]

The CI lower bound is > 0 only at k=8, and the rule requires both k.

## Control (every 8th evaluated query, k=8, frames from a different scene and video)
Real-frame accuracy 0.3978 vs donor-frame accuracy 0.1166 (n=313). The control lift is 0.583 [0.418, 0.763], with the CI entirely below 1.

## Limitations
27 videos; same-video distractors may be paraphrases; one model size and one prompt (preregistered); no untouched test.
