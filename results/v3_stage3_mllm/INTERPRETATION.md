# v3 Stage 3: MLLM multiple-choice baseline + fusion: interpretation (facts only)

Model: `Qwen/Qwen2.5-VL-3B-Instruct` (bf16, no fine-tuning, no prompt search). The 7B model was not used: its bf16 weights are 16.58 GB, more than the 16 GB GPU. Queries: all 2503 train/val queries at k = 2 and 8 (the probe measured 3.2 s/query, ETA 2.2 h ≤ 8 h). Frames were resized to a 448 px longer side. Two option orders (canonical and reversed) were averaged per candidate, with fractional tie credit. The run was interrupted (OOM kills, then a power cut) and resumed from per-(sample, k) files. Every query was scored exactly once per k. The test split was not read.

## V3: PASS
MLLM 5-way lift CI lower bound > 1 at both k: k=2, 0.308, lift 1.540 [1.358, 1.730]; k=8, 0.398, lift 1.991 [1.767, 2.216].

Paired vs F on identical queries: k=2 −0.033 [−0.060, −0.003]; k=8 +0.011 [−0.026, +0.051].

Mismatched-video control (313 queries, k=8, frames from another scene): real 0.398 vs control 0.117. The control lift is 0.583 [0.418, 0.763], with the CI below 1.

## V4: FAIL
Nested fusion s = z(F) + γ·z(MLLM), with γ chosen per outer fold (0.5 in 7 folds, 1.0 in 1). 5-way delta vs F: k=2 +0.001 [−0.018, +0.022]; k=8 +0.022 [+0.001, +0.044]. The CI lower bound is > 0 at k=8 only; the rule requires both k.

## Limitations
27 videos; same-video distractors may be paraphrases; one 3B model and a single prompt; no untouched test (the locked test is spent).
