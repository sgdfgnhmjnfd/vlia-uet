# Pre-registration: baselines from the anticipation literature (written 2026-10-27, before any number)

Script: `scripts/run_literature_baselines.py`. Data: EgoIntent train/val only (27 videos); the locked
test split is never read. "Ours" = frozen F (siglip-so400m | mean | C2 | T1), no parameter selected.

Gates: `horizon` must reproduce the Stage 7 WHAT/C2 5-way accuracy (thesis Table 4.15) within 0.0006 at
H = 2, 8, 16, 24, 32, 48, 64; `next` must reproduce `results/v3_next_step/summary.csv` exactly.

## Baselines (all on the SAME 5 same-video candidates as ours, strict top-1, ties wrong)
- **blind** (text-only prior, no video): score = cos(candidate, unit mean of the label embeddings of all
  other videos).
- **copy_rec** (recognise, then assume the person keeps doing it; persistence baseline of the
  EPIC-KITCHENS / RULSTM line): the current step's label is recognised by ours (top-1 among the video's
  distinct labels, horizon: context frames; NEXT: first k frames, WHAT labels), and the candidate whose
  SigLIP text embedding is closest to the recognised label is chosen.
- **copy_true**: copy_rec with the true current label (upper bound of recognise-then-copy).
- **AntGPT-style** (Zhao et al., ICLR 2024; NEXT task only): Qwen2.5-VL-3B-Instruct, text only, sees the
  WHATs of up to 5 previous steps plus the current step (`true` labels, or recognised by ours: previous
  steps with 8 frames, current with k in {2, 8}) and picks the next step among the 5 NEXT candidates.
  Letter log-probabilities averaged over two option orders; fractional tie credit. One fixed prompt.

## Reported
Accuracy with video-bootstrap 95% CI (B = 10000, seed 0), and ours-minus-baseline paired differences.
Horizon: per H and label (WHAT, WHY), all anchors and the "changed step" subset (t+H in a later step).
"Ours beats X" at a setting = paired-difference CI lower bound > 0. Descriptive: no pass/fail verdict
on the thesis; every comparison is reported whichever way it goes.
