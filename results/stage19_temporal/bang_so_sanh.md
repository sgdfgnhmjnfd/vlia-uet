# Stage 19 vs Stage 18: scene / motion / both (%, 95% CI)

| dataset | features | metric | chance | S | M p50 | S+M p50 | M p100 | S+M p100 |
|---|---|---|---:|---|---|---|---|---|
| own videos | Stage 16 SigLIP per frame (S+M = F+) | acc | 44.4 | 42.0 [31.8, 52.8] | 67.9 [57.1, 77.1] | 74.1 [63.6, 82.4] | 71.6 [61.0, 80.3] | 72.8 [62.3, 81.3] |
| own videos | Stage 19 B VideoMAE-SSv2 | acc | 44.4 | 43.2 [33.3, 54.3] | 70.4 [61.7, 79.0] | 75.3 [67.9, 82.7] | 91.4 [85.2, 96.3] | 91.4 [85.2, 96.3] |
| EgoIntent | Stage 18 SigLIP per frame | bal_acc | 10.0 | 11.2 [8.6, 13.7] | 9.9 [8.0, 13.1] | 10.4 [8.3, 12.9] | 10.2 [9.1, 11.9] | 12.5 [9.9, 15.5] |
| EgoIntent | Stage 19 B VideoMAE-SSv2 | bal_acc | 10.0 | 9.8 [7.8, 12.2] | 11.9 [9.7, 14.1] | 11.9 [9.9, 13.8] | 17.1 [14.4, 20.8] | 17.8 [14.0, 22.5] |

Own videos: Stage 16 S = mean of idle frames; Stage 19 S = first frame repeated 16x (static clip).
- own: Evidence the model uses action, not only scene.
- EgoIntent: Insufficient evidence.
- shuffled-label check (own, S+M p50): 38.3
- shuffled-label check (EgoIntent, S+M p50): 9.6
