# Stage 18: scene vs motion (% , 95% CI)

| dataset | split | metric | chance | S | M p25 | M p50 | S+M p25 | S+M p50 | conclusion (rule) |
|---|---|---|---:|---|---|---|---|---|---|
| LIBERO-10 (robot demos) | 5-fold by demo (Stage 15 folds) | acc | 10.0 | 91.0 [88.7, 93.4] | 99.5 [98.7, 100.0] | 98.4 [97.1, 99.5] | 99.5 [98.7, 100.0] | 99.2 [98.2, 100.0] | Scene suffices; cannot claim action understanding. |
| Own videos (shared tables) | leave-one-person-out (Stage 16, not rerun) | acc | 44.4 | 42.0 [31.8, 52.8] | 75.3 [64.9, 83.4] | 67.9 [57.1, 77.1] | 71.6 [61.0, 80.3] (F+) | 74.1 [63.6, 82.4] (F+) | Evidence the model uses action, not only scene. (Stage 16 H1-a/b/c PASS, copied) |
| EgoIntent (first-person, verb of step) | unseen videos and scenes (8 scene folds) | bal_acc | 10.0 | 11.2 [8.6, 13.7] | 11.3 [9.2, 13.9] | 9.9 [8.0, 13.1] | 11.7 [9.5, 14.2] | 10.4 [8.3, 12.9] | Insufficient evidence. |

Own videos: S+M column is Stage 16 F+ = [S, mean z_t] (not [z_1, M]); S there is the idle part before the action. Source: e41e72c:results/stage16_scene_motion/accuracy.csv (A, pooled), Stage 16 prereg sha256 4e041e2e.
EgoIntent S-vid (video identity only): 5.7 [3.6, 8.1]; majority-class accuracy 25.5.
