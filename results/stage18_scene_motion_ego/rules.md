# Stage 18 rules (pre-registered)

| dataset | rule | value | 95% CI | verdict |
|---|---|---|---|---|
| EgoIntent | R1 S >= 0.80 | 0.112 | | does not hold |
| EgoIntent | R2 (S+M p50) - S, CI lo > 0 | -0.009 | [-0.034, +0.021] | FAIL |
| EgoIntent | R3 M p50 CI lo > 1/K | 0.099 | lo 0.080 | FAIL |
| LIBERO-10 | R1 S >= 0.80 | 0.910 | | HOLDS |
| LIBERO-10 | R2 (S+M p50) - S, CI lo > 0 | +0.082 | [+0.058, +0.106] | PASS |
| LIBERO-10 | R3 M p50 CI lo > 1/K | 0.984 | lo 0.971 | PASS |
| Own videos | R1 S >= 0.80 | 0.420 | | does not hold |

- EgoIntent: Insufficient evidence.
- LIBERO-10: Scene suffices; cannot claim action understanding.
- Own videos: Evidence the model uses action, not only scene. (Stage 16 H1-a/b/c PASS, copied)

EgoIntent M p50 recall per class: hold 0.36, place 0.18, pick_up 0.14, grasp 0.11, align 0.06, wipe 0.04, adjust 0.03, position 0.03, walk 0.02, remove 0.02
