# Stage 19 rules (pre-registered)

| dataset | rule | value | 95% CI | verdict |
|---|---|---|---|---|
| own | R1 S >= 0.80 | | | does not hold |
| own | R2 (S+M p50) - S | +0.321 | [+0.198, +0.444] | PASS |
| own | R3 M p50 > chance (0.444) | 0.704 | lo 0.617 | PASS |
| own | R4 B S+M p50 - Stage 18/16 S+M p50 | +0.012 | [-0.062, +0.086] | FAIL |
| EgoIntent | R1 S >= 0.80 | | | does not hold |
| EgoIntent | R2 (S+M p50) - S | +0.021 | [-0.004, +0.044] | FAIL |
| EgoIntent | R3 M p50 > chance (0.100) | 0.119 | lo 0.097 | FAIL |
| EgoIntent | R4 B S+M p50 - Stage 18/16 S+M p50 | +0.015 | [-0.009, +0.037] | FAIL |

- own: Evidence the model uses action, not only scene.
- EgoIntent: Insufficient evidence.
