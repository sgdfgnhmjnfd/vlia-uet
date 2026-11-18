# Stage 17 Step 2: exploratory (outside the prereg; config (ii) from G0b, D-E 55.6% < 60%)

R0 (Stage 16 reproduction, pooled): S 0.420, M p50 0.679, F+ p50 0.741 -> PASS

| rule | value | 95% CI | verdict |
|---|---:|---|---|
| H4-a: F+H - F+ at p25 (pooled, paired) | +0.099 | [+0.000, +0.198] | FAIL |
| H4-b: H at p50 > chance 0.444 (Wilson lower bound) | 0.753 | [0.649, 0.834] | PASS |
| H4-c (stable-recognition time) | not run (Trang's Step 2 scope: p25 and p50 only) | | |

## H4-c (exploratory, run 18/11 after Step 2)

| rule | value | 95% CI | verdict |
|---|---:|---|---|
| H4-c: median per-clip tau(F+H) - tau(F+), pooled A-D | +0.00 s | [+0.00, +0.00] | FAIL (upper bound < 0 required) |
