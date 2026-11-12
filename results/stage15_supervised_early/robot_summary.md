# Stage 15 robot: VLIA with the supervised label

299 paired units (Stage 12 amendment 1).

| condition | successes / n | rate | Wilson 95% | episodes reused / new |
|---|---:|---:|---|---|
| E2E-sup-p25 | 52 / 299 | 0.174 | [0.135, 0.221] | 298 / 1 |
| E2E-sup-p50 | 52 / 299 | 0.174 | [0.135, 0.221] | 299 / 0 |
| I-only | 52 / 299 | 0.174 | [0.135, 0.221] |  /  |
| Stage12 E2E | 35 / 299 | 0.117 | [0.085, 0.158] |  /  |
| I-wrong | 3 / 299 | 0.010 | [0.003, 0.029] |  /  |

| rule | comparison | delta [95% CI] |
|---|---|---|
| descriptive | E2E-sup-p25 - I-only | +0.000 [+0.000, +0.000] |
| descriptive | E2E-sup-p25 - Stage12 E2E | +0.057 [+0.023, +0.094] |
| R4 | E2E-sup-p25 - I-wrong | +0.164 [+0.120, +0.207] |
| descriptive | E2E-sup-p50 - I-only | +0.000 [+0.000, +0.000] |
| descriptive | E2E-sup-p50 - Stage12 E2E | +0.057 [+0.023, +0.094] |
| descriptive | E2E-sup-p50 - I-wrong | +0.164 [+0.120, +0.207] |
