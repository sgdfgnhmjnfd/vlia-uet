# Block 1: closed-loop C (`E2E-sup-p25-CL`), VLIA 25k, seed 1000 + e

`E2E-sup-p25-CL`: **51/299** = 17.1% (Wilson 95% 13.2–21.7)

| compared with | old | CL − old (pp) | bootstrap 95% | CL won / old won | McNemar p |
|---|---:|---:|---|---:|---:|
| built estimate `E2E-sup-p25` (52/299) | 52/299 | -0.3 | -3.0 … +2.3 | 7 / 8 | 1 |
| `E2E` (F) | 35/299 | +5.4 | +3.0 … +8.0 | 16 / 0 | 3.05e-05 |
| `I-only` (same units) | 52/299 | -0.3 | -3.0 … +2.3 | 7 / 8 | 1 |
| `I-wrong` | 3/299 | +16.1 | +11.7 … +20.4 | 51 / 3 | 2.92e-12 |

2x2 against the built estimate (units):

| | built success | built failure |
|---|---:|---:|
| CL success | 44 | 7 |
| CL failure | 8 | 240 |

Per task (successes):

| task | CL | built | I-only | E2E |
|---:|---:|---:|---:|---:|
| 0 | 0/30 | 0 | 0 | 0 |
| 1 | 0/30 | 0 | 0 | 0 |
| 2 | 7/30 | 7 | 7 | 7 |
| 3 | 20/30 | 18 | 18 | 20 |
| 4 | 0/30 | 0 | 0 | 0 |
| 5 | 18/30 | 18 | 18 | 2 |
| 6 | 1/30 | 1 | 1 | 1 |
| 7 | 0/30 | 0 | 0 | 0 |
| 8 | 0/29 | 0 | 0 | 0 |
| 9 | 5/30 | 8 | 8 | 5 |

The old built estimate stays in every table; this row is an independent supplementary measurement.
