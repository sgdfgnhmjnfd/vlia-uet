| row | model | successes / units | rate | Wilson 95% | 25k |
|---|---|---|---|---|---|
| I-only | VLIA | 52/300 | 0.173 | [0.135, 0.220] | / |
| I-wrong | VLIA | 3/300 | 0.010 | [0.003, 0.029] | / |
| I-para | VLIA | 52/300 | 0.173 | [0.135, 0.220] | / |
| I-goal | VLIA | 44/300 | 0.147 | [0.111, 0.191] | / |
| E2E | VLIA | 35/299 | 0.117 | [0.085, 0.158] | / |
| E2E-ask | VLIA | 50/299 | 0.167 | [0.129, 0.214] | / |
| E2E-sup-p25 | VLIA | 52/299 | 0.174 | [0.135, 0.221] | / |
| E2E-gen-p25 | VLIA | 3/299 | 0.010 | [0.003, 0.029] | / |
| correct | baseline | 55/300 | 0.183 | [0.144, 0.231] | / |
| empty | baseline | 0/300 | 0.000 | [0.000, 0.013] | / |
| L-para | baseline | 1/300 | 0.003 | [0.001, 0.019] | / |
| L-goal | baseline | 1/300 | 0.003 | [0.001, 0.019] | / |
| L-E2E | baseline | 48/299 | 0.161 | [0.123, 0.206] | / |
| L-sup-p25 | baseline | 55/299 | 0.184 | [0.144, 0.232] | / |
| L-gen-p25 | baseline | 0/299 | 0.000 | [0.000, 0.013] | / |

| rule | a − b | units | diff | 95% CI | verdict |
|---|---|---|---|---|---|
| T1 | I-only − empty | 300 | +0.173 | [+0.130, +0.217] | PASS |
| T2 | I-only − I-wrong | 300 | +0.163 | [+0.120, +0.207] | PASS |
| T3 | I-goal − I-wrong | 300 | +0.137 | [+0.097, +0.177] | PASS |
| T4 | I-para − I-wrong | 300 | +0.163 | [+0.120, +0.210] | PASS |
| T5 | E2E − I-wrong | 299 | +0.107 | [+0.070, +0.147] | PASS |

| token − instruction | units | diff | 95% CI | better |
|---|---|---|---|---|
| I-para − L-para | 300 | +0.170 | [+0.127, +0.217] | token |
| I-goal − L-goal | 300 | +0.143 | [+0.103, +0.183] | token |
| E2E − L-E2E | 299 | -0.043 | [-0.084, -0.003] | instruction |
| E2E-sup-p25 − L-sup-p25 | 299 | -0.010 | [-0.057, +0.033] | no clear difference |
| E2E-gen-p25 − L-gen-p25 | 299 | +0.010 | [+0.000, +0.023] | no clear difference |

| descriptive | units | diff | 95% CI |
|---|---|---|---|
| I-only − correct | 300 | -0.010 | [-0.057, +0.037] |
| L-para − correct | 300 | -0.180 | [-0.227, -0.137] |
| L-goal − correct | 300 | -0.180 | [-0.227, -0.137] |
| E2E-ask − E2E | 299 | +0.050 | [+0.027, +0.077] |
