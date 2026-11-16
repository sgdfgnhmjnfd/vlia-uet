# Stage 12: from video to robot action

299 paired units. tau = 0.010561 on 80 calibration demos; ask rate 0.171.

| condition | successes / n | rate | Wilson 95% |
|---|---:|---:|---|
| E2E | 35 / 299 | 0.117 | [0.085, 0.158] |
| E2E-ask | 50 / 299 | 0.167 | [0.129, 0.214] |
| I-only | 52 / 299 | 0.174 | [0.135, 0.221] |
| I-wrong | 3 / 299 | 0.010 | [0.003, 0.029] |

| rule | comparison | delta [95% CI] |
|---|---|---|
| P2 | accuracy p=25 - accuracy p=0 | +0.050 [-0.040, +0.181] |
| E1 | E2E - I-wrong | +0.107 [+0.070, +0.147] |
| descriptive | E2E - I-only | -0.057 [-0.094, -0.023] |
| descriptive | E2E-ask - E2E | +0.050 [+0.027, +0.077] |

```
{
  "P1": {
    "pass": true,
    "accuracy": 0.6989966555183946,
    "ci95": [
      0.41,
      0.97
    ],
    "rule": "accuracy p=25 CI lower bound > 10%"
  },
  "P2": {
    "pass": false,
    "delta": 0.05016722408026756,
    "ci95": [
      -0.04013377926421405,
      0.18120805369127516
    ],
    "rule": "accuracy p=25 - p=0 CI lower bound > 0"
  },
  "E1": {
    "pass": true,
    "delta": 0.10702341137123746,
    "ci95": [
      0.07023411371237458,
      0.14715719063545152
    ],
    "rule": "success(E2E) - success(I-wrong) CI lower bound > 0"
  }
}
```
