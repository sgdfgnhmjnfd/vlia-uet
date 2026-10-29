# v3 Stage 3: MLLM multiple-choice baseline (Qwen/Qwen2.5-VL-3B-Instruct) + fusion

Evaluated queries: 2503 (stride 1). Video bootstrap B=10000, seed 0.

| method | k | n | 5-way acc | lift [95% CI] | F 5-way (same queries) |
|---|---:|---:|---:|---|---:|
| MLLM | 2 | 2503 | 0.3080 | 1.540 [1.358, 1.730] | 0.3408 |
| MLLM | 8 | 2503 | 0.3981 | 1.991 [1.767, 2.216] | 0.3871 |
| fusion(F + gamma*MLLM) | 2 | 2503 | 0.3420 | — | 0.3408 |
| fusion(F + gamma*MLLM) | 8 | 2503 | 0.4091 | — | 0.3871 |

| comparison | k | 5-way delta [95% CI] |
|---|---:|---|
| MLLM - F | 2 | -0.0328 [-0.0600, -0.0033] |
| MLLM - F | 8 | +0.0110 [-0.0260, +0.0508] |
| fusion - F | 2 | +0.0012 [-0.0183, +0.0218] |
| fusion - F | 8 | +0.0220 [+0.0008, +0.0439] |

Control (k=8, n=313): real 0.3978, control 0.1166, control lift 0.583 [0.418, 0.763]

Selected gamma per fold: 0:1.0, 1:0.5, 2:0.5, 3:0.5, 4:0.5, 5:0.5, 6:0.5, 7:0.5

```
{
  "V3": {
    "pass": true,
    "prefixes_lift_ci_lo_gt1": [
      2,
      8
    ],
    "rule": "MLLM 5-way lift CI lower bound > 1 at both k (fractional tie credit)"
  },
  "V4": {
    "pass": false,
    "prefixes_5way_delta_ci_lo_gt0": [
      8
    ],
    "rule": "nested fusion: 5-way delta vs F CI lower bound > 0 at both k"
  }
}
```
