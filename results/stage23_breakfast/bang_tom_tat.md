# Stage 23 Breakfast: summary

Balanced accuracy, pooled over 4 official splits, 95% CI by person bootstrap (10,000). Chance Part A = 0.10.
half1/half2 = first/second 1024 dims of I3D vector (no source for RGB/flow order).

## Part A: early dish prediction (from frame 0)

| row | p10 | p25 | p50 | p100 |
|---|---|---|---|---|
| S | 0.194 [0.174, 0.215] | 0.194 [0.174, 0.215] | 0.194 [0.174, 0.215] | 0.194 [0.174, 0.215] |
| M-delta | 0.219 [0.199, 0.239] | 0.356 [0.333, 0.377] | 0.495 [0.470, 0.521] | 0.537 [0.508, 0.565] |
| M-flow | 0.288 [0.268, 0.307] | 0.478 [0.451, 0.505] | 0.682 [0.658, 0.707] | 0.757 [0.734, 0.778] |
| S+M | 0.256 [0.235, 0.276] | 0.382 [0.359, 0.405] | 0.490 [0.463, 0.518] | 0.532 [0.506, 0.558] |
| T | 0.307 [0.285, 0.328] | 0.450 [0.426, 0.473] | 0.631 [0.603, 0.658] | 0.702 [0.677, 0.727] |
| H-oracle | 0.673 [0.630, 0.715] | 0.906 [0.874, 0.935] | 0.987 [0.973, 0.999] | 0.998 [0.993, 1.000] |
| H-pred | 0.258 [0.234, 0.282] | 0.422 [0.389, 0.453] | 0.524 [0.495, 0.554] | 0.567 [0.538, 0.595] |
| S+M+H-pred | 0.265 [0.244, 0.286] | 0.407 [0.386, 0.428] | 0.555 [0.526, 0.583] | 0.581 [0.552, 0.612] |

T seeds: p10: 0.299, 0.319, 0.304; p25: 0.443, 0.461, 0.445; p50: 0.630, 0.639, 0.625; p100: 0.715, 0.695, 0.696

### Part A from first non-SIL frame (descriptive)

| row | p10 | p25 | p50 | p100 |
|---|---|---|---|---|
| S | 0.246 [0.223, 0.268] | 0.246 [0.223, 0.268] | 0.246 [0.223, 0.268] | 0.246 [0.223, 0.268] |
| M-delta | 0.210 [0.191, 0.229] | 0.305 [0.286, 0.325] | 0.405 [0.383, 0.427] | 0.423 [0.398, 0.448] |
| M-flow | 0.369 [0.339, 0.397] | 0.532 [0.502, 0.560] | 0.704 [0.678, 0.729] | 0.761 [0.739, 0.783] |
| S+M | 0.275 [0.256, 0.294] | 0.385 [0.361, 0.410] | 0.455 [0.431, 0.477] | 0.463 [0.435, 0.491] |

### Checks (p25)

- S+M-shuffled: 0.114 [0.098, 0.130] n=1712
- S@onecam: 0.176 [0.142, 0.210] n=503
- M-delta@onecam: 0.308 [0.272, 0.345] n=503
- S+M@onecam: 0.340 [0.297, 0.382] n=503
- majority baseline acc (S folds): 0.107
- majority baseline acc (S+M folds): 0.107

### Hypotheses Part A (p25)

- HA-1: S = 0.194 -> < 0.80: scene alone not sufficient
- HA-2: (S+M) - S = +0.188 [+0.162, +0.213] -> PASS
- HA-3: M-delta = 0.356 [0.333, 0.377] vs 0.10 -> PASS
- HA-4: T - (S+M) = +0.068 [+0.041, +0.094] -> PASS
- (descr.) H-oracle - H-pred p10: +0.415 [+0.378, +0.453]
- (descr.) H-oracle - H-pred p25: +0.484 [+0.442, +0.528]
- (descr.) H-oracle - H-pred p50: +0.463 [+0.431, +0.495]
- (descr.) H-oracle - H-pred p100: +0.431 [+0.403, +0.460]
- (descr.) frame classifier per-second acc on test (48 cls incl SIL): raw 0.314, smoothed 0.328

## Part B: same action, different intent (balanced accuracy)

| row | pour_milk (K=4) | crack_egg (K=3) | take_plate (K=6) | butter_pan (K=3) | take_cup (K=3) | pour_oil (K=3) | MACRO |
|---|---|---|---|---|---|---|---|
| chance | 0.250 | 0.333 | 0.167 | 0.333 | 0.333 | 0.333 | 0.292 |
| n test segs | 681 | 496 | 427 | 225 | 212 | 158 | |
| SEG | 0.575 | 0.431 | 0.254 | 0.372 | 0.365 | 0.308 | 0.384 [0.359, 0.410] |
| SEG-half1 | 0.476 | 0.404 | 0.284 | 0.329 | 0.325 | 0.399 | 0.370 [0.348, 0.390] |
| SEG-half2 | 0.458 | 0.417 | 0.221 | 0.375 | 0.405 | 0.279 | 0.359 [0.332, 0.386] |
| CTX-S | 0.477 | 0.374 | 0.303 | 0.381 | 0.393 | 0.419 | 0.391 [0.370, 0.412] |
| HIST-oracle | 1.000 | 0.528 | 0.685 | 0.720 | 0.333 | 0.792 | 0.676 [0.650, 0.704] |
| HIST-pred | 0.647 | 0.398 | 0.421 | 0.536 | 0.372 | 0.459 | 0.472 [0.446, 0.497] |
| PRE | 0.727 | 0.443 | 0.481 | 0.524 | 0.368 | 0.529 | 0.512 [0.487, 0.537] |
| SEG+HIST-pred | 0.621 | 0.424 | 0.299 | 0.429 | 0.324 | 0.341 | 0.406 [0.382, 0.428] |
| SEG+CTX-S | 0.580 | 0.424 | 0.329 | 0.395 | 0.388 | 0.372 | 0.415 [0.389, 0.439] |

### Hypotheses Part B (macro over actions)

- HB-1: SEG = 0.384 [0.359, 0.410] vs chance 0.292 -> PASS
- HB-2: HIST-pred - SEG = +0.088 [+0.051, +0.125] -> PASS
- HB-3: HIST-pred - CTX-S = +0.081 [+0.050, +0.112] -> PASS
- (descr.) HIST-oracle - HIST-pred = +0.204 [+0.171, +0.240]
- (descr.) SEG-half2 - SEG-half1 = -0.010 [-0.042, +0.022]
- (descr.) SEG+HIST-pred - SEG = +0.022 [+0.005, +0.038]
- (descr.) PRE - SEG = +0.128 [+0.096, +0.160]

**Conclusion (by rule):** 1. On Breakfast, a single action is not enough to know the intent; the preceding action sequence predicts the intent better than both the action segment and the scene just before it.
