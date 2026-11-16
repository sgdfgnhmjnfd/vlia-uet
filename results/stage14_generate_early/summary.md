# Stage 14 part A: the model writes the intention

## A3. Open-ended early intention (LIBERO-10, 299 demos, nearest of the 10 descriptions, scoring only)

| p | open-ended 10-way [95% CI] | within-scene 0 vs 1 [95% CI] | distinct sentences |
|---:|---|---|---:|
| 0 | 33.8 [30.8; 36.8] | 50.0 [50.0; 50.0] | 134 |
| 10 | 28.8 [26.1; 31.4] | 50.0 [50.0; 50.0] | 78 |
| 25 | 29.4 [26.8; 32.1] | 50.0 [50.0; 50.0] | 107 |
| 50 | 36.5 [33.1; 39.8] | 50.0 [50.0; 50.0] | 113 |
| 75 | 41.8 [38.5; 45.2] | 50.0 [50.0; 50.0] | 129 |
| 100 | 41.8 [38.8; 44.8] | 50.0 [50.0; 50.0] | 130 |

## A2. Robot (299 units)

| condition | successes / n | Wilson 95% | episodes reused / new |
|---|---:|---|---|
| E2E-gen | 6 / 299 | [0.009, 0.043] | 0 / 299 |
| E2E-gen-blank | 4 / 299 | [0.005, 0.034] | 0 / 299 |
| E2E-gen-p50 | 10 / 299 | [0.018, 0.060] | 74 / 225 |
| I-only | 52 / 299 | [0.135, 0.221] |  /  |
| Stage12 E2E | 35 / 299 | [0.085, 0.158] |  /  |
| I-wrong | 3 / 299 | [0.003, 0.029] |  /  |

| rule | comparison | delta [95% CI] |
|---|---|---|
| GA3 | E2E-gen - I-wrong | +0.010 [-0.010, +0.030] |
| descriptive | E2E-gen - I-only | -0.154 [-0.197, -0.110] |
| descriptive | E2E-gen - Stage12 E2E | -0.097 [-0.134, -0.060] |
| descriptive | E2E-gen - E2E-gen-blank | +0.007 [-0.013, +0.027] |
| descriptive | E2E-gen-p50 - E2E-gen | +0.013 [-0.007, +0.033] |
| descriptive | E2E-gen-p50 - I-wrong | +0.023 [+0.000, +0.047] |

Text match at p=25 (nearest description = true task): 29.4%.
E2E-gen successes by text match: {"0.0": {"n": 211, "E2E_gen_successes": 0}, "1.0": {"n": 88, "E2E_gen_successes": 6}}.
By tertile of cos(s, true description): [{"tertile": 0, "n": 100, "cos_true_min": 0.49300237699154115, "cos_true_max": 0.5924401282822065, "E2E_gen_successes": 0}, {"tertile": 1, "n": 100, "cos_true_min": 0.5936767539266419, "cos_true_max": 0.6684581274393371, "E2E_gen_successes": 1}, {"tertile": 2, "n": 99, "cos_true_min": 0.6684581274393371, "cos_true_max": 0.8147174719108033, "E2E_gen_successes": 5}].
Snap variant: pending ([Errno 2] No such file or directory: '/media/dhqg/d3/vlia_outputs/token_matrix/matrix_seed0.npz').

## A1. EgoIntent (development steps, open-ended, scoring vs the 5 F candidates)

| k | open-ended 5-way | MRR (same video) | F 5-way, same steps |
|---|---|---|---:|
| 8 | 31.2 [27.6; 34.9] | 0.111 [0.093; 0.132] | 38.7 |
| 2 | 28.2 [25.0; 31.8] | 0.097 [0.082; 0.113] | 34.1 |
| 4 | 29.3 [25.6; 33.2] | 0.104 [0.088; 0.121] | 34.9 |
| 8 (blank) | 18.9 |  |  |
| 2 (Stage 3 Qwen multiple choice) | 30.5 |  |  |
| 8 (Stage 3 Qwen multiple choice) | 39.6 |  |  |

Blank-control sentence (EgoIntent): 'I want to pick up the bottle from the floor.'

## Rules

| rule | value | 95% CI | verdict |
|---|---:|---|---|
| GA1: open-ended 5-way k=8 CI lower bound > 20% | +0.312 | [+0.276, +0.349] | PASS |
| GA2: video - blank, k=8, CI lower bound > 0 | +0.122 | [+0.088, +0.159] | PASS |
| GA3: success(E2E-gen) - success(I-wrong) CI lower bound > 0 (paired_bootstrap, 299 units) | +0.010 | [-0.010, +0.030] | FAIL |
| GA4: open-ended 10-way at p=100 CI lower bound > 50% | +0.418 | [+0.388, +0.448] | FAIL |
| GA5: open-ended 10-way at p=50 CI lower bound > 30% | +0.365 | [+0.331, +0.398] | PASS |
| GA6: hit(p=100) - hit(p=25) CI lower bound > 0 | +0.124 | [+0.094, +0.154] | PASS |
