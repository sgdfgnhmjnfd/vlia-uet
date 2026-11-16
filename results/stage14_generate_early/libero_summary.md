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
