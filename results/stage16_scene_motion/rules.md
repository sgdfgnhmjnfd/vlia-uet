# Stage 16 rules (pre-registered)

| rule | value | 95% CI | verdict |
|---|---:|---|---|
| H1-a: acc(F+, p50) - acc(S), paired, CI lower bound > 0 | +0.321 | [+0.235, +0.420] | PASS |
| H1-b: Wilson lower bound of pooled S accuracy <= pooled chance | +0.420 | [+0.318, +0.528] | PASS |
| H1-c: Wilson lower bound of pooled acc(M, p50) > pooled chance | +0.679 | [+0.571, +0.771] | PASS |
| H1-d: hand-visible at H=0.5 (share 33.3% < 60%): F+ - S | -0.037 | [-0.111, +0.000] | NOT TESTABLE (pre-registered) |
| H3-a (p25): confidence ask - random ask at the same rate, CI lower bound > 0 | -0.013 | [-0.052, +0.028] | FAIL |
| H3-b (p25): ECE after < ECE before | +0.251 | [+0.184, -] | FAIL |
| H3-a (p50): confidence ask - random ask at the same rate, CI lower bound > 0 | -0.007 | [-0.043, +0.032] | FAIL |
| H3-b (p50): ECE after < ECE before | +0.121 | [+0.177, -] | PASS |

Pooled chance (shared tables A-D, clip-weighted): 0.444.

## Main question: onset of stable recognition (pooled A-D)

| alignment | quantity | onset (s) | bootstrap 95% | never in bootstrap |
|---|---|---|---|---|
| start_s | tau_S | None | [-2.5, -1.0] | 0.991 |
| start_s | tau_M | 0.5 | [0.0, 1.0] | 0.001 |
| start_s | tau_F+ | 1.25 | [0.5, 1.75] | 0.2695 |
| start_s | tau_gain (F+ - S > 0) | 1.0 | [0.043750000000001066, 1.5] | 0.156 |
| start_s | F+ >= 80% | None | [None, None] |  |
| start_s | lead of F+ over S (s) | >= 0.75 (S never stably above chance; F+ from +1.25 s) | [None, None] |  |
| t_on | tau_S | None | [-1.0, -0.75] | 0.991 |
| t_on | tau_M | 0.5 | [0.0, 1.0] | 0.004 |
| t_on | tau_F+ | 1.0 | [0.25, 3.0] | 0.1375 |
| t_on | tau_gain (F+ - S > 0) | 1.0 | [0.0, 3.0] | 0.0425 |
| t_on | F+ >= 80% | None | [None, None] |  |
| t_on | lead of F+ over S (s) | >= 2.00 (S never stably above chance; F+ from +1.00 s) | [None, None] |  |

## Part B

```
{
 "25": {
  "acc_no_ask": 0.7160493827160493,
  "acc_confidence_ask": 0.7654320987654321,
  "ask_rate": 0.25925925925925924,
  "acc_random_ask_expected": 0.7784636488340192,
  "random_draws_mean": 0.7781111111111111,
  "random_draws_95": [
   0.7407407407407407,
   0.8148148148148148
  ],
  "H3a_diff": [
   -0.013031550068587106,
   -0.05212620027434842,
   0.02812071330589849
  ],
  "ECE_before": 0.18379381499794442,
  "ECE_after": 0.2508088831496025
 },
 "50": {
  "acc_no_ask": 0.7407407407407407,
  "acc_confidence_ask": 0.7901234567901234,
  "ask_rate": 0.2345679012345679,
  "acc_random_ask_expected": 0.7969821673525378,
  "random_draws_mean": 0.7965925925925926,
  "random_draws_95": [
   0.7530864197530864,
   0.8395061728395061
  ],
  "H3a_diff": [
   -0.006858710562414261,
   -0.043209876543209874,
   0.032235939643347054
  ],
  "ECE_before": 0.17683535590769642,
  "ECE_after": 0.12078633526294671
 },
 "stage12_margin_rule_on_untrained_F": {
  "ask_rate": 0.25925925925925924,
  "acc_no_ask": 0.49382716049382713,
  "acc_ask": 0.654320987654321
 }
}
```
