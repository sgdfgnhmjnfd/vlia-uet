# E1 zero-shot SigLIP alignment

Train/validation-only, scene-grouped 8-fold evaluation. MRR lift intervals bootstrap source videos; these are descriptive.

| Branch | k | MRR | MRR chance | MRR lift [95% CI] | Top-1 | Top-3 | 5-way acc. | 5-way lift [95% CI] | 10-way acc. |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| full | 2 | 0.0671 | 0.02977 | 2.25 [1.87, 2.75] | 0.0172 | 0.0479 | 0.2773 | 1.39 [1.23, 1.56] | 0.1570 |
| full | 4 | 0.0715 | 0.02977 | 2.40 [2.02, 2.88] | 0.0212 | 0.0563 | 0.2753 | 1.38 [1.22, 1.55] | 0.1614 |
| full | 6 | 0.0734 | 0.02977 | 2.47 [2.06, 2.97] | 0.0220 | 0.0571 | 0.2809 | 1.40 [1.24, 1.58] | 0.1674 |
| full | 8 | 0.0765 | 0.02977 | 2.57 [2.13, 3.14] | 0.0236 | 0.0595 | 0.2885 | 1.44 [1.27, 1.62] | 0.1774 |

## Paired differences

| Comparison | Metric | From -> to | Delta [95% CI] |
|---|---|---|---|
| prefix full - full | mrr | 2 -> 4 | +0.00442 [+0.00145, +0.00743] |
| prefix full - full | kway5 | 2 -> 4 | -0.00200 [-0.00820, +0.00405] |
| prefix full - full | mrr | 4 -> 6 | +0.00187 [-0.00044, +0.00377] |
| prefix full - full | kway5 | 4 -> 6 | +0.00559 [-0.00040, +0.01148] |
| prefix full - full | mrr | 6 -> 8 | +0.00314 [+0.00023, +0.00599] |
| prefix full - full | kway5 | 6 -> 8 | +0.00759 [+0.00000, +0.01557] |
| prefix full - full | mrr | 2 -> 8 | +0.00943 [+0.00528, +0.01415] |
| prefix full - full | kway5 | 2 -> 8 | +0.01119 [+0.00139, +0.02114] |
