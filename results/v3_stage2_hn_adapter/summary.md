# v3 Stage 2: hard-negative residual adapter (mean over 3 seeds)

| pipeline | k | MRR | 5-way | 10-way |
|---|---:|---:|---:|---:|
| F | 2 | 0.0747 | 0.3408 | 0.2085 |
| F | 4 | 0.0777 | 0.3492 | 0.2101 |
| F | 6 | 0.0825 | 0.3624 | 0.2189 |
| F | 8 | 0.0850 | 0.3871 | 0.2349 |
| hn_adapter | 2 | 0.0585 | 0.3194 | 0.1677 |
| hn_adapter | 4 | 0.0612 | 0.3296 | 0.1745 |
| hn_adapter | 6 | 0.0648 | 0.3383 | 0.1863 |
| hn_adapter | 8 | 0.0670 | 0.3471 | 0.1955 |
| random_neg_adapter_control | 2 | 0.0591 | 0.3156 | 0.1733 |
| random_neg_adapter_control | 4 | 0.0600 | 0.3235 | 0.1711 |
| random_neg_adapter_control | 6 | 0.0625 | 0.3364 | 0.1815 |
| random_neg_adapter_control | 8 | 0.0637 | 0.3497 | 0.1911 |

| comparison | k | metric | delta [95% CI] |
|---|---:|---|---|
| hn_adapter - F | 2 | MRR | -0.0162 [-0.0215, -0.0110] |
| hn_adapter - F | 2 | kway5 | -0.0214 [-0.0407, -0.0025] |
| hn_adapter - F | 2 | kway10 | -0.0409 [-0.0605, -0.0243] |
| hn_adapter - F | 4 | MRR | -0.0165 [-0.0230, -0.0104] |
| hn_adapter - F | 4 | kway5 | -0.0196 [-0.0386, -0.0023] |
| hn_adapter - F | 4 | kway10 | -0.0357 [-0.0548, -0.0178] |
| hn_adapter - F | 6 | MRR | -0.0177 [-0.0242, -0.0110] |
| hn_adapter - F | 6 | kway5 | -0.0241 [-0.0456, -0.0035] |
| hn_adapter - F | 6 | kway10 | -0.0326 [-0.0491, -0.0163] |
| hn_adapter - F | 8 | MRR | -0.0180 [-0.0253, -0.0112] |
| hn_adapter - F | 8 | kway5 | -0.0401 [-0.0635, -0.0166] |
| hn_adapter - F | 8 | kway10 | -0.0394 [-0.0585, -0.0190] |
| random_neg_adapter (control) - F | 2 | MRR | -0.0156 [-0.0229, -0.0082] |
| random_neg_adapter (control) - F | 2 | kway5 | -0.0252 [-0.0403, -0.0106] |
| random_neg_adapter (control) - F | 2 | kway10 | -0.0353 [-0.0551, -0.0169] |
| random_neg_adapter (control) - F | 4 | MRR | -0.0177 [-0.0254, -0.0096] |
| random_neg_adapter (control) - F | 4 | kway5 | -0.0257 [-0.0382, -0.0136] |
| random_neg_adapter (control) - F | 4 | kway10 | -0.0390 [-0.0553, -0.0230] |
| random_neg_adapter (control) - F | 6 | MRR | -0.0200 [-0.0282, -0.0116] |
| random_neg_adapter (control) - F | 6 | kway5 | -0.0260 [-0.0402, -0.0131] |
| random_neg_adapter (control) - F | 6 | kway10 | -0.0374 [-0.0529, -0.0237] |
| random_neg_adapter (control) - F | 8 | MRR | -0.0213 [-0.0293, -0.0133] |
| random_neg_adapter (control) - F | 8 | kway5 | -0.0374 [-0.0533, -0.0227] |
| random_neg_adapter (control) - F | 8 | kway10 | -0.0438 [-0.0563, -0.0308] |

Selected (weight decay, epochs) per fold: 0:(1.0, 5), 1:(1.0, 5), 2:(1.0, 5), 3:(1.0, 5), 4:(1.0, 5), 5:(1.0, 5), 6:(1.0, 5), 7:(1.0, 5)

```
{
  "V2": {
    "5way_ci_lo_gt0_prefixes": [],
    "5way_part": false,
    "MRR_ci_hi_gt0_prefixes": [],
    "MRR_part": false,
    "pass": false,
    "rule": "improves on F (hard-negative adapter, mean over 3 seeds)"
  },
  "control": {
    "5way_ci_lo_gt0_prefixes": [],
    "5way_part": false,
    "MRR_ci_hi_gt0_prefixes": [],
    "MRR_part": false,
    "pass": false,
    "note": "descriptive only"
  }
}
```
