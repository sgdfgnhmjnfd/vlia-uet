# E1b v2: SmolVLM2-500M zero-shot K=5 goal scoring

Fractional tie credit; lift CI = video bootstrap B=10000, seed 0. Train/val only.

| k | score | n | 5-way acc | lift [95% CI] |
|---:|---|---:|---:|---|
| 2 | pmi_mean | 2503 | 0.2353 | 1.177 [1.074, 1.285] |
| 2 | pmi_sum | 2503 | 0.2369 | 1.185 [1.078, 1.289] |
| 2 | lp_img_mean | 2503 | 0.2313 | 1.157 [1.068, 1.249] |
| 2 | lp_img_sum | 2503 | 0.2471 | 1.236 [1.122, 1.352] |
| 8 | pmi_mean | 2503 | 0.2461 | 1.231 [1.135, 1.333] |
| 8 | pmi_sum | 2503 | 0.2393 | 1.197 [1.082, 1.317] |
| 8 | lp_img_mean | 2503 | 0.2425 | 1.213 [1.115, 1.319] |
| 8 | lp_img_sum | 2503 | 0.2541 | 1.270 [1.147, 1.398] |

## E1b-B (descriptive): SmolVLM2 minus SigLIP float32 baseline, identical queries

| k | SmolVLM2 score | SmolVLM2 acc | SigLIP acc | delta [95% CI] |
|---:|---|---:|---:|---|
| 2 | pmi_mean | 0.2353 | 0.2773 | -0.0419 [-0.0753, -0.0105] |
| 2 | pmi_sum | 0.2369 | 0.2773 | -0.0404 [-0.0716, -0.0118] |
| 2 | lp_img_mean | 0.2313 | 0.2773 | -0.0459 [-0.0772, -0.0177] |
| 2 | lp_img_sum | 0.2471 | 0.2773 | -0.0302 [-0.0659, +0.0050] |
| 8 | pmi_mean | 0.2461 | 0.2885 | -0.0423 [-0.0746, -0.0126] |
| 8 | pmi_sum | 0.2393 | 0.2885 | -0.0491 [-0.0812, -0.0204] |
| 8 | lp_img_mean | 0.2425 | 0.2885 | -0.0459 [-0.0740, -0.0198] |
| 8 | lp_img_sum | 0.2541 | 0.2885 | -0.0344 [-0.0696, -0.0015] |

## Mismatched-video control (every 8th query, k=8)

| score | n | real acc | control acc | control lift [95% CI] |
|---|---:|---:|---:|---|
| pmi_mean | 313 | 0.2428 | 0.2077 | 1.038 [0.794, 1.296] |
| pmi_sum | 313 | 0.2141 | 0.1981 | 0.990 [0.807, 1.165] |
| lp_img_mean | 313 | 0.2588 | 0.2236 | 1.118 [0.906, 1.326] |
| lp_img_sum | 313 | 0.2620 | 0.2268 | 1.134 [0.903, 1.368] |

## Scorer diagnostics

```
{
  "2": {
    "lp_img_mean_range": [
      -17.306564331054688,
      -2.1672797203063965
    ],
    "mean_within_query_std_lp_img": 1.497599165114676,
    "mean_within_query_std_lp_null": 1.8337400632603762,
    "std_ratio_null_over_img": 1.2244531821169657,
    "fraction_pmi_gt0": 0.9944866160607271,
    "queries_with_tied_max_pmi_mean": 0
  },
  "8": {
    "lp_img_mean_range": [
      -18.561471939086914,
      -2.07971986134847
    ],
    "mean_within_query_std_lp_img": 1.4819216058298967,
    "mean_within_query_std_lp_null": 1.82314885136759,
    "std_ratio_null_over_img": 1.2302599841957236,
    "fraction_pmi_gt0": 0.9947263284059129,
    "queries_with_tied_max_pmi_mean": 0
  }
}
```

## Verdict

```
{
  "E1b-A": {
    "pass": true,
    "prefixes_lift_ci_lo_gt1": [
      2,
      8
    ],
    "prefixes_evaluated": [
      2,
      8
    ],
    "rule": "primary per-token-mean PMI, fractional tie credit: 5-way lift CI lower bound > 1 (video bootstrap B=10000, seed 0) at BOTH evaluated prefixes"
  },
  "E1b-B": "descriptive only (paired_vs_siglip.csv)"
}
```
