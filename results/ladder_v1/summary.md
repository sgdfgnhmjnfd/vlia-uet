# Ladder v1 Stage 3: nested pipeline ladder

Out-of-fold, scene-grouped 8-fold; train/val only. Lifts: video bootstrap B=10000, seed 0.

| Pipeline | k | MRR | MRR lift [95% CI] | 5-way acc | 5-way lift [95% CI] | 10-way acc | 10-way lift [95% CI] |
|---|---:|---:|---|---:|---|---:|---|
| baseline | 2 | 0.0671 | 2.254 [1.865, 2.748] | 0.2773 | 1.386 [1.230, 1.559] | 0.1570 | 1.570 [1.353, 1.824] |
| baseline | 4 | 0.0715 | 2.403 [2.015, 2.878] | 0.2753 | 1.376 [1.219, 1.548] | 0.1614 | 1.614 [1.395, 1.873] |
| baseline | 6 | 0.0734 | 2.466 [2.062, 2.969] | 0.2809 | 1.404 [1.238, 1.580] | 0.1674 | 1.674 [1.430, 1.973] |
| baseline | 8 | 0.0765 | 2.571 [2.125, 3.136] | 0.2885 | 1.442 [1.273, 1.623] | 0.1774 | 1.774 [1.501, 2.091] |
| nested_selected | 2 | 0.0747 | 2.510 [1.941, 3.175] | 0.3408 | 1.704 [1.520, 1.893] | 0.2085 | 2.085 [1.787, 2.410] |
| nested_selected | 4 | 0.0777 | 2.610 [2.032, 3.295] | 0.3492 | 1.746 [1.560, 1.935] | 0.2101 | 2.101 [1.789, 2.431] |
| nested_selected | 6 | 0.0825 | 2.771 [2.180, 3.466] | 0.3624 | 1.812 [1.610, 2.011] | 0.2189 | 2.189 [1.873, 2.516] |
| nested_selected | 8 | 0.0850 | 2.855 [2.241, 3.562] | 0.3871 | 1.936 [1.718, 2.148] | 0.2349 | 2.349 [2.021, 2.678] |

## Secondary (no verdict)

| Pipeline | k | clean 5-way acc | clean 5-way lift [95% CI] | para MRR | para MRR lift [95% CI] |
|---|---:|---:|---|---:|---|
| baseline | 2 | 0.2877 | 1.438 [1.285, 1.605] | 0.1003 | 1.906 [1.537, 2.368] |
| baseline | 4 | 0.2940 | 1.470 [1.309, 1.649] | 0.1046 | 1.987 [1.638, 2.430] |
| baseline | 6 | 0.2980 | 1.490 [1.321, 1.677] | 0.1072 | 2.036 [1.664, 2.512] |
| baseline | 8 | 0.3080 | 1.540 [1.374, 1.717] | 0.1120 | 2.127 [1.713, 2.662] |
| nested_selected | 2 | 0.3416 | 1.708 [1.485, 1.938] | 0.1010 | 1.919 [1.441, 2.500] |
| nested_selected | 4 | 0.3604 | 1.802 [1.587, 2.020] | 0.1055 | 2.005 [1.514, 2.607] |
| nested_selected | 6 | 0.3783 | 1.892 [1.674, 2.108] | 0.1106 | 2.101 [1.596, 2.722] |
| nested_selected | 8 | 0.3939 | 1.970 [1.728, 2.201] | 0.1147 | 2.179 [1.650, 2.826] |

## Paired: nested-selected minus baseline

| k | metric | delta [95% CI] |
|---:|---|---|
| 2 | MRR | +0.0076 [-0.0037, +0.0195] |
| 2 | kway5 | +0.0635 [+0.0389, +0.0892] |
| 2 | kway10 | +0.0515 [+0.0331, +0.0708] |
| 4 | MRR | +0.0062 [-0.0059, +0.0183] |
| 4 | kway5 | +0.0739 [+0.0491, +0.0999] |
| 4 | kway10 | +0.0487 [+0.0278, +0.0707] |
| 6 | MRR | +0.0091 [-0.0024, +0.0206] |
| 6 | kway5 | +0.0815 [+0.0510, +0.1120] |
| 6 | kway10 | +0.0515 [+0.0286, +0.0750] |
| 8 | MRR | +0.0085 [-0.0026, +0.0195] |
| 8 | kway5 | +0.0987 [+0.0621, +0.1344] |
| 8 | kway10 | +0.0575 [+0.0329, +0.0829] |

## Selected configuration per outer fold

| fold | held-out scenes | encoder | agg | centering | template | train mean 5-way | baseline train mean 5-way | ties |
|---:|---|---|---|---|---|---:|---:|---:|
| 0 | workshop | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3704 | 0.2954 | 1 |
| 1 | kitchen | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3517 | 0.2633 | 1 |
| 2 | garage | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3475 | 0.2848 | 1 |
| 3 | farm | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3529 | 0.2742 | 1 |
| 4 | hallway,living_room | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3642 | 0.2809 | 1 |
| 5 | bedroom,yard | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3620 | 0.2771 | 1 |
| 6 | clothing_closet,space | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3711 | 0.2893 | 1 |
| 7 | art_studio,deck,laundry_room | siglip-so400m-patch14-384 | A0 | C2 | T1 | 0.3590 | 0.2791 | 1 |

Selection stability: 1 distinct configs over 8 folds; most common `siglip-so400m-patch14-384|A0|C2|T1` in 8 folds.

## Mismatched-video control (every 8th query by sample_id, k=8)

| config | selected in folds | n | real 5-way | control 5-way |
|---|---|---:|---:|---:|
| siglip-so400m-patch14-384|A0|C2|T1 | 0,1,2,3,4,5,6,7 | 313 | 0.3962 | 0.1885 |
| siglip-base-patch16-224|A0|C0|T0 | baseline | 313 | 0.2620 | 0.1949 |

## Verdicts

```
{
  "N1": {
    "pass": false,
    "label": "Selection does not help (rule not met)",
    "MRR_ci_lo_gt0_prefixes": [],
    "5way_ci_lo_gt0_prefixes": [
      2,
      4,
      6,
      8
    ],
    "rule": "per metric: CI lower bound > 0 at >= 2 of 4 prefixes, for both MRR and 5-way",
    "both_metrics_same_prefix": []
  },
  "N2": {
    "pass": true,
    "5way_acc_ge_0.30_prefixes": [
      2,
      4,
      6,
      8
    ],
    "MRR_lift_ci_lo_gt1_prefixes": [
      2,
      4,
      6,
      8
    ],
    "5way_lift_ci_lo_gt1_prefixes": [
      2,
      4,
      6,
      8
    ],
    "G1": "FAIL (unchanged)"
  }
}
```
