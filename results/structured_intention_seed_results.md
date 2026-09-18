# Structured Intention Prediction Results

This file records the current Stage-A intention-prediction results for the
structured What–Why–Next experiments on EgoIntent.

## Experimental setup

- Train split: 556 samples
- Validation split: 215 samples
- Visual input: cached temporal frame features with shape `[8, 960]`
- Primary target: `Why`
- Auxiliary targets: `What` and `Next`
- Intention dimension: 256
- Structured model: temporal attention backbone with auxiliary What and Next heads
- Full structured loss:

```text
L = L_why + 0.3 * L_what + 0.3 * L_next
```

- Each branch uses multi-positive InfoNCE plus cosine alignment.
- Checkpoint selection is based only on validation `Why` MRR.
- `Next` is supervision only and is never used as predictor input.
- `Why` features are copied exactly from the original Stage-A cache for fair
  comparison with the Clean V1 target space.
- `What` and `Next` use separate train-only PCA projections from 960 to 256 dimensions.

## Target preparation checks

The temporal-cache `Why` features were synchronized with the original Stage-A
cache before structured training.

```text
matched: 556
exact equal: 556
max abs diff: 0.0
source: stage_a_v0_exact
```

The structured PCA artifacts have the same format as the original Why PCA:

```text
mean:       [1, 960]
components: [960, 256]
input_dim:  960
output_dim: 256
fit_split:  train
```

Retained PCA energy:

| Target | Retained energy |
| --- | ---: |
| What | 0.99105 |
| Next | 0.99480 |

## Seed-0 ablation

| Configuration | What weight | Next weight | Best epoch | Top-1 | Top-3 | MRR | Margin | Mean cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Why only | 0.0 | 0.0 | 23 | 0.02326 | 0.03721 | 0.05157 | -0.15389 | 0.02674 |
| Why + What | 0.3 | 0.0 | 10 | 0.02326 | 0.04186 | 0.05314 | -0.13771 | 0.02364 |
| Why + Next | 0.0 | 0.3 | 10 | 0.02791 | 0.03721 | 0.05299 | -0.12385 | 0.02349 |
| Why + What + Next | 0.3 | 0.3 | 10 | 0.03256 | 0.03721 | 0.05714 | -0.14773 | 0.01953 |

At seed 0, both auxiliary targets individually improve MRR over the Why-only
control, while the full What–Why–Next model gives the highest MRR.

However, this seed-0 improvement is not sufficient by itself to establish a
stable gain, so the Why-only and full structured configurations were repeated
with seeds 1 and 2.

## Multi-seed results

### Why-only control

| Seed | Best epoch | Top-1 | Top-3 | MRR | Margin | Mean cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 23 | 0.02326 | 0.03721 | 0.05157 | -0.15389 | 0.02674 |
| 1 | 59 | 0.02791 | 0.04186 | 0.05381 | -0.15729 | -0.00278 |
| 2 | 66 | 0.02326 | 0.05581 | 0.06130 | -0.11763 | 0.02996 |

MRR over three seeds:

```text
0.05157
0.05381
0.06130
```

Mean ± sample standard deviation:

```text
0.05556 ± 0.00510
```

### Full What–Why–Next

| Seed | Best epoch | Top-1 | Top-3 | MRR | Margin | Mean cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10 | 0.03256 | 0.03721 | 0.05714 | -0.14773 | 0.01953 |
| 1 | 70 | 0.01860 | 0.03721 | 0.05008 | -0.13101 | 0.01013 |
| 2 | 7 | 0.02326 | 0.04651 | 0.05813 | -0.08238 | 0.01216 |

MRR over three seeds:

```text
0.05714
0.05008
0.05813
```

Mean ± sample standard deviation:

```text
0.05512 ± 0.00439
```

## Multi-seed comparison

| Configuration | Seed 0 | Seed 1 | Seed 2 | Mean MRR | Sample std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Why only | 0.05157 | 0.05381 | 0.06130 | 0.05556 | 0.00510 |
| Why + What + Next | 0.05714 | 0.05008 | 0.05813 | 0.05512 | 0.00439 |

Difference in mean MRR:

```text
Full structured - Why only = -0.00044
```

Relative difference with respect to the Why-only mean:

```text
-0.80%
```

## Interpretation

The seed-0 ablation initially suggests that structured semantic supervision is
useful: What-only and Next-only auxiliary supervision each improve validation
MRR slightly, and the combined What–Why–Next configuration reaches the best
seed-0 MRR of 0.05714.

The three-seed comparison does not reproduce that gain consistently. The
Why-only control reaches `0.05556 ± 0.00510` MRR, while the full structured
model reaches `0.05512 ± 0.00439`. The difference in means is only `-0.00044`,
which is much smaller than the observed variation across seeds.

Therefore, the current evidence does **not** support a claim that structured
What–Why–Next supervision consistently improves Why/intention prediction. The
appropriate conclusion at this stage is that structured supervision shows a
promising seed-dependent signal, but the effect is not stable under the current
small pilot dataset and training protocol.

The full structured model also has slightly lower MRR variance across these
three seeds, but three seeds are not enough to interpret this as a reliable
stability benefit.

## Reference baselines

| Model | Best MRR |
| --- | ---: |
| Random multi-positive baseline | 0.04203 |
| Temporal Transformer | 0.04079 |
| Temporal Attention, Why only | 0.05157 |
| Mean-pool temporal-cache diagnostic | 0.05196 |
| Clean V1 static mean-pool baseline | 0.05424 |
| Structured What–Why–Next, seed 0 | 0.05714 |
| Structured What–Why–Next, 3-seed mean | 0.05512 |

These rows are not all multi-seed estimates. In particular, the final two
structured rows should be interpreted together with the multi-seed table above.

## Current artifact locations

Why-only ablation:

```text
/media/dhqg/d1/vlia_outputs/structured_ablation/why_only/
```

Why + What seed-0 ablation:

```text
/media/dhqg/d1/vlia_outputs/structured_ablation/why_what/
```

Why + Next seed-0 ablation:

```text
/media/dhqg/d1/vlia_outputs/structured_ablation/why_next/
```

Full structured runs:

```text
/media/dhqg/d1/vlia_outputs/structured_intention_v1/
```

PCA artifacts:

```text
/media/dhqg/d1/vlia_outputs/structured_intention_v1/what_pca_960_to_256.pt
/media/dhqg/d1/vlia_outputs/stage_a_clean_v1/why_pca_960_to_256.pt
/media/dhqg/d1/vlia_outputs/structured_intention_v1/next_pca_960_to_256.pt
```

## Current conclusion

The cleanest finding so far is:

1. The lightweight temporal-attention implementation reproduces the Why-only
   structured control correctly.
2. Seed-0 structured supervision gives a noticeable gain.
3. That gain does not remain consistent over three seeds.
4. Seed sensitivity is substantial on the current 556/215 pilot split.
5. The next experiments should focus on robustness, regularization, more data,
   or cross-view supervision rather than simply increasing encoder complexity.