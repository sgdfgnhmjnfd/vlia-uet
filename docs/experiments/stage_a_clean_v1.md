# Stage-A Clean Intention V1

## Data

EgoIntent pilot:

- 771 micro-step videos
- 8 distinct source `video_uid`s
- train: 556 samples from 6 indoor videos
- validation: 215 samples from 2 outdoor videos
- train/validation `video_uid` overlap: 0

The validation split contains both unseen-video and cross-domain/cross-event
shift and should therefore be interpreted as a pilot stress test.

## V0: cosine-only alignment

The first Stage-A objective aligned the predicted 960-D representation with
the frozen SmolVLM WHY embedding using cosine loss.

Best validation cosine:

- 0.7951

However, the constant mean-WHY baseline achieved:

- cosine: 0.8247

Retrieval remained approximately random, indicating that cosine-only alignment
was not sufficiently discriminative.

## V1: multi-positive contrastive alignment

The objective was changed to:

- multi-positive InfoNCE
- plus 0.1 cosine alignment

Duplicate normalized WHY strings are treated as multiple valid positives.

The multi-positive random baseline on the validation set is:

- Top-1: 0.00868
- Top-3: 0.02591
- MRR: 0.04203

Ablation experiments across multiple random seeds showed that visual input
alone consistently produced above-random retrieval performance. Annotated
semantic history also contained predictive signal, but it is not considered a
clean deployment-time input.

The clean predictor therefore uses visual observation only.

## Clean Intention V1

Architecture:

- frozen SmolVLM visual feature: 960-D
- fusion: 2880 -> 1024 -> 960
- intention projection: 960 -> 256
- final intention representation: 256-D

Only the visual feature is supplied to the fusion network; unavailable
task/history branches are zeroed in the current implementation.

The 256-D training target is produced by PCA from frozen 960-D WHY embeddings.
The PCA basis is fitted only on the training split.

Training objective:

- multi-positive InfoNCE
- 0.1 cosine alignment
- checkpoint selected by validation MRR

Best clean V1 checkpoint:

- epoch: 14
- validation Top-1: 0.02326
- validation Top-3: 0.03721
- validation MRR: 0.05424
- validation margin: -0.12483

The 256-D representation therefore retains the above-random retrieval signal
observed in the clean visual-only Stage-A experiments.

The negative margin shows that the representation is not yet strongly
separated from hard negatives. These results demonstrate a learnable
intention-related signal but do not establish accurate intention prediction or
downstream action-policy improvement.

## Frozen artifact

The Stage-A predictor head is exported as:

`intention_encoder_v1.pt`

It contains:

- fusion weights
- 960 -> 256 intention projection weights

PCA is required only for Stage-A supervision and is not required during
Predicted VLIA inference.

The next experimental stage is Oracle VLIA, followed by Predicted VLIA.