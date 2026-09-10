# Stage-A EgoIntent Pilot

Date: 2026-09-10

## Dataset

A multi-video EgoIntent pilot was prepared using 8 event directories and 8 distinct `video_uid`s.

- Total samples: 771
- Train samples: 556
- Validation samples: 215
- Train videos: 6
- Validation videos: 2
- Train/validation `video_uid` overlap: 0

The current pilot split uses six indoor videos for training and two outdoor videos for validation.

This validation split therefore measures both unseen-video generalization and cross-domain/cross-event transfer. It should not be interpreted as an isolated measure of encoder quality.

## Stage-A frozen feature cache

Frozen SmolVLM features were successfully extracted for all 771 samples.

Cached feature tensors:

- visual: 960-D
- task: 960-D
- history: 960-D
- WHY target: 960-D

Train cache:

- visual: `[556, 960]`
- task: `[556, 960]`
- history: `[556, 960]`
- why: `[556, 960]`

Validation cache:

- visual: `[215, 960]`
- task: `[215, 960]`
- history: `[215, 960]`
- why: `[215, 960]`

All cached tensors passed shape and finite-value validation.

## Current Stage-A status

The Stage-A alignment objective operates in the frozen SmolVLM 960-D semantic space.

The current `960 -> 256` intention projection is not optimized by this alignment loss and must not yet be interpreted as a learned semantic intention representation.

A previous one-sample optimization smoke test verified that the trainable fusion path can optimize the cosine-alignment objective. This is an implementation sanity check only, not a generalization result.

## Next step

Train the fusion module on the cached multi-video training set and evaluate held-out validation performance using:

- mean cosine similarity
- Top-1 retrieval
- Top-3 retrieval
- MRR

No claim about downstream action-policy improvement is made at this stage.