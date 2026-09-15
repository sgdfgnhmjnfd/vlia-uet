# VLIA-UET

## Intention Prediction for Vision-Language-Action Models

This repository focuses on **human intention prediction for Vision-Language-Action (VLA) systems**.

The main research problem is not to redesign the downstream action policy itself, but to build an intention prediction module that can infer a structured semantic state from visual observations and expose it as a compact representation for a VLA policy.

The current direction is:

```text
Human visual observation
        ↓
Temporal / cross-view intention predictor
        ↓
What / Why / Next reasoning
        ↓
predicted intention representation z_int
        ↓
VLA policy interface
        ↓
robot action generation
```

The downstream VLA policy is treated as a consumer of the predicted intention representation. The core contribution of this repository is the **intention prediction side**.

---

## 1. Research Motivation

Vision-Language-Action models typically map visual observations, language instructions, and robot state directly to low-level action chunks.

However, long-horizon manipulation is naturally organized into semantic phases. A task instruction may remain unchanged while the local behavioral purpose changes over time.

For example:

```text
Task:
place the cup under the faucet and fill it with water

Semantic progression:
approach cup
→ grasp cup
→ move cup toward faucet
→ align cup under faucet
→ activate faucet
→ fill cup
```

The low-level robot actions are embodiment-specific, but the local semantic intention can be more abstract.

This motivates the following hypothesis:

> **Intention should be represented as a temporally evolving semantic state that can be inferred from visual behavior and consumed by a downstream VLA policy.**

---

## 2. Main Research Questions

This project currently studies three main questions:

1. **Can phase-level intention be predicted from temporal visual observations?**

2. **Does modeling temporal progression improve intention prediction compared with static mean-pooled visual representations?**

3. **Can structured What–Why–Next supervision and cross-view training improve the quality and generalization of the predicted intention representation?**

A downstream VLA experiment is used only as a final validation step:

> Can the predicted intention representation serve as a useful conditioning signal for a VLA policy?

---

## 3. Intention Definition

In this project, intention is defined as a **local semantic objective at the current behavioral phase**.

The representation is organized around three related concepts:

### What

What is currently happening.

Examples:

```text
holding the cup
moving the cup toward the faucet
aligning the cup under the faucet
```

### Why

Why the current behavior is being performed.

This is the main intention target.

Examples:

```text
position the cup so that it can receive water
establish a stable grasp before transport
align the object with the target before placement
```

### Next

What semantic step is likely to follow.

Examples:

```text
place cup under faucet
→ activate faucet
→ fill cup
```

The project does **not** use future annotations as predictor inputs. Future information is only used as supervision or evaluation targets.

---

## 4. Structured Intention Prediction

Instead of predicting only a single isolated intention label, the current direction models a structured semantic state:

\[
\hat{\mathcal{S}}_t
=
\{
\hat{W}_t,
\hat{Y}_t,
\hat{N}_{t+1:t+K}
\}
\]

where:

- \(W_t\): current **What**
- \(Y_t\): current **Why / Intention**
- \(N_{t+1:t+K}\): short future semantic sequence

The main exported representation remains:

\[
z_t^{\mathrm{int}}
\in
\mathbb{R}^{256}
\]

This latent is intended to be used as the interface to a downstream VLA policy.

---

## 5. Current Model Direction

### 5.1 Static Mean-Pooling Baseline

The current clean baseline uses frozen SmolVLM visual features.

```text
video frames
→ frozen SmolVLM
→ frame / tile features
→ temporal mean pooling
→ visual feature [960]
→ intention predictor
→ z_int [256]
```

This baseline is used to measure whether temporal reasoning provides a real improvement.

---

### 5.2 Temporal Intention Encoder

The proposed temporal module keeps the frame sequence instead of averaging over time.

```text
8 visual frames
→ frozen SmolVLM
→ per-frame features [8, 960]
→ Linear 960→256
→ temporal positional embeddings
→ 2-layer Transformer Encoder
→ learnable INTENT query
→ attention pooling
→ z_int [256]
```

Current implementation:

```text
models/temporal_intention_encoder.py
```

The module can optionally return temporal attention weights for qualitative analysis.

---

### 5.3 Cross-View Intention Prediction

The next extension is to train the predictor on multiple visual viewpoints.

Target setting:

```text
egocentric clip ──┐
                  ├── Temporal Intention Predictor
exocentric clip ──┘
                           ↓
                     shared z_int
```

The goal is to learn an intention representation that is less dependent on camera viewpoint.

The intended training principle is:

```text
same intention + different view
→ similar latent representation

same task + different phase
→ hard negative
```

This is particularly important because a representation trained only on egocentric data may not generalize well to observations with different viewpoints.

---

## 6. Multi-View Dataset Direction

The project is moving toward a unified intention dataset that can mix:

```text
egocentric human video
exocentric human video
optional robot-view video
```

The canonical sample schema is intended to include:

```text
sample_id
source_dataset

task
phase
what
why
next

view_type
view_id

subject_id
scene_id
video_uid
video_path

start_time
end_time

pair_group_id
split
```

The most important field for cross-view learning is:

```text
pair_group_id
```

Samples with the same semantic intention but different views can be grouped as positives.

Perfect temporal synchronization is not required for the first version. Weak semantic pairing by:

```text
task + phase + intention
```

is acceptable.

---

## 7. Dataset Sources

### EgoIntent

The current Stage-A experiments use EgoIntent as the main egocentric human-intention source.

The mapping currently used is:

```text
procedural_intent      → Why / Intention
local_intent           → What
observed_next_step     → Next supervision / evaluation only
plausible_next_steps   → auxiliary supervision / evaluation only
```

Future annotations are never used as predictor inputs.

Current pilot:

```text
8 source videos
771 micro-step clips

train:
556 samples
6 indoor videos

validation:
215 samples
2 outdoor videos
```

The split is performed by `video_uid` to avoid leakage across clips from the same source video.

The validation split is intentionally difficult because it includes:

```text
unseen video
cross-event transfer
indoor → outdoor domain shift
```

---

## 8. Temporal Feature Cache

To train temporal models efficiently, frozen visual features are cached per frame.

Current cache representation:

```text
frame_features.shape = [8, 960]
```

For clips that decode fewer than 8 frames, the pipeline performs uniform temporal repetition while preserving frame order.

Example:

```text
4 decoded frames
→ 8 temporal positions
```

The temporal cache script is:

```bash
python -P scripts/cache_egointent_temporal_features.py \
  --split train
```

and:

```bash
python -P scripts/cache_egointent_temporal_features.py \
  --split val
```

Default output:

```text
/media/dhqg/d1/datasets/egointent/cache/temporal_intention_v1/
```

---

## 9. Stage-A Intention Prediction Results

The current Stage-A results should be interpreted as baselines for the new temporal predictor.

### Random Multi-Positive Baseline

```text
Top-1: 0.00868
Top-3: 0.02591
MRR:   0.04203
```

### V0: Cosine-Only Alignment

The cosine-only objective produced high cosine similarity but poor retrieval.

```text
Top-1: ~0.0047
Top-3: ~0.0140
MRR:   ~0.028
```

This indicated representation collapse toward the semantic centroid.

---

### V1: Multi-Positive Contrastive Learning

A contrastive objective improved retrieval.

Important reference:

```text
visual-only 960-D
MRR ≈ 0.0540
```

Multi-input variants using task/history sometimes achieved higher MRR, but annotation-derived history is not treated as a clean deployment-time input.

---

### Clean V1: Mean-Pooling Baseline

The current clean baseline predicts a 256-D latent intention from mean-pooled visual features.

Best validation result:

```text
Top-1: 0.0233
Top-3: 0.0372
MRR:   0.0542
Margin: -0.1248
```

This model is now treated as the **static mean-pooling baseline**.

It is not the final proposed method.

The next main comparison is:

```text
Clean V1 mean-pooling
vs.
Temporal Intention Encoder
vs.
Cross-view Temporal Intention Encoder
```

---

## 10. Training Objective Direction

The intended structured training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{why}}
+
\lambda_w \mathcal{L}_{\mathrm{what}}
+
\lambda_n \mathcal{L}_{\mathrm{next}}
+
\lambda_{cv} \mathcal{L}_{\mathrm{crossview}}
\]

where:

- `L_why` is the main intention prediction objective
- `L_what` encourages understanding of the current behavior
- `L_next` encourages predictive temporal structure
- `L_crossview` encourages viewpoint-invariant intention representations

The final design of these losses is still under active development.

---

## 11. VLA Interface

The intention prediction module exports:

```text
z_int ∈ R^256
```

The downstream VLA policy consumes this representation through an intention adapter.

Conceptually:

```text
visual observation
→ intention predictor
→ z_int
────────────────────────
prediction / policy boundary
────────────────────────
→ intention adapter
→ VLA policy
→ action chunk
```

The policy side is not the primary methodological contribution of this repository.

---

## 12. Oracle Intention

Oracle intention is used only as a controlled upper-bound experiment.

Conceptually:

```text
semantic Why
→ frozen text encoder
→ oracle intention embedding
→ downstream VLA
```

The purpose is to answer:

> If a high-quality phase-level intention signal is available, can the downstream VLA policy make use of it?

Oracle intention is **not** treated as ground-truth human cognitive intention.

It is also not the main contribution of the intention prediction module.

---

## 13. Current Oracle / Matched Training Status

A matched LIBERO-10 pipeline has been prepared for downstream evaluation.

Current matched subset:

```text
88,302 training frames
335 canonical episodes
38 Oracle Why embeddings
```

Both baseline and Oracle conditions use matched:

```text
training samples
action chunks
semantic padding
optimizer
scheduler
pretrained initialization
seed protocol
```

A 500-step pilot completed successfully.

```text
Baseline loss_100: 0.583925
Oracle loss_100:   0.565286
```

These numbers are only optimization diagnostics.

They are **not** evidence that Oracle improves task success.

Task-level conclusions require matched LIBERO evaluation.

---

## 14. Experimental Plan

The main intention-prediction experiments are:

### Stage A — Representation / Prediction

```text
Random baseline
V0 cosine-only
V1 contrastive
Clean V1 mean-pooling
Temporal Intention Encoder
Cross-view Temporal Intention Encoder
```

Primary metrics:

```text
Top-1
Top-3
MRR
positive-negative cosine margin
```

Additional structured metrics are planned for:

```text
What prediction
Why prediction
Next@1
Next@K
cross-view retrieval
unseen-view generalization
```

---

### Stage B — Downstream Validation

The downstream evaluation will compare:

```text
No intention
Oracle intention
Predicted intention
```

The purpose is to verify whether intention prediction quality transfers to a VLA consumer.

---

## 15. Repository Structure

Current relevant structure:

```text
vlia-uet/
├── config/
│   └── data/
│
├── datasets/
│   ├── egointent_adapter.py
│   ├── egointent_video.py
│   └── intention_alignment_dataset.py
│
├── models/
│   ├── intention_encoder.py
│   └── temporal_intention_encoder.py
│
├── policies/
│   ├── intention/
│   └── smolvla/
│
├── scripts/
│   ├── cache_egointent_stage_a_features.py
│   ├── cache_egointent_temporal_features.py
│   ├── train_matched_oracle.py
│   └── test_*.py
│
└── README.md
```

---

## 16. Environment

The project currently uses the LeRobot virtual environment.

Example:

```bash
cd ~/lerobot
source .venv/bin/activate

cd ~/vlia-uet
which python
```

Expected interpreter:

```text
/home/dhqg/lerobot/.venv/bin/python
```

Because the repository contains a local `datasets/` directory, some scripts are run with:

```bash
python -P ...
```

to avoid Python package resolution issues.

---

## 17. Important Experimental Constraints

The following rules are treated as part of the experimental protocol.

### No future leakage

The predictor must not receive:

```text
observed_next_step
plausible_next_steps
future semantic actions
future frames beyond the allowed observation window
```

as input.

These fields may only be used as supervision or evaluation targets.

### Video-level split

Samples from the same source video must not be split across training and validation.

### Annotation-derived history

History created from manually annotated semantic actions is not assumed to be available at deployment time.

It is used only for controlled ablation experiments.

### Oracle is an upper bound

Oracle semantic intention is privileged information and must not be presented as normal deployment input.

---

## 18. Current Research Direction

The repository is currently transitioning from:

```text
single-view static intention alignment
```

to:

```text
temporal structured intention prediction
+
cross-view semantic alignment
```

The target final predictor is:

```text
Ego / Exo visual sequence
        ↓
Temporal visual encoder
        ↓
What / Why / Next reasoning
        ↓
view-robust intention latent
        ↓
z_int
```

The intended main contribution is therefore:

> **A temporal and cross-view intention prediction module that reasons jointly about current behavior, current semantic purpose, and short-horizon semantic progression, while exposing a compact latent representation for downstream VLA models.**

---

## 19. Project Status

Completed:

```text
EgoIntent pilot dataset
video-level train/validation split
frozen visual feature extraction
cosine-only baseline
contrastive baseline
modality ablations
Clean V1 256-D mean-pooling baseline
Temporal Intention Encoder implementation
temporal frame-feature cache smoke test
VLA intention-token integration tests
matched Oracle training pipeline
```

In progress:

```text
full temporal feature caching
Temporal Intention Encoder training
structured What–Why–Next formulation
cross-view dataset construction
cross-view intention learning
matched Oracle training / evaluation
```

Planned:

```text
cross-view benchmark
unseen-view evaluation
Predicted intention export
downstream VLA evaluation
final ablation study
```

---

## 20. Research Scope

This repository focuses primarily on:

```text
visual intention prediction
temporal reasoning
cross-view generalization
structured semantic supervision
```

The downstream robot policy is used as an evaluation interface, not as the main algorithmic contribution.

---

## License

This repository contains research code only.

External datasets, pretrained models, and third-party repositories remain subject to their original licenses and terms of use.

In particular, EgoIntent-related data may depend on the licensing and usage conditions of the underlying Ego4D source material.

---

## Citation

Citation information will be added when the corresponding thesis or paper is publicly available.