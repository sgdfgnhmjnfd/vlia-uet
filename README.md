# VLIA-UET

**VLIA (Vision-Language-Intention-Action)** is a research project that extends **SmolVLA** with an explicit human-intention representation for robot manipulation.

The main research question is:

> Can a VLA policy benefit from an intention representation predicted from human egocentric observations?

The project separates **intention prediction** from **action prediction**. A frozen vision-language backbone extracts semantic visual features from egocentric video, an intention module predicts a compact latent intention representation, and that representation is injected into SmolVLA as an additional prefix token before action generation.

---

## Overview

```text
Human egocentric observation
        │
        ▼
Frozen SmolVLM visual backbone
        │
        ▼
Temporal Intention Predictor
        │
        ▼
Predicted intention z_int
        │
        ▼
VLIA / SmolVLA
        │
        ▼
Robot action
```

VLIA uses the prefix structure:

```text
[IMAGE] [LANGUAGE] [INTENTION] [STATE]
```

The intention representation is intended to capture a **phase-level semantic behavioral objective**: why the current manipulation behavior is being performed, rather than the low-level action itself.

---

## Research Scope

This repository currently focuses on three questions:

1. **Intention prediction** — Can phase-level human intention be predicted from egocentric visual observations?
2. **Temporal intention modeling** — Does explicit temporal reasoning improve over mean-pooled visual representations?
3. **Downstream VLA conditioning** — Can a predicted intention representation improve robot action prediction when injected into SmolVLA?

An Oracle intention condition is also used as a controlled diagnostic upper bound. It is **not** the main method and should not be interpreted as ground-truth human cognition.

---

## Intention Prediction

### Baseline: mean-pooled visual representation

The current Stage-A baseline uses a frozen `SmolVLM2-500M-Video-Instruct` visual backbone.

```text
8 RGB frames
   │
   ▼
SmolVLM image processor
   │
   ▼
Vision model + connector
   │
   ▼
mean visual-token pooling
   │
   ▼
mean tile pooling
   │
   ▼
per-frame features [T, 960]
   │
   ▼
temporal mean pooling
   │
   ▼
visual feature [960]
   │
   ▼
intention encoder
   │
   ▼
z_int [256]
```

The clean visual-only Stage-A predictor uses no future annotation, observed-next-step label, or privileged intention text as predictor input.

### Proposed temporal intention module

The proposed module replaces temporal mean pooling with learnable temporal reasoning:

```text
Per-frame SmolVLM features [B, T, 960]
        │
        ▼
Linear projection 960 -> 256
        │
        ▼
Learnable temporal positional embeddings
        │
        ▼
2-layer Transformer Encoder
        │
        ▼
Learnable INTENT query
        │
        ▼
Cross-attention over temporal features
        │
        ▼
z_int [B, 256]
```

The module can also return attention weights over frames for qualitative analysis.

---

## VLIA Integration

The predicted intention latent is mapped into the SmolVLA hidden space using an intention adapter:

```text
predicted z_int [256]
        │
        ▼
IntentionAdapter
        │
        ▼
intention token [1, 960]
        │
        ▼
SmolVLA prefix
```

A synthetic integration test currently verifies:

```text
baseline prefix:  [1, 113, 960]
VLIA prefix:      [1, 114, 960]
```

The additional token corresponds to the predicted intention representation.

The repository also supports a separate Oracle experiment in which a privileged 960-D semantic WHY representation is injected into the same VLIA path.

---

## Data

### EgoIntent

EgoIntent is used for Stage-A human-intention prediction experiments.

Current pilot split:

- 771 egocentric clips
- 8 event categories
- 8 unique source videos
- 556 training samples
- 215 validation samples
- zero `video_uid` overlap between train and validation

The validation split is intentionally difficult: it combines unseen videos, unseen event categories, and a domain shift between indoor and outdoor scenes.

Predictor inputs do **not** use:

- `observed_next_step`
- future actions
- plausible future steps
- WHY text

WHY text is used only as semantic supervision.

### LIBERO

LIBERO is used for downstream action-policy experiments.

The matched LIBERO-10 training subset contains:

- 88,302 canonical frames
- 335 episodes
- action chunk size: 50
- semantic action padding at episode boundaries

---

## Current Experimental Results

These results are intermediate research diagnostics and should not be interpreted as final task-level conclusions.

### Stage-A intention retrieval

Random multi-positive retrieval baseline:

| Metric | Value |
| --- | ---: |
| Top-1 | 0.0087 |
| Top-3 | 0.0259 |
| MRR | 0.0420 |

Clean visual-only Stage-A predictor:

| Metric | Value |
| --- | ---: |
| Top-1 | 0.0233 |
| Top-3 | 0.0372 |
| MRR | 0.0542 |
| Positive cosine | 0.0504 |
| Margin | -0.1248 |

The clean predictor is above the random ranking baseline in MRR, but absolute retrieval performance remains low. The temporal intention module is being developed to test whether explicit temporal modeling improves this result.

### LIBERO reference baseline

A previous LIBERO-10 evaluation produced:

- 50 successes / 100 episodes
- 50% success rate

This result is retained only as a reference. Final VLIA comparisons use a matched retraining protocol.

### Oracle diagnostic

A 500-step matched optimization pilot showed lower training loss for Oracle VLIA than the matched baseline. This is only an optimization signal; no downstream success-rate improvement is claimed from that pilot.

---

## Repository Structure

```text
vlia-uet/
├── config/
│   ├── data/
│   ├── suites/
│   └── tasks/
│
├── models/
│   ├── intention_encoder.py
│   └── temporal_intention_encoder.py
│
├── policies/
│   ├── intention/
│   │   └── clean_stage_a_encoder.py
│   └── smolvla/
│       ├── configuration_smolvla.py
│       └── modeling_smolvla.py
│
├── vlia_data/
│   └── libero_oracle_dataset.py
│
├── scripts/
│   ├── cache_egointent_stage_a_features.py
│   ├── cache_egointent_temporal_features.py
│   ├── eval_stage_a_cached.py
│   ├── train_matched_oracle.py
│   ├── test_clean_stage_a_encoder.py
│   ├── test_predicted_intention_adapter.py
│   ├── test_predicted_vlia_prefix.py
│   └── test_predicted_intention_real_cached.py
│
├── docs/
├── robots/
├── urdf/
├── meshes/
├── README.md
├── LICENSE
└── requirements.txt
```

Some older phase-based scripts and configuration files may remain in the repository for experiment history. They are not necessarily part of the current VLIA training path.

---

## Environment

The project is developed on top of the Hugging Face LeRobot stack.

Activate the LeRobot virtual environment:

```bash
cd ~/lerobot
source .venv/bin/activate
cd ~/vlia-uet
```

The expected Python executable is:

```bash
/home/dhqg/lerobot/.venv/bin/python
```

Because this repository contains a local `datasets/` package, the current development setup uses:

```bash
export PYTHONPATH=/home/dhqg/lerobot/.venv/lib/python3.12/site-packages:/home/dhqg/lerobot/src:/home/dhqg/vlia-uet
```

For repository scripts, `python -P` is recommended to avoid local package shadowing:

```bash
python -P scripts/<script_name>.py
```

---

## Key Validation Commands

### Clean Stage-A encoder

```bash
python -P scripts/test_clean_stage_a_encoder.py
```

Expected:

```text
CLEAN STAGE-A ENCODER: PASS
```

### Predicted intention adapter

```bash
python -P scripts/test_predicted_intention_adapter.py
```

Expected:

```text
PREDICTED INTENTION ADAPTER: PASS
```

### Predicted VLIA prefix injection

```bash
python -P scripts/test_predicted_vlia_prefix.py
```

Expected:

```text
PREDICTED VLIA PREFIX: PASS
```

### Real cached EgoIntent feature

```bash
python -P scripts/test_predicted_intention_real_cached.py
```

Expected:

```text
REAL CACHED PREDICTED INTENTION: PASS
```

---

## Matched LIBERO Training

Baseline:

```bash
python -u -P scripts/train_matched_oracle.py \
  --condition baseline \
  --output-root /media/dhqg/d1/vlia_outputs/oracle_matched_full \
  --steps 25000 \
  --batch-size 32 \
  --warmup-steps 1000 \
  --save-freq 1000 \
  --log-freq 50 \
  --seed 0
```

Oracle:

```bash
python -u -P scripts/train_matched_oracle.py \
  --condition oracle \
  --output-root /media/dhqg/d1/vlia_outputs/oracle_matched_full \
  --steps 25000 \
  --batch-size 32 \
  --warmup-steps 1000 \
  --save-freq 1000 \
  --log-freq 50 \
  --seed 0
```

The trainer stores:

```text
checkpoint_xxxxxx/
├── pretrained_model/
└── training_state.pt
```

`training_state.pt` contains the training step, optimizer state, scheduler state, arguments, and running information.

Resume example:

```bash
python -u -P scripts/train_matched_oracle.py \
  --condition baseline \
  --output-root /media/dhqg/d1/vlia_outputs/oracle_matched_full \
  --steps 25000 \
  --batch-size 32 \
  --warmup-steps 1000 \
  --save-freq 1000 \
  --log-freq 50 \
  --seed 0 \
  --resume /path/to/checkpoint_xxxxxx
```

---

## Development Status

| Component | Status |
| --- | --- |
| SmolVLA baseline integration | Implemented |
| VLIA intention token injection | Implemented |
| Oracle intention conditioning | Implemented |
| Clean visual-only Stage-A predictor | Implemented |
| 256 -> 960 intention adapter | Implemented |
| Predicted VLIA prefix path | Implemented |
| Real cached EgoIntent -> predicted intention path | Implemented |
| Temporal Intention Encoder | Implemented, training pipeline in progress |
| Per-frame EgoIntent feature cache | In progress |
| Predicted-intention downstream training | Planned |
| Matched LIBERO baseline / Oracle evaluation | In progress |
| UR3 Gazebo + ROS2 deployment path | In progress |

---

## Experimental Principles

To keep comparisons interpretable:

- intention prediction is evaluated separately from action prediction;
- train/validation splits are separated by source video;
- future information is never used as predictor input;
- Oracle intention is treated as privileged supervision/diagnostic information;
- baseline and Oracle downstream runs use matched training settings;
- training loss alone is not treated as evidence of task-level improvement;
- final claims require downstream rollout evaluation.

---

## Project Goal

The final goal is a VLA system in which a robot action policy can be conditioned on a compact semantic representation of **predicted human intention**, enabling intention-aware robot manipulation and future human-robot collaboration experiments.

The current deployment platform is based on **UR3 + ROS2 + Gazebo**, while LIBERO is used as the main policy benchmark.