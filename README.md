<div align="center">

# VLIA-UET

### Vision · Language · Intention · Action

**Explicit semantic intention conditioning for Vision-Language-Action robot policies**

[![Project Status](https://img.shields.io/badge/status-active%20research-2ea44f)](#current-status)
[![Benchmark](https://img.shields.io/badge/benchmark-LIBERO--10-blue)](#stage-b--intention-conditioned-smolvla)
[![Backbone](https://img.shields.io/badge/backbone-SmolVLA-purple)](#stage-b--intention-conditioned-smolvla)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## Why VLIA?

A standard VLA policy sees the scene, reads the instruction, and predicts robot actions.

VLIA asks a narrower question:

> **Can an explicit semantic representation of human intention help a VLA policy disambiguate what the robot should do next?**

The project separates the problem into two stages:

```text
Human egocentric video                          Robot observation
         │                                            │
         ▼                                            │
  Stage-A intention reasoning                         │
         │                                            │
         ▼                                            │
      z_int ∈ R^256                                   │
         │                                            │
         └──────────────┐                             │
                        ▼                             ▼
                 Intention Adapter              Image + Language
                     256 → 960                    + Robot State
                        │                             │
                        └──────────────┬──────────────┘
                                       ▼
                        [Image | Language | Intention | State]
                                       │
                                       ▼
                                    SmolVLA
                                       │
                                       ▼
                                Robot action chunk
```

The current implementation adds **one intention token** to the SmolVLA prefix, immediately before the state token.

---

## Research questions

VLIA is organized around three questions:

1. **Can semantic intention be predicted from egocentric human observations?**
2. **What form of semantic reasoning is useful for fine-grained within-task disambiguation?**
3. **Does an explicit intention signal improve downstream VLA behavior under a matched LIBERO protocol?**

The intended claim is stronger than “an extra token helps”:

> **Semantic intention should act as a disambiguating control variable when observation and language alone are insufficient.**

---

# Stage A — Human intention reasoning

## Input and representation

Stage A uses **EgoIntent** egocentric clips and a frozen
`SmolVLM2-500M-Video-Instruct` visual backbone.

Current pilot protocol:

| Item | Setting |
|---|---|
| Input | 8 RGB frames per clip |
| Train samples | 556 |
| Validation samples | 215 |
| Split | by `video_uid` |
| Primary semantic target | WHY |
| Future annotations as input | No |

A major empirical finding is that most retrieval errors are **within the same task**, indicating that the bottleneck is not broad task recognition but **fine-grained semantic state / purpose disambiguation**.

---

## Selected reasoning path

```text
visual feature
    │
    ├────────────► direct WHY branch
    │
    └────────────► predicted WHAT
                         │
                         ▼
                  WHAT-guided reasoning
                         │
                         ▼
                    gated WHY output
                         │
                         ▼
                  residual reranker
               (base predictor frozen)
                         │
                         ▼
                    z_int ∈ R^256
```

The important architectural result is that **WHAT helps WHY when it lies on the computational path**, not merely as an auxiliary prediction target.

---

## Stage-A results

Three-seed validation results:

| Method | WHY MRR |
|---|---:|
| Direct WHY | 0.04754 ± 0.00608 |
| Predicted-WHAT-guided WHY | 0.05178 ± 0.00151 |
| + same-task WHAT hard negatives | 0.05401 ± 0.00227 |
| + Soft-MRR objective | 0.05454 ± 0.00231 |
| Frozen-base residual reranker | **0.05508 ± 0.00087** |
| Oracle WHAT diagnostic | 0.10546 ± 0.00665 |

### Interpretation

- Predicted WHAT improves WHY when used as a reasoning input.
- Same-task hard negatives help build more task-discriminative WHAT geometry.
- Oracle WHAT exposes substantial semantic headroom.
- The Oracle condition is **privileged diagnostic supervision**, not a deployable system.
- The selected Stage-A branch does **not** clearly exceed the strongest historical mean-pool result (`MRR ≈ 0.0589`), so Stage A is not treated as a solved problem.

---

## Cross-view diagnostic

ENIGMA-360 is used to study ego/exo representation alignment.

Current conclusion:

```text
ego ↔ exo alignment can be learned
            │
            ▼
but viewpoint alignment alone
does not reliably improve WHY retrieval
```

Cross-view learning is therefore kept as a **representation diagnostic**, not as a required component of the selected predictor.

---

# Stage B — Intention-conditioned SmolVLA

## Intention injection

The predicted latent is projected into the SmolVLA hidden dimension:

```text
z_int [B, 256]
      │
      ▼
Intention Adapter
      │
      ▼
[B, 1, 960]
```

The token is inserted as:

```text
[Image | Language | Intention | State]
```

The integration has been tested for:

- forward / backward training,
- gradient flow into the adapter,
- action sampling,
- KV-cache inference,
- baseline fallback without intention,
- mask consistency,
- checkpoint serialization,
- pretrained initialization compatibility.

---

# LIBERO schema validation

## Important historical note

Early custom matched-training runs inherited an incorrect action schema:

```text
action = 6-D
empty_cameras = 0
```

Those runs are **invalid for scientific comparison**.

The current validated LIBERO protocol requires:

```text
action = 7-D
state = 6-D
empty_cameras = 1
```

The active downstream trainer is:

```text
scripts/train_matched_oracle_LIBERO7D_FIXED.py
```

It copies the LIBERO feature schema from a validated standard SmolVLA checkpoint and hard-asserts the expected action and camera configuration.

---

# Current downstream status

Latest documented 10-episode smoke evaluations of the fixed 7-D baseline:

| Checkpoint | Success | Successful task |
|---:|---:|---|
| 1k | 0% | — |
| 5k | 10% | task 5 |
| 10k | 10% | task 2 |

These are **1 episode per task** smoke tests and should not be interpreted as final benchmark results.

A separate validated standard SmolVLA 25k checkpoint produced 30% success in the same small smoke-evaluation setting.

The main downstream comparison is:

```text
SmolVLA baseline
       │
       ├──────────────► Oracle semantic intention
       │
       ├──────────────► Predicted semantic intention
       │
       └──────────────► Controls
                         ├─ null token
                         ├─ shuffled intention
                         ├─ phase ID
                         └─ random code
```

The controls are necessary to separate **semantic utility** from simply adding extra conditioning capacity.

---

# Current status

| Component | Status |
|---|---|
| Stage-A visual baseline | ✅ Complete |
| Predicted-WHAT-guided WHY | ✅ Complete |
| Same-task WHAT hard negatives | ✅ Complete |
| Soft-MRR experiments | ✅ Complete |
| Frozen-base residual reranker | ✅ Complete |
| ENIGMA-360 cross-view diagnostic | ✅ Complete |
| SmolVLA intention-token integration | ✅ Complete |
| LIBERO 7-D schema validation | ✅ Complete |
| Fixed matched baseline training | 🚧 In progress / being evaluated |
| Oracle VLIA matched training | ⏳ Next |
| Null / shuffled / phase controls | ⏳ Planned |
| Predicted VLIA downstream evaluation | ⏳ Planned |
| Ambiguity / counterfactual evaluation | ⏳ Planned |
| UR3 / ROS2 deployment validation | ⏳ Planned |

---

# Repository layout

```text
vlia-uet/
├── config/
│   ├── data/
│   ├── suites/
│   └── tasks/
│
├── datasets/
│   └── enigma360_adapter.py
│
├── docs/
│   └── ROADMAP.md
│
├── models/
├── policies/
│   ├── intention/
│   └── smolvla/
│
├── results/
│
├── scripts/
│   ├── archive/
│   ├── experiments/
│   ├── train_matched_oracle_LIBERO7D_FIXED.py
│   ├── build_matched_eval_processors.py
│   ├── plot_training_history.py
│   └── ...
│
├── tests/
├── vlia_data/
│   └── libero_oracle_dataset.py
│
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

Historical ablations are intentionally retained for research provenance, but the active experimental path is documented in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

# Environment

VLIA currently uses a **separate LeRobot checkout** and a separate VLIA checkout.

## 1. Clone

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
git clone https://github.com/sgdfgnhmjnfd/vlia-uet.git ~/vlia-uet
```

## 2. Prepare LeRobot environment

```bash
cd ~/lerobot

python -m venv .venv
source .venv/bin/activate

pip install -e ".[smolvla]"
```

> **Important:** do not rely on `pip install -e ~/vlia-uet` as the primary setup yet.

The repository currently contains a local Python package named `datasets/`, which can shadow Hugging Face `datasets`.

---

# Running VLIA scripts safely

The tested execution pattern is:

```bash
cd ~/lerobot
source .venv/bin/activate
unset PYTHONPATH

SITEPKG=$(python -c 'import site; print(site.getsitepackages()[0])')

env PYTHONPATH="$SITEPKG:$HOME/vlia-uet" \
python "$HOME/vlia-uet/scripts/train_matched_oracle_LIBERO7D_FIXED.py" --help
```

This checks that the fixed 7-D trainer is importable without launching a training run.

## Verify imports

```bash
cd ~/lerobot
source .venv/bin/activate
unset PYTHONPATH

SITEPKG=$(python -c 'import site; print(site.getsitepackages()[0])')

env PYTHONPATH="$SITEPKG:$HOME/vlia-uet" python - <<'PY'
import datasets
import vlia_data

print("datasets:", datasets.__file__)
print("vlia_data:", vlia_data.__file__)
PY
```

Expected behavior:

```text
datasets  -> .../.venv/.../site-packages/datasets/__init__.py
vlia_data -> ~/vlia-uet/vlia_data/__init__.py
```

For other scripts, replace only the final script path with an **actual file under**:

```text
$HOME/vlia-uet/scripts/
```

Do not execute documentation placeholders such as `/path/to/script.py`.

---

# Dataset and output locations

Large datasets, model checkpoints, videos, cached embeddings, and generated features are intentionally excluded from Git.

Recommended external layout:

```text
$VLIA_DATA_ROOT/
├── egointent/
├── egointent_full/
├── enigma360/
└── libero_oracle/

$VLIA_OUTPUT_ROOT/
├── stage_a/
├── crossview/
└── libero/
```

The Oracle dataset wrapper supports:

```bash
export VLIA_ORACLE_ROOT=/path/to/libero_oracle/v0
```

Historical experiment scripts may still contain machine-specific default paths. Those defaults are retained only for provenance and should be overridden for new runs.

---

# Reproducibility checklist

Before reporting a downstream result, verify:

```text
[ ] action dimension = 7
[ ] state dimension = 6
[ ] empty_cameras = 1
[ ] intention token is inserted before state
[ ] baseline and Oracle share the same initialization
[ ] same optimizer and LR schedule
[ ] same training subset
[ ] same number of steps
[ ] same evaluation protocol
[ ] number of evaluation episodes is reported
[ ] Oracle is labeled as privileged supervision
[ ] 6-D historical runs are excluded
[ ] semantic controls are included before making causal claims
```

Training loss alone is **not** treated as downstream evidence.

---

# Research roadmap

The current critical path is:

```text
Fixed LIBERO 7-D baseline
          │
          ▼
Matched baseline evaluation
          │
          ▼
Oracle VLIA
          │
          ▼
Null / shuffle / phase controls
          │
          ▼
Predicted VLIA
          │
          ▼
Ambiguity & counterfactual evaluation
          │
          ▼
UR3 / ROS2 / MoveIt2 validation
```

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed experiment plan.

---

# Scientific scope

The project currently treats explicit intention as a **semantic control variable** rather than as an estimate of a person's hidden mental state.

The strongest downstream claim will require evidence that:

1. Oracle semantic intention is useful under a matched protocol.
2. The gain is not reproduced by null, shuffled, phase-only, or random conditioning.
3. A predicted intention representation preserves enough of that utility to improve robot behavior.
4. The effect is strongest in conditions where observation and language are genuinely ambiguous.

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**VLIA-UET — from seeing actions to reasoning about intent, then acting with it.**

</div>
