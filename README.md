# VLIA-UET

**VLIA (Vision-Language-Intention-Action)** is a research project on explicit semantic intention conditioning for Vision-Language-Action (VLA) robot policies.

The current research question is:

> **Does an explicit semantic intention representation provide useful information to a VLA policy beyond the robot observation and task instruction alone?**

The project separates the problem into two stages:

1. **Stage A — Intention reasoning from egocentric human video**
2. **Stage B — Intention-conditioned robot policy learning with SmolVLA**

The main downstream benchmark is **LIBERO-10**. Human intention prediction is studied on **EgoIntent**, with cross-view diagnostics using **ENIGMA-360**.

> This repository is an active research codebase. Some scripts are retained for ablation history and are not part of the current main training path.

---

## 1. Method overview

```text
Stage A: Human intention reasoning
─────────────────────────────────────────────────────────────

Egocentric video
      │
      ▼
Frozen visual backbone
(SmolVLM2-500M-Video-Instruct)
      │
      ▼
Visual representation
      │
      ├──────────────► WHAT predictor
      │                    │
      │                    ▼
      └──────────────► WHAT-guided WHY reasoner
                           │
                           ▼
                  Frozen residual reranker
                           │
                           ▼
                    z_int ∈ R^256


Stage B: Intention-conditioned robot policy
─────────────────────────────────────────────────────────────

z_int ∈ R^256
      │
      ▼
Intention adapter
256 → 960
      │
      ▼
one intention token
      │
      ▼
[Image | Language | Intention | State]
      │
      ▼
SmolVLA
      │
      ▼
robot action chunk
```

The architectural intervention is intentionally lightweight: a single intention token is inserted into the SmolVLA prefix before the state token.

---

## 2. Research hypothesis

Standard VLA policies already receive visual observations and a language instruction, but these inputs can still be ambiguous about the **local semantic purpose of the current behavior**.

VLIA therefore treats intention as a compact semantic control variable. The intention representation is designed to capture information such as:

- **What** is currently being done,
- **Why** that behavior is being performed,
- and, where useful for supervision, the likely **next semantic step**.

The main scientific test is downstream:

> If intention contains useful task-relevant information, adding a valid intention signal should improve robot-policy performance under a matched training protocol.

This motivates three principal downstream conditions:

```text
SmolVLA baseline
        vs.
Oracle VLIA
        vs.
Predicted VLIA
```

Additional controls such as null, shuffled, phase-ID, and random-code intention are used to test whether any gain is genuinely semantic rather than merely caused by adding an extra token.

---

## 3. Stage A — Intention reasoning

### 3.1 Dataset

Stage-A experiments use **EgoIntent** egocentric clips.

Current pilot protocol:

- 8 RGB frames per clip
- 556 training samples
- 215 validation samples
- split by `video_uid`
- no train/validation video overlap
- primary target: semantic **WHY**

The predictor does **not** use future actions or future annotations as input.

### 3.2 Key empirical finding

A large fraction of Stage-A retrieval errors occur between samples from the **same task**. This indicates that the main difficulty is not broad task recognition, but fine-grained within-task semantic disambiguation.

This motivated the current reasoning path:

```text
visual feature
    │
    ├──► direct WHY branch
    │
    └──► predicted WHAT
              │
              ▼
       WHAT-guided reasoning
              │
              ▼
        gated WHY output
              │
              ▼
     frozen residual reranker
```

### 3.3 Stage-A results

Selected three-seed results:

| Method | WHY MRR |
| --- | ---: |
| Direct WHY | 0.04754 ± 0.00608 |
| Predicted-WHAT-guided WHY | 0.05178 ± 0.00151 |
| + same-task WHAT hard negatives | 0.05401 ± 0.00227 |
| + Soft-MRR objective | 0.05454 ± 0.00231 |
| Frozen-base residual reranker | **0.05508 ± 0.00087** |
| Oracle WHAT diagnostic | 0.10546 ± 0.00665 |

The Oracle-WHAT result is a **privileged diagnostic**, not a deployable method.

An important caveat is that the best selected guided branch does **not** clearly exceed the strongest historical mean-pool result (`MRR ≈ 0.0589`). Therefore, Stage A should not be presented as a solved intention-prediction problem. Its main value is to establish a semantic reasoning pipeline and quantify the remaining headroom.

### 3.4 Cross-view diagnostic

Cross-view experiments on ENIGMA-360 show that a learned projection can align ego and exo representations, but that viewpoint alignment did not consistently improve EgoIntent WHY retrieval.

Current interpretation:

> cross-view learning provides a useful representation diagnostic, but viewpoint alignment alone is not sufficient to solve fine-grained semantic intention reasoning.

---

## 4. Stage B — VLIA / SmolVLA integration

The intention latent is mapped into the SmolVLA hidden space:

```text
z_int [B, 256]
      │
      ▼
IntentionAdapter
      │
      ▼
[B, 1, 960]
```

The resulting token is inserted before the state token:

```text
[task/language tokens] [image tokens] [INTENTION] [state token]
```

The current implementation supports:

- training with intention conditioning,
- inference with intention conditioning,
- baseline fallback with no intention,
- KV-cache inference,
- action sampling,
- gradient flow into the intention adapter,
- checkpoint serialization.

---

## 5. LIBERO schema validation

### Important: invalid historical matched runs

Early custom matched-training runs accidentally inherited the wrong output schema from the base checkpoint:

```text
action shape = 6
empty_cameras = 0
```

Canonical LIBERO evaluation requires:

```text
action shape = 7
state shape  = 6
empty_cameras = 1
```

Therefore, the old `oracle_matched_full` 6-D baseline and Oracle runs are retained only for debugging/provenance and **must not be used as reported downstream results**.

### Current fixed protocol

The active trainer is:

```text
scripts/train_matched_oracle_LIBERO7D_FIXED.py
```

It copies the input/output feature schema from a validated standard LIBERO SmolVLA checkpoint and asserts:

```text
action = 7-D
state = 6-D
empty_cameras = 1
```

This is the downstream protocol used for new matched experiments.

---

## 6. Current downstream status

The fixed 7-D baseline is trained on LIBERO-10 using a matched protocol.

Current 10-episode smoke evaluations:

| Checkpoint | Success rate | Successful task |
| ---: | ---: | --- |
| 1k | 0% | — |
| 5k | 10% | task 5 |
| 10k | 10% | task 2 |

These evaluations use only one episode per task and are therefore **high-variance smoke tests**, not final performance estimates.

A separate validated standard SmolVLA 25k checkpoint produced 30% success in the same 10-episode smoke setting. Final claims require matched evaluation with more episodes.

The current critical path is:

```text
Fixed baseline 25k
       │
       ▼
Matched baseline evaluation
       │
       ▼
Oracle VLIA 25k
       │
       ▼
Null / shuffled / phase-ID controls
       │
       ▼
Predicted VLIA
       │
       ▼
Ambiguity / counterfactual evaluation
```

---

## 7. Repository structure

```text
vlia-uet/
├── config/
│   ├── data/
│   ├── suites/
│   └── tasks/
├── datasets/
├── docs/
├── models/
├── policies/
│   ├── intention/
│   └── smolvla/
├── results/
├── scripts/
│   ├── archive/
│   ├── tools/
│   ├── train_matched_oracle_LIBERO7D_FIXED.py
│   ├── build_matched_eval_processors.py
│   ├── plot_training_history.py
│   └── ...
├── tests/
├── vlia_data/
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

The repository contains historical ablation scripts. The files listed in the roadmap define the current main experimental path.

---

## 8. Environment

VLIA is developed on top of the Hugging Face **LeRobot** stack.

A typical setup is:

```bash
git clone https://github.com/huggingface/lerobot.git
git clone https://github.com/sgdfgnhmjnfd/vlia-uet.git

cd lerobot
python -m venv .venv
source .venv/bin/activate

pip install -e ".[smolvla]"
pip install -e ../vlia-uet
```

### Local `datasets/` name collision

This repository contains a local Python package named `datasets/`, which can shadow Hugging Face `datasets` when commands are launched from the repository root.

The current tested workaround is to launch LeRobot-dependent commands from the LeRobot repository and put site-packages before the VLIA repository on `PYTHONPATH`:

```bash
cd /path/to/lerobot
source .venv/bin/activate

SITEPKG=$(python -c 'import site; print(site.getsitepackages()[0])')

env PYTHONPATH="$SITEPKG:/path/to/vlia-uet" \
python /path/to/vlia-uet/scripts/<script>.py
```

A future cleanup should rename the local `datasets/` package to remove this collision.

---

## 9. Data and output paths

Large datasets, model checkpoints, videos, and generated features are intentionally not committed to Git.

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

Many historical scripts still contain local default paths from the original development machine. For public/reproducible use, pass paths explicitly through command-line arguments when available. Path cleanup is ongoing.

---

## 10. Main validation targets

Before reporting a downstream run, verify:

```text
LIBERO action dimension     = 7
SmolVLA state dimension     = 6
empty_cameras               = 1
intention token position    = before state token
baseline/oracle initialization matched
same optimizer/scheduler protocol
same training subset
same evaluation protocol
```

The repository includes tests for prefix injection, action sampling, preprocessing, semantic action padding, checkpoint initialization, and intention-path integration.

---

## 11. Reproducibility principles

The project follows the following experimental rules:

- Stage-A intention prediction is evaluated separately from robot action prediction.
- Train/validation splits are separated by source video.
- Future annotations are not used as predictor input.
- Oracle intention is treated only as privileged supervision/diagnostic information.
- Baseline and intention-conditioned policies use matched training settings.
- Training loss alone is not considered evidence of downstream improvement.
- Smoke evaluations are not treated as final benchmark results.
- Historical invalid 6-D LIBERO runs are excluded from scientific comparison.
- Final semantic claims require controls against null, shuffled, or non-semantic intention tokens.

---

## 12. Current roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

The immediate milestones are:

1. finish fixed 7-D baseline training;
2. run matched baseline evaluation;
3. train and evaluate Oracle VLIA with the same schema;
4. add semantic-control ablations;
5. connect the Stage-A predictor to the downstream policy;
6. evaluate predicted VLIA;
7. test ambiguity/counterfactual settings where explicit intention should matter most;
8. validate the final system on the UR3/ROS2/Gazebo stack.

---

## License

See [`LICENSE`](LICENSE).
