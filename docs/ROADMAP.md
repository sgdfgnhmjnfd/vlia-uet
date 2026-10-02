# VLIA Research Roadmap

This document tracks the **current** research path. Historical experiments are retained in the repository for provenance, but they are not all part of the main pipeline.

---

## Goal

The project tests whether an explicit semantic intention representation can improve a Vision-Language-Action policy under observation/task ambiguity.

The final target pipeline is:

```text
Human egocentric video
        │
        ▼
Stage-A intention predictor
        │
        ▼
z_int ∈ R^256
        │
        ▼
Intention adapter 256→960
        │
        ▼
SmolVLA prefix token
        │
        ▼
Robot action policy
```

---

## Phase A — Human intention reasoning

### A1. Visual baseline

Input:

```text
8 egocentric RGB frames
```

Backbone:

```text
SmolVLM2-500M-Video-Instruct
```

The backbone is used to extract frozen visual features.

Primary supervision:

```text
WHY / semantic intention
```

### A2. Semantic reasoning

Current selected branch:

```text
visual feature
    │
    ├──► direct WHY branch
    │
    └──► predicted WHAT
              │
              ▼
       WHAT-guided WHY reasoner
              │
              ▼
       frozen residual reranker
              │
              ▼
          z_int [256]
```

Main Stage-A conclusion:

- fine-grained within-task semantic disambiguation is the dominant failure mode;
- predicted WHAT is useful when it lies on the computational path to WHY;
- same-task WHAT hard negatives improve task-discriminative geometry;
- Oracle WHAT exposes substantial semantic headroom;
- more Stage-A architectural sweeps are lower priority than downstream validation.

### A3. Cross-view diagnostic

ENIGMA-360 is used to test ego/exo representation alignment.

Current conclusion:

- learned cross-view alignment is measurable;
- cross-view alignment did not reliably improve EgoIntent WHY retrieval;
- cross-view remains a representation diagnostic rather than a core component of the selected predictor.

---

## Phase B — SmolVLA intention integration

### B1. Intention adapter

```text
z_int [256]
   │
   ▼
Linear / adapter
   │
   ▼
intention token [960]
```

### B2. Prefix integration

The intention token is inserted before the state token:

```text
[Image | Language | Intention | State]
```

Required checks:

- prefix length increases by exactly one token;
- masks remain valid;
- training forward pass works;
- gradients reach the adapter;
- action sampling works;
- KV-cache inference works;
- baseline fallback works;
- checkpoint serialization works.

These checks are already implemented in the current codebase.

---

## Phase C — Matched LIBERO-10 downstream evaluation

### C0. Schema requirement

All reported LIBERO experiments must use:

```text
action dimension = 7
state dimension  = 6
empty_cameras    = 1
```

Historical custom matched runs using a 6-D action schema are invalid for scientific comparison and are retained only for provenance/debugging.

### C1. Fixed baseline

Active trainer:

```text
scripts/train_matched_oracle_LIBERO7D_FIXED.py
```

Target:

```text
25,000 optimization steps
```

Evaluation:

- 10-episode smoke tests during training;
- larger matched evaluation at the final checkpoint.

### C2. Oracle VLIA

Train from scratch with the same:

- model initialization,
- dataset subset,
- action/state schema,
- optimizer,
- scheduler,
- seed,
- batch size,
- number of steps.

Only the intention condition differs.

Oracle intention is privileged and is used to answer:

> Can the policy exploit an explicit semantic intention signal at all?

### C3. Semantic controls

Before claiming that semantic intention helps, compare against:

```text
Baseline
Null intention token
Shuffled intention
Phase-ID / random code
Oracle semantic intention
```

These controls distinguish:

- semantic information,
- generic extra-token capacity,
- task-phase leakage,
- and random conditioning effects.

### C4. Predicted VLIA

After Oracle utility is established, connect the Stage-A predictor to the robot policy:

```text
egocentric video
   ↓
Stage-A predictor
   ↓
z_int [256]
   ↓
adapter
   ↓
SmolVLA
```

This is the deployable condition.

---

## Phase D — Ambiguity-focused evaluation

Standard LIBERO language instructions can already be highly informative. Therefore, a stronger test should create conditions where intention can matter more directly.

Planned evaluations:

1. paired ambiguous observations with different intended outcomes;
2. counterfactual intention swaps;
3. same observation + same language + different intention;
4. history-only implicit-intention baseline;
5. semantic WHY vs phase-ID/random-code controls.

The target claim is not merely that "an extra token helps", but that:

> explicit semantic intention acts as a disambiguating control variable for VLA behavior.

---

## Phase E — Robot-system validation

After simulation evidence is stable:

```text
UR3 + ROS2 + MoveIt2 + RealSense
```

Use the real/sim robot stack as a sanity check for:

- interface compatibility,
- observation formatting,
- action execution,
- intention-conditioned policy integration.

This is not a substitute for matched benchmark evaluation.

---

## Current priority order

```text
1. Finish fixed 7-D baseline 25k
2. Evaluate baseline
3. Train fixed 7-D Oracle 25k
4. Build Oracle evaluator with intention injection
5. Run null/shuffle/phase controls
6. Integrate predicted Stage-A intention
7. Evaluate predicted VLIA
8. Add ambiguity/counterfactual benchmark
9. UR3/ROS2/Gazebo validation
```

---

## Reporting rules

Do not report:

- old 6-D matched runs as valid LIBERO results;
- Oracle intention as deployable;
- 10-episode smoke tests as final benchmark performance;
- training loss as evidence of policy success;
- a Stage-A improvement without stating the comparison baseline.

Do report:

- exact schema,
- matched training protocol,
- number of evaluation episodes,
- success rate,
- per-task results,
- seed information,
- semantic-control ablations,
- limitations and domain gap.
