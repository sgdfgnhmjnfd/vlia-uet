# VLIA-UET

## Vision · Language · Intention · Action

### Early Robot-Relevant Goal Disambiguation from Egocentric Manipulation Video

VLIA-UET investigates **early robot-relevant goal disambiguation from
egocentric manipulation video**.

The central research question is:

> **How early can a robot-relevant goal be disambiguated from egocentric
> manipulation video when multiple candidate goals share the same
> observed sub-actions?**

The project focuses on intention/goal prediction as the primary research
problem. A VLA policy is treated as a downstream interface rather than
the main scientific contribution.

------------------------------------------------------------------------

## 1. Research Problem

In egocentric manipulation videos, the same early action can be
compatible with several different future goals.

For example:

``` text
pick up cup
   ├── move to shelf
   ├── put into box
   └── move to tray
```

The observed sub-action alone may therefore be insufficient to determine
the intended goal.

The project studies this ambiguity as a **fine-grained goal
disambiguation problem**, with particular attention to cases where
candidate goals share the same early manipulation behavior.

The target is not generic action recognition. The goal is to infer a
**robot-relevant intended outcome** before the final trajectory or
outcome becomes explicit.

------------------------------------------------------------------------

## 2. Research Hypothesis

The current hypothesis is:

> When multiple goals are compatible with the same observed sub-action,
> explicitly conditioning goal inference on the predicted sub-action can
> provide a useful inductive bias for fine-grained goal disambiguation.

The proposed reasoning structure is:

``` text
Egocentric video prefix
        │
        ▼
 Frozen visual representation
        │
        ├──────────────► WHAT
        │                observed sub-action
        │
        ▼
      WHY
 intended goal / purpose
        │
        ▼
 Robot-relevant intention
```

Here:

-   **WHAT** describes what is currently being done.
-   **WHY** describes the intended purpose or goal.
-   **Intention** is the robot-relevant goal representation derived from
    WHY.

The WHAT → WHY relationship is treated as a **research
hypothesis/mechanism**, not as an established bottleneck.

------------------------------------------------------------------------

## 3. Research Questions

### RQ1 --- Goal disambiguation

Can a model distinguish between robot-relevant goals that share the same
early sub-actions?

### RQ2 --- Role of observed sub-actions

Does explicitly conditioning goal inference on WHAT improve fine-grained
goal prediction compared with direct goal prediction?

### RQ3 --- Early anticipation

How early can the intended goal be inferred while excluding evidence
that directly reveals the final outcome?

### RQ4 --- Robot relevance

Can predicted goals be mapped reliably to robot-relevant skills or
targets?

------------------------------------------------------------------------

## 4. Method

The main comparison is deliberately simple.

### A. Direct WHY

``` text
Video prefix → visual representation → WHY
```

### B. Auxiliary WHAT + WHY

``` text
Video prefix → shared representation
                    ├── WHAT supervision
                    └── WHY prediction
```

### C. Predicted WHAT → WHY

``` text
Video prefix
      │
      ▼
    WHAT
      │
      ▼
    WHY
```

The third configuration is the main proposed mechanism.

### D. Oracle WHAT

``` text
Video prefix + ground-truth WHAT → WHY
```

Oracle WHAT is used only as a privileged diagnostic to estimate the
remaining headroom. It is not a deployable setting and must not be
presented as the main method.

------------------------------------------------------------------------

## 5. Strong Baseline

A simple temporal mean-pooling baseline is included because temporal
architecture alone should not be assumed to solve the problem.

``` text
8 sampled frames
      │
      ▼
Frozen visual encoder
      │
      ▼
Mean pooling
      │
      ▼
WHY prediction
```

This baseline is important because previous GRU and Transformer temporal
variants did not outperform mean pooling in preliminary experiments.

Therefore, the project does not rely on increasing temporal model
complexity as the main research contribution.

------------------------------------------------------------------------

## 6. Same-Task Goal Disambiguation

Preliminary error analysis indicates that many top-1 failures are not
broad task mistakes. They are **same-task goal confusions**.

Current preliminary analysis:

``` text
510 / 626 Top-1 errors
≈ 81.5%
```

were same-task confusions.

This motivates a dedicated evaluation setting in which candidate goals
share the same observed task/sub-action but differ in their intended
outcome.

The benchmark should therefore report:

-   Overall performance
-   Cross-task performance
-   Same-task performance

This makes the evaluation target more specific than generic action
anticipation.

------------------------------------------------------------------------

## 7. Temporal Anticipation

Fixed observation percentages are useful for controlled experiments:

-   10%
-   25%
-   50%
-   75%

However, percentage-of-video alone is not a semantic definition of
anticipation difficulty.

The stronger protocol is based on **trajectory/outcome exclusion**.

Let:

-   `t_obs` = end of the observed prefix
-   `t_divergence` = point at which physical motion begins to reveal the
    intended goal
-   `Δ` = anticipation margin

Then an observation is valid when:

``` text
t_obs ≤ t_divergence - Δ
```

This prevents the model from being evaluated after the goal has already
become physically obvious.

A pilot annotation of approximately 100--200 clips can first be used to
verify whether fixed percentage cuts actually occur before goal-specific
evidence becomes explicit.

If they do not, time-to-divergence or time-to-outcome should be
preferred.

------------------------------------------------------------------------

## 8. Data and Evaluation

### Main benchmark

The current Stage-A experiments use EgoIntent.

Preliminary setup:

-   8 videos
-   6 indoor training videos
-   2 outdoor validation videos
-   556 training microsteps
-   215 validation samples
-   split by `video_uid`
-   3 random seeds
-   MRR retrieval metric

The current validation set has been used during tuning, so these results
should **not** be treated as final generalization results.

A clean untouched test split is required for the final benchmark.

------------------------------------------------------------------------

## 9. Preliminary Stage-A Results

Current preliminary MRR results:

  Configuration                                       MRR
  ------------------------------- -----------------------
  Direct WHY                        `0.047538 ± 0.006076`
  Auxiliary WHAT + WHY                  `0.0471 ± 0.0030`
  Predicted-WHAT-guided WHY         `0.051780 ± 0.001507`
  WHAT same-task hard-negative      `0.054014 ± 0.002265`
  Soft-MRR                          `0.054538 ± 0.002305`
  Frozen-base residual reranker     `0.055083 ± 0.000865`
  Mean-pool 8-frame baseline          `0.05887 ± 0.00159`
  Oracle WHAT                           `0.1055 ± 0.0067`

### Interpretation

The current results do **not** demonstrate that the proposed method
beats the strongest simple baseline.

In particular:

-   Predicted WHAT → WHY is more promising than auxiliary WHAT
    supervision alone.
-   Oracle WHAT shows substantial diagnostic headroom.
-   The mean-pool baseline currently remains stronger.
-   The current evidence is insufficient for a claim of superiority.
-   Repeated tuning on the same validation set means these results are
    preliminary.

The correct scientific response is to test the hypothesis more carefully
rather than add increasingly complex architectures without evidence.

------------------------------------------------------------------------

## 10. Required Ablations

The minimum experimental matrix should include:

### Core ablation

1.  Direct WHY
2.  Auxiliary WHAT + WHY
3.  Predicted WHAT → WHY
4.  Oracle WHAT → WHY

### Parameter-matched control

A Direct WHY model with comparable additional parameters should be
compared against WHAT → WHY.

This tests whether any improvement comes from the proposed conditioning
mechanism rather than simply increased model capacity.

### Hard-negative evaluation

Compare:

``` text
Normal objective
vs.
Same-task hard-negative objective
```

The hard-negative setting directly targets the dominant failure mode.

### Temporal evaluation

Evaluate only prefixes that satisfy the temporal exclusion protocol.

### OOD evaluation

Use the self-collected dataset as an out-of-distribution stress test
rather than as a large-scale training set.

------------------------------------------------------------------------

## 11. Robot Relevance

The robotics component should remain lightweight.

Instead of training a complete robot policy, use a downstream proxy:

``` text
Predicted goal
      │
      ▼
Robot skill / target selection
```

Example:

``` text
"move cup to shelf"
        ↓
ShelfPlacementSkill
```

Possible metrics:

-   Skill-selection accuracy
-   Target-selection accuracy
-   Top-k skill retrieval

This provides a measurable connection to robot execution without turning
robot policy learning into a second research problem.

------------------------------------------------------------------------

## 12. SmolVLA Integration

SmolVLA is used as a downstream VLA interface and engineering testbed.

The VLIA integration adds one intention token:

``` text
Baseline:
[Image tokens ; Language tokens ; State]

VLIA:
[Image tokens ; Language tokens ; Intention token ; State]
```

The current implementation has been checked for:

-   Intention adapter shape
-   Prefix length increase by one token
-   Intention token placement before state
-   Attention masks
-   Forward pass
-   Backward pass
-   Gradient propagation to the adapter
-   Action sampling
-   KV cache
-   Training/inference paths
-   Baseline fallback
-   Pretrained initialization
-   Matched preprocessing
-   Semantic action padding
-   Checkpoint serialization

These checks establish **implementation correctness**.

They do not establish scientific benefit from intention conditioning.

------------------------------------------------------------------------

## 13. LIBERO Status

LIBERO is not the primary research benchmark for the current direction.

Earlier matched runs exposed schema mismatches, including:

-   Action dimension mismatch
-   Camera/empty-camera configuration mismatch

The canonical setup uses:

``` text
state: [6]
action: [7]
camera1 / camera2 / camera3
empty_camera_0
empty_cameras = 1
```

The current research direction therefore does not depend on large-scale
LIBERO policy training.

LIBERO/SmolVLA can remain useful as an optional downstream demonstration
after the intention prediction experiments are established.

------------------------------------------------------------------------

## 14. Self-Collected OOD Dataset

A small egocentric manipulation dataset is planned for
out-of-distribution evaluation.

Current pilot design:

-   2 participants
-   12 tasks
-   2 repetitions
-   48 videos
-   Chest/neck-mounted egocentric camera
-   Hands, objects, workspace, and targets visible
-   No verbalized intention

The pilot task taxonomy includes:

``` text
T01  Cup → Shelf
T02  Bottle → Box
T03  Block → Tray
T04  Sort Blocks
T05  Sort Objects
T06  Object → Target
T07  Peg → Hole
T08  Object → Container
T09  Simple Assembly
T10  Component → Base
T11  Open → Retrieve
T12  Retrieve → Put Back
```

The exact taxonomy should be finalized only after checking the actual
objects and hardware used during collection.

### Annotation schema

``` text
video_id
participant_id
task_id
repetition_id
video_duration
observation_end
outcome_start
object
target
what
why
next
intention_id
```

Definitions:

-   `WHAT` = `[verb] + [object]`
-   `WHY` = intended purpose/goal, not a paraphrase of WHAT
-   `NEXT` = likely next manipulation step

With only two participants, participant-disjoint machine-learning
training is not a reliable objective. The pilot should instead serve
primarily as an OOD stress test.

------------------------------------------------------------------------

## 15. Experimental Roadmap

The recommended order is:

``` text
Clean train/validation/test split
            ↓
Strong mean-pool baseline
            ↓
Direct WHY
            ↓
Auxiliary WHAT
            ↓
Predicted WHAT → WHY
            ↓
Parameter-matched control
            ↓
Oracle WHAT diagnostic
            ↓
Same-task hard negatives
            ↓
Temporal exclusion protocol
            ↓
Same-task evaluation
            ↓
OOD evaluation
            ↓
Robot skill-selection proxy
```

If WHAT → WHY still fails to outperform the strongest simple baseline
under a clean evaluation protocol, the hypothesis should be weakened or
rejected rather than protected by adding arbitrary architectural
complexity.

------------------------------------------------------------------------

## 16. Repository Structure

``` text
vlia-uet/
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt
├── src/
├── scripts/
├── configs/
├── tests/
├── docs/
│   ├── architecture/
│   ├── experiments/
│   └── notes/
├── results/
│   └── README.md
└── artifacts/
    └── checkpoint_025000/
        └── pretrained_model/
```

### Recommended source organization

``` text
src/
├── intention/
├── models/
├── datasets/
├── evaluation/
└── utils/
```

------------------------------------------------------------------------

## 17. Reproducibility Principles

All experiments should record:

-   Dataset version
-   Split definition
-   Model configuration
-   Random seed
-   Training configuration
-   Evaluation protocol
-   Observation horizon
-   Hard-negative definition
-   Retrieval metric
-   Checkpoint used

Final claims should be based on an untouched test set whenever possible.

Validation results used repeatedly during development must be labeled as
preliminary.

------------------------------------------------------------------------

## 18. What Belongs in Git

### Commit

-   Source code
-   WHAT/WHY/intention code
-   Intention Adapter
-   SmolVLA integration
-   Integration tests
-   Stage-A experiment code
-   Configuration files
-   Training/evaluation scripts
-   Results and metrics
-   README and documentation
-   Environment information
-   Architecture diagrams/notes

### Do not commit

-   Raw videos
-   Full datasets
-   Hugging Face/model caches
-   Virtual environments
-   Full `~/lerobot`
-   Large temporary logs
-   Dataset caches
-   Rendered videos

Large selected checkpoints may be stored with Git LFS when explicitly
required.

------------------------------------------------------------------------

## 19. Scientific Scope

The primary contribution is:

> **A focused study of early robot-relevant goal disambiguation from
> egocentric manipulation video, with explicit evaluation of same-task
> goal confusion and temporal exclusion.**

The project does not claim to solve general human intention
understanding.

It does not claim that the robot can autonomously understand arbitrary
human intentions.

It does not make large-scale VLA policy learning the central research
problem.

Instead, the research isolates the intention-prediction problem and
evaluates whether the inferred goal can provide useful information for a
downstream robot interface.

------------------------------------------------------------------------

## 20. Future Work

Potential future directions include:

-   Larger participant-diverse egocentric datasets
-   Stronger trajectory-aware temporal modeling
-   Explicit uncertainty estimation
-   Multi-modal cues such as gaze
-   Online intention updating
-   Closed-loop human-robot interaction
-   Integration with full VLA policy execution
-   More realistic robot skill planning
-   Larger-scale cross-dataset evaluation

These are future directions rather than requirements for the current
core contribution.

------------------------------------------------------------------------

## License

This repository is intended for academic/research use. Add the final
project-specific license before public release.