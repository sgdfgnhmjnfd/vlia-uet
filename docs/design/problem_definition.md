# VLIA Problem Definition

## Problem

Vision-Language-Action policies condition robot actions on visual observations,
language instructions, and robot states. In long-horizon manipulation, the
global task may remain unchanged while the semantic purpose of the current
behavior changes across manipulation phases.

VLIA introduces an explicit intention representation as an additional
conditioning variable for the action policy.

## Intention Definition

Intention is defined as:

> A phase-level semantic representation of why the agent is performing the
> current manipulation behavior.

It is distinguished from:

- **Task**: the global episode objective.
- **What**: the current observed state or ongoing behavior.
- **Why / Intention**: the semantic objective of the current manipulation phase.
- **Next**: the immediate next subgoal or phase.
- **Action**: the embodiment-specific motor command.

## Formulation

The VLIA policy is formulated as:

`Action = Policy(Vision, Language, State, Intention)`

The current design uses an independent intention latent representation
`z_int` with `d_int = 256`.
