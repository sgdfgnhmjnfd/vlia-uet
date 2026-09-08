# VLIA V0 Scope

## Core Contribution

The core contribution is the explicit semantic intention representation and
its integration into a Vision-Language-Action policy.

## Mandatory

- Phase-level semantic Why definition
- Independent intention representation
- IntentionAdapter and intention token
- SmolVLA baseline
- Oracle VLIA
- Predicted VLIA
- Intention evaluation
- Downstream policy evaluation

## Supporting Components

- INSIGHT-style data: intention supervision source
- Robosuite: controlled data sandbox
- UR3 Gazebo: deployment platform
- VLIA intention representation: core research contribution

## Optional

The following components can be removed first if time is limited:

- GRPO
- Explicit HOI modeling
- Long-history modeling
- Additional simulator tasks
- Additional auxiliary objectives

## V0 Completion Criterion

VLIA V0 must support a fair comparison of:

SmolVLA vs Oracle VLIA vs Predicted VLIA
