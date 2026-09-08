# INSIGHT Source Audit

## Scope

This audit inspects the public INSIGHT source code to determine which parts are
useful for VLIA intention supervision.

## Hand-Object Stage

The public `HandObject/dataset.py` groups actions by `clip_uid`, orders them by
`action_idx`, and constructs temporal windows.

The default window size is 8 actions.

Each action step contains:

- full-frame feature tensor
- hand/object mask feature tensor
- verb label
- noun label
- action ID

The loader uses pre-extracted feature tensors rather than raw video frames.

## Cognitive Reasoning Stage

The Cognitive Reasoning code uses an RLHF/GRPO chat-style data format.

The target output has the structure:

`<think>...</think><intention>...</intention><answer>verb noun, ...</answer>`

The model is designed to:

1. reason about visual context,
2. infer intention,
3. predict future semantic actions.

The reward code can receive `gt_intention` directly or extract intention text
from tagged ground-truth content.

The code also evaluates intention and future actions separately.

## Public Data Availability

The public training script refers to placeholder dataset paths such as:

`.../data_Ego4D/...`

The processed Cognitive Reasoning dataset is not included in the repository.

Therefore, the public source does not establish:

- the complete processed dataset schema,
- the exact provenance of the intention annotations,
- the exact intention generation pipeline.

These properties must not be assumed.

## License

The public INSIGHT repository code is released under the MIT License.

Dataset licenses and access requirements remain separate from the code license.

## Reuse Decision for VLIA

VLIA will reuse the conceptual structure:

`observed context -> semantic intention -> future semantic actions`

The complete INSIGHT pipeline is not a mandatory VLIA dependency.

The following components are optional:

- hand-object recognition
- explicit HOI features
- GRPO optimization
- INSIGHT-specific reward functions

## Audit Result

Verified from public source:

- temporal observed-action windowing
- verb/noun semantic action representation
- explicit intention output
- future semantic action output
- separate intention/action evaluation
- MIT code license

Not exposed by public source:

- processed Cognitive Reasoning dataset
- complete processed data schema
- exact intention annotation provenance
- exact intention generation pipeline
