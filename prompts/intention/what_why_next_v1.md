# VLIA What-Why-Next Annotation Prompt V1

## Role

You annotate phase-level semantic intentions for egocentric human
manipulation trajectories.

Given:

- TASK: the global objective of the trajectory.
- PLAN: a high-level textual plan.
- HISTORY: semantic actions already observed before the current boundary.
- FUTURE: semantic actions occurring after the current boundary.

Generate exactly three fields:

WHAT:
A concise description of the semantic situation at the current boundary.

WHY:
The local semantic purpose of the current behavioral phase.

NEXT:
The immediate semantic subgoal following the current boundary.

## Core definition

WHY represents the purpose of the current phase.

WHY is NOT:

- a repetition or paraphrase of TASK;
- a repetition or paraphrase of NEXT;
- a low-level motor command;
- merely the next verb-action pair.

WHY should explain what the current phase accomplishes toward the
larger task.

## Temporal constraint

The global TASK may remain constant across a trajectory, but WHY should
change when the semantic phase changes.

## Information-use constraint

HISTORY and TASK represent information potentially available to the
intention predictor.

PLAN and FUTURE may be used only by this offline annotation process to
infer the intended semantic phase.

They must not be treated as inputs available to the final intention
encoder at inference time.

## Abstraction

Prefer semantic purposes such as:

- gain access to the target container
- acquire the object needed for the task
- transfer the object to the target location
- restore the container to its closed state
- prepare the object for manipulation

Avoid outputs such as:

- open lid
- pick up cup
- close drawer

when these simply duplicate NEXT.

## Example

TASK:
puts clothes in the washing machine

PLAN:
collect the clothes. open the lid of the washing machine.
put the clothes in the washing machine. close the lid.

HISTORY:
collect(clothes)

FUTURE:
open(lid)
put in(clothes, washing machine)
close(lid)

Output:

WHAT:
The clothes have been collected and the washing machine has not yet
been accessed.

WHY:
Gain access to the target container for the upcoming transfer.

NEXT:
Open the washing machine lid.

## Output format

Return only:

WHAT: <text>
WHY: <text>
NEXT: <text>
