from datasets.phase_taxonomy import classify_phase
def action_to_text(action):
    if not action:
        return ""

    return (
        action.replace("(", " ")
        .replace(")", "")
        .replace(",", " in relation to ")
        .strip()
    )


def label_boundary_v0(sample):
    history = sample["history"]["semantic_actions"]
    future = sample["future_actions"]
    task = sample["language"]["task"]

    if not future:
        raise ValueError("Boundary sample has no future actions.")

    next_action = future[0]
    next_text = action_to_text(next_action)

    if history:
        previous_text = action_to_text(history[-1])
        what = f"Completed {previous_text}."
    else:
        what = f"At the beginning of the task: {task}."

    next_label = next_text.capitalize() + "."

    # V0 deterministic pseudo-intention.
    # This is deliberately simple and must not be treated as
    # ground-truth intention supervision.
    if len(future) > 1:
        why = (
            f"Prepare to {next_text} as part of accomplishing "
            f"the task: {task}."
        )
    else:
        why = f"Complete the task by {next_text}."

    result = dict(sample)
    result["intention"] = {
        "what": what,
        "why": why,
        "next": next_label,
        "phase": f"boundary_{sample['provenance']['step_id']}",
        "source": "pseudo",
    }

    return result
def _clean_action(action):
    if not action:
        return ""

    return (
        action.replace("(", " ")
        .replace(")", "")
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
    )


def _what_from_history(history, task):
    if not history:
        return f"The task has started with the goal to {task}."

    previous = _clean_action(history[-1])
    return f"The previous semantic action has completed: {previous}."


def _why_from_phase(phase, action, task):
    action_text = _clean_action(action)

    templates = {
        "acquire":
            "Acquire control of the object or tool needed for the current task phase.",

        "access":
            "Gain access to the relevant object, container, or manipulation region.",

        "extract":
            "Retrieve the required object from its current location or container.",

        "transfer":
            "Transfer the relevant object or material toward its intended target.",

        "manipulate":
            "Change the state or configuration of the relevant object to advance the task.",

        "restore":
            "Restore or secure the manipulated object or environment in its intended final state.",
    }

    return templates.get(
        phase,
        f"Advance the current task phase associated with {action_text}.",
    )


def label_boundary_v1(sample):
    history = sample["history"]["semantic_actions"]
    future = sample["future_actions"]
    task = sample["language"]["task"]

    if not future:
        raise ValueError("Boundary sample has no future actions.")

    current_action = future[0]

    phase = classify_phase(
        action=current_action,
        task=task,
        history=history,
        future=future,
    )

    if phase == "unknown":
        raise ValueError(
            f"Unknown semantic phase for action: {current_action}"
        )

    result = dict(sample)

    result["intention"] = {
        "what": _what_from_history(history, task),
        "why": _why_from_phase(
            phase,
            current_action,
            task,
        ),
        "next": _clean_action(current_action).capitalize() + ".",
        "phase": phase,
        "source": "pseudo",
    }

    return result
import re


def parse_action_components(action):
    action = (action or "").strip()

    match = re.match(r"\s*([^(]+)\((.*)\)\s*$", action)
    if not match:
        return action.lower(), []

    verb = match.group(1).strip().lower()

    args = [
        x.strip()
        for x in match.group(2).split(",")
        if x.strip()
    ]

    return verb, args


def _arg(args, index, default="relevant object"):
    if index < len(args):
        return args[index]
    return default


def _why_v12(phase, action, task):
    verb, args = parse_action_components(action)

    obj = _arg(args, 0)
    target = _arg(args, 1, "target location")

    if phase == "acquire":
        return (
            f"Establish control of the {obj} so it can be used "
            f"in the current task phase."
        )

    if phase == "access":
        return (
            f"Make the {obj} accessible so the subsequent "
            f"task operation can be performed."
        )

    if phase == "extract":
        if len(args) >= 2:
            return (
                f"Retrieve the {obj} from the {target} so it is "
                f"available for the next task phase."
            )

        return (
            f"Retrieve the {obj} from its current location so it "
            f"is available for subsequent manipulation."
        )

    if phase == "transfer":
        if len(args) >= 2:
            return (
                f"Relocate the {obj} toward the {target} to establish "
                f"the required object-target relation."
            )

        return (
            f"Relocate the {obj} to the location required by "
            f"the current task."
        )

    if phase == "manipulate":
        return (
            f"Change the state or configuration of the {obj} "
            f"to establish the condition required for progress."
        )

    if phase == "activate":
        return (
            f"Operate the {obj} to establish the functional state "
            f"required by the current task."
        )

    if phase == "restore":
        return (
            f"Secure or restore the {obj} to the state required "
            f"after the preceding manipulation."
        )

    if phase == "navigate":
        return (
            f"Reduce the spatial separation from the {obj} so the "
            f"next interaction can be performed."
        )

    raise ValueError(
        f"No V1.2 WHY template for phase={phase}, action={action}"
    )


def label_boundary_v12(sample):
    history = sample["history"]["semantic_actions"]
    future = sample["future_actions"]
    task = sample["language"]["task"]

    if not future:
        raise ValueError("Boundary sample has no future actions.")

    current_action = future[0]

    phase = classify_phase(
        action=current_action,
        task=task,
        history=history,
        future=future,
    )

    if phase == "unknown":
        raise ValueError(
            f"Unknown semantic phase for action: {current_action}"
        )

    result = dict(sample)

    result["intention"] = {
        "what": _what_from_history(history, task),
        "why": _why_v12(
            phase=phase,
            action=current_action,
            task=task,
        ),
        "next": _clean_action(current_action).capitalize() + ".",
        "phase": phase,
        "source": "pseudo",
    }

    return result