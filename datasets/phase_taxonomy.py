import re


def get_action_verb(action):
    action = (action or "").strip()

    match = re.match(r"\s*([^(]+)\(", action)
    if match:
        return match.group(1).strip().lower()

    # Support simple verb-only annotations such as "grasp".
    if re.fullmatch(r"[a-zA-Z][a-zA-Z\s-]*", action):
        return action.lower()

    return ""


def classify_phase(action, task="", history=None, future=None):
    history = history or []
    future = future or []

    verb = get_action_verb(action)
    task_lower = (task or "").lower()

    if not verb:
        return "unknown"

    # Acquire/control an object or tool.
    if verb in {
        "pick up",
        "pick up with",
        "pick up with both hands",
        "collect",
        "grasp",
        "take",
        "hold",
        "lift up",
    }:
        return "acquire"

    # Gain access to a container or manipulation region.
    if verb in {
        "open",
        "unzip",
    }:
        return "access"

    # "pull" is context-dependent:
    # - pull(zipper) while opening -> access
    # - other pull actions -> manipulation
    if verb == "pull":
        if "open" in task_lower:
            return "access"
        return "manipulate"

    # Remove/retrieve from a source.
    if verb in {
        "take out of",
        "remove",
    }:
        return "extract"

    # Relocate / place / transfer an object or material.
    if verb in {
        "put",
        "put in",
        "put into",
        "put inside",
        "put on",
        "put on with",
        "put under",
        "place",
        "place in",
        "place on",
        "transfer",
        "carry",
        "carry to",
        "drop",
        "drop onto",
        "pour",
        "pour into",
        "pour on",
        "pour out",
        "fill with",
        "add",
        "pull away",
        "change",
    }:
        return "transfer"

    # Change physical state/configuration.
    if verb in {
        "turn clockwise",
        "rotate",
        "apply",
        "pulls together",
        "shake",
        "stir",
        "mix",
        "adjust",
        "touch",
        "press on",
        "wipe",
        "rinse",
        "rinse in",
        "move",
        "raise",
        "turn around",
    }:
        return "manipulate"

    # Operate a device/control.
    if verb in {
        "turn on",
        "turn off",
        "press",
        "press the button",
        "push down",
        "connect to",
        "turn",
        "turn to",
    }:
        return "activate"

    # Restore/secure final configuration.
    if verb in {
        "close",
        "closes",
        "zip",
        "put back",
        "push",
    }:
        return "restore"

    # Move/reach toward the relevant workspace.
    if verb in {
        "reach",
        "reach for",
        "walk towards",
        "move closer to",
    }:
        return "navigate"

    return "unknown"