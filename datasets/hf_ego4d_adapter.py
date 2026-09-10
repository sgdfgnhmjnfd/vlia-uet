import re


def get_gpt_response(row):
    for message in row.get("conversations", []):
        if message.get("from") == "gpt":
            return message.get("value", "")
    return ""


def extract_section(text, start, ends):
    pattern = rf"{start}\s*:\s*(.*?)(?=" + "|".join(
        rf"\n\s*{end}\s*:" for end in ends
    ) + r"|$)"

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_actions(text):
    match = re.search(
        r"\n\s*actions?\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return []

    block = match.group(1)
    actions = []

    for line in block.splitlines():
        line = line.strip()

        line = re.sub(r"^\d+\s*[\.\)]\s*", "", line)

        if line:
            actions.append(line)

    return actions


def parse_hf_ego4d_row(row):
    text = get_gpt_response(row)

    task = extract_section(text, "Task", ["Plan", "Actions"])
    plan = extract_section(text, "Plan", ["Actions"])
    actions = parse_actions(text)

    return {
        "id": str(row.get("id", "")),
        "task": task,
        "plan": plan,
        "actions": actions,
        "video": row.get("video"),
    }


def to_vlia_bootstrap_sample(row, split="train"):
    parsed = parse_hf_ego4d_row(row)

    return {
        "sample_id": f"hf_ego4d_{parsed['id']}",
        "source": "ego4d",
        "split": split,

        "observation": {
            "images": [],
            "state": None,
        },

        "language": {
            "task": parsed["task"],
        },

        "history": {
            "semantic_actions": [],
        },

        "intention": {
            "what": "",
            "why": "",
            "next": "",
            "phase": "",
            "source": "unknown",
        },

        "future_actions": parsed["actions"],

        "action": {
            "low_level": None,
        },

        "provenance": {
            "dataset": "wofmanaf/ego4d-video",
            "trajectory_id": parsed["video"] or parsed["id"],
            "step_id": None,
            "video_ref": parsed["video"],
            "plan": parsed["plan"],
        },
    }
def expand_to_boundary_samples(row, split="train"):
    parsed = parse_hf_ego4d_row(row)

    actions = parsed["actions"]
    samples = []

    for boundary in range(len(actions)):
        history = actions[:boundary]
        future = actions[boundary:]

        sample = {
            "sample_id": f"hf_ego4d_{parsed['id']}_b{boundary}",
            "source": "ego4d",
            "split": split,

            "observation": {
                "images": [],
                "state": None,
            },

            "language": {
                "task": parsed["task"],
            },

            "history": {
                "semantic_actions": history,
            },

            "intention": {
                "what": "",
                "why": "",
                "next": "",
                "phase": "",
                "source": "unknown",
            },

            "future_actions": future,

            "action": {
                "low_level": None,
            },

            "provenance": {
                "dataset": "wofmanaf/ego4d-video",
                "trajectory_id": parsed["video"] or parsed["id"],
                "step_id": boundary,
                "video_ref": parsed["video"],
                "plan": parsed["plan"],
            },
        }

        samples.append(sample)

    return samples
