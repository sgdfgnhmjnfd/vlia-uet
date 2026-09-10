"""Ego4D LTA annotation adapter for VLIA bootstrap samples."""

from collections import defaultdict
from typing import Any


def semantic_action(action: dict[str, Any]) -> str:
    verb = str(action.get("verb", "")).strip().lower()
    noun = str(action.get("noun", "")).strip().lower()

    if not verb or not noun:
        raise ValueError("Each Ego4D action must contain verb and noun")

    return f"{verb} {noun}"


def build_lta_windows(
    annotations: list[dict[str, Any]],
    *,
    observed_window: int = 8,
    future_window: int = 20,
) -> list[dict[str, Any]]:
    """Build temporal windows from Ego4D-style LTA action annotations."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for action in annotations:
        clip_uid = action.get("clip_uid")
        if not clip_uid:
            raise ValueError("Missing clip_uid")

        grouped[str(clip_uid)].append(action)

    windows = []

    for clip_uid, actions in grouped.items():
        actions.sort(key=lambda x: int(x["action_idx"]))

        min_len = observed_window + 1
        if len(actions) < min_len:
            continue

        for start in range(len(actions) - observed_window):
            observed = actions[start : start + observed_window]

            future_end = min(
                start + observed_window + future_window,
                len(actions),
            )
            future = actions[start + observed_window : future_end]

            if not future:
                continue

            windows.append(
                {
                    "clip_uid": clip_uid,
                    "start_action_idx": int(observed[0]["action_idx"]),
                    "end_observed_action_idx": int(
                        observed[-1]["action_idx"]
                    ),
                    "observed_actions": [
                        semantic_action(a) for a in observed
                    ],
                    "future_actions": [
                        semantic_action(a) for a in future
                    ],
                }
            )

    return windows


def window_to_vlia_sample(
    window: dict[str, Any],
    *,
    sample_id: str,
    split: str,
) -> dict[str, Any]:
    """Convert one Ego4D LTA temporal window to VLIA bootstrap schema."""

    return {
        "sample_id": sample_id,
        "source": "ego4d",
        "split": split,
        "observation": {
            "images": [],
            "state": None,
        },
        "language": {
            "task": "",
        },
        "history": {
            "semantic_actions": window["observed_actions"],
        },
        "intention": {
            "what": "",
            "why": "",
            "next": "",
            "phase": "",
            "source": "unknown",
        },
        "future_actions": window["future_actions"],
        "action": {
            "low_level": None,
        },
        "provenance": {
            "dataset": "Ego4D-LTA",
            "trajectory_id": window["clip_uid"],
            "step_id": window["end_observed_action_idx"],
        },
    }
