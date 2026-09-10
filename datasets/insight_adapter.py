"""Adapter from INSIGHT-style cognitive reasoning samples to VLIA samples."""

import re
from typing import Any


def _extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _parse_future_actions(text: str) -> list[str]:
    answer = _extract_tag(text, "answer")
    if not answer:
        return []

    return [
        action.strip().lower()
        for action in answer.split(",")
        if action.strip()
    ]


def _extract_task(sample: dict[str, Any]) -> str:
    task = sample.get("task")
    if isinstance(task, str):
        return task.strip()

    for message in sample.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()

    return ""


def adapt_insight_sample(
    sample: dict[str, Any],
    *,
    sample_id: str,
    split: str,
) -> dict[str, Any]:
    """Convert one INSIGHT-style sample to the VLIA common schema."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    target = sample.get("gt_answer") or sample.get("solution") or ""

    intention = sample.get("gt_intention")
    if not isinstance(intention, str) or not intention.strip():
        intention = _extract_tag(target, "intention")

    future_actions = _parse_future_actions(target)

    if not intention:
        raise ValueError(
            "INSIGHT sample does not contain gt_intention or "
            "an <intention> tag in gt_answer/solution."
        )

    if not future_actions:
        raise ValueError(
            "INSIGHT sample does not contain a valid <answer> action sequence."
        )

    return {
        "sample_id": sample_id,
        "source": "insight",
        "split": split,
        "observation": {
            "images": sample.get("images", []),
            "state": None,
        },
        "language": {
            "task": _extract_task(sample),
        },
        "history": {
            "semantic_actions": sample.get("observed_actions", []),
        },
        "intention": {
            "what": sample.get("what", ""),
            "why": intention.strip(),
            "next": sample.get("next", ""),
            "phase": sample.get("phase", ""),
            "source": "ground_truth",
        },
        "future_actions": future_actions,
        "action": {
            "low_level": None,
        },
        "provenance": {
            "dataset": sample.get("dataset", "INSIGHT"),
            "trajectory_id": sample.get("trajectory_id", ""),
            "step_id": sample.get("step_id"),
        },
    }
