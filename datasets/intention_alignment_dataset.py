from dataclasses import dataclass
from typing import Any


@dataclass
class IntentionAlignmentSample:
    sample_id: str
    task: str
    history: list[str]
    observation: dict[str, Any]
    why: str
    phase: str


def to_intention_alignment_sample(sample):
    """
    Convert a labeled VLIA boundary sample into a Stage-A sample.

    Predictor-visible inputs:
        - observation
        - task
        - semantic action history

    Supervision:
        - WHY
        - phase (for evaluation only)

    Future actions and offline annotation metadata are intentionally
    excluded to prevent future-information leakage.
    """

    intention = sample["intention"]

    why = intention["why"].strip()
    if not why:
        raise ValueError("Stage-A sample requires a non-empty WHY label.")

    return IntentionAlignmentSample(
        sample_id=sample["sample_id"],
        task=sample["language"]["task"],
        history=list(
            sample["history"]["semantic_actions"]
        ),
        observation=dict(sample["observation"]),
        why=why,
        phase=intention["phase"],
    )