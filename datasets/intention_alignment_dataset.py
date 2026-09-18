from dataclasses import dataclass
from typing import Any


@dataclass
class IntentionAlignmentSample:
    sample_id: str
    task: str
    history: list[str]
    observation: dict[str, Any]

    # Structured semantic supervision.
    what: str
    why: str
    next: str

    phase: str


def to_intention_alignment_sample(sample):
    """
    Convert a labeled VLIA boundary sample into a Stage-A sample.

    Predictor-visible inputs:
        - observation
        - task
        - semantic action history

    Supervision:
        - WHAT
        - WHY
        - NEXT
        - phase (for evaluation only)

    Important:
        WHAT and NEXT are supervision targets only.
        NEXT must never become a predictor input, otherwise future
        information leakage is introduced.
    """

    intention = sample["intention"]

    what = str(
        intention.get(
            "what",
            "",
        )
    ).strip()

    why = str(
        intention.get(
            "why",
            "",
        )
    ).strip()

    next_text = str(
        intention.get(
            "next",
            "",
        )
    ).strip()

    if not why:
        raise ValueError(
            "Stage-A sample requires "
            "a non-empty WHY label."
        )

    return IntentionAlignmentSample(
        sample_id=sample[
            "sample_id"
        ],

        task=sample[
            "language"
        ][
            "task"
        ],

        history=list(
            sample[
                "history"
            ][
                "semantic_actions"
            ]
        ),

        observation=dict(
            sample[
                "observation"
            ]
        ),

        what=what,

        why=why,

        next=next_text,

        phase=intention[
            "phase"
        ],
    )