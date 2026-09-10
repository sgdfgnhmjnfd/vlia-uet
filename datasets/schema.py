"""Validation utilities for the VLIA common sample schema."""

VALID_SOURCES = {"insight", "ego4d", "robosuite", "ur3_gazebo"}
VALID_SPLITS = {"train", "val", "test"}


def validate_vlia_sample(
    sample: dict,
    *,
    require_intention: bool = True,
) -> None:
    required = {
        "sample_id",
        "source",
        "split",
        "observation",
        "language",
        "history",
        "intention",
        "future_actions",
        "action",
        "provenance",
    }

    missing = required - sample.keys()
    if missing:
        raise ValueError(f"Missing VLIA fields: {sorted(missing)}")

    if sample["source"] not in VALID_SOURCES:
        raise ValueError(f"Invalid source: {sample['source']}")

    if sample["split"] not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {sample['split']}")

    intention = sample["intention"]
    for key in ("what", "why", "next", "phase", "source"):
        if key not in intention:
            raise ValueError(f"Missing intention field: {key}")

    why = intention["why"]
    if not isinstance(why, str):
        raise ValueError("intention.why must be a string")

    if require_intention and not why.strip():
        raise ValueError(
            "intention.why must be non-empty for training-ready samples"
        )

    if not isinstance(sample["future_actions"], list):
        raise ValueError("future_actions must be a list")

    provenance = sample["provenance"]
    for key in ("dataset", "trajectory_id", "step_id"):
        if key not in provenance:
            raise ValueError(f"Missing provenance field: {key}")
