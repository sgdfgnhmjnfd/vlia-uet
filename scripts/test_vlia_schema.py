"""Smoke tests for VLIA sample validation modes."""

from datasets.schema import validate_vlia_sample


def main() -> None:
    sample = {
        "sample_id": "bootstrap_0001",
        "source": "insight",
        "split": "train",
        "observation": {
            "images": [],
            "state": None,
        },
        "language": {
            "task": "",
        },
        "history": {
            "semantic_actions": [
                "open fridge",
                "take bottle",
            ],
        },
        "intention": {
            "what": "",
            "why": "",
            "next": "",
            "phase": "",
            "source": "unknown",
        },
        "future_actions": [
            "close fridge",
            "carry bottle",
        ],
        "action": {
            "low_level": None,
        },
        "provenance": {
            "dataset": "Ego4D",
            "trajectory_id": "synthetic_structure_test",
            "step_id": None,
        },
    }

    validate_vlia_sample(sample, require_intention=False)

    try:
        validate_vlia_sample(sample, require_intention=True)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Training-ready validation should reject empty intention.why"
        )

    print("VLIA SCHEMA MODES: PASS")


if __name__ == "__main__":
    main()
