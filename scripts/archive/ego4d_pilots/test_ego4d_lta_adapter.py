"""Smoke test for Ego4D LTA bootstrap conversion."""

from datasets.ego4d_lta_adapter import (
    build_lta_windows,
    window_to_vlia_sample,
)
from datasets.schema import validate_vlia_sample


def main() -> None:
    actions = []

    for idx in range(12):
        actions.append(
            {
                "clip_uid": "clip_test_001",
                "action_idx": idx,
                "verb": f"verb{idx}",
                "noun": f"noun{idx}",
            }
        )

    windows = build_lta_windows(
        actions,
        observed_window=8,
        future_window=3,
    )

    assert len(windows) == 4

    first = windows[0]

    assert len(first["observed_actions"]) == 8
    assert first["observed_actions"][0] == "verb0 noun0"
    assert first["observed_actions"][-1] == "verb7 noun7"

    assert first["future_actions"] == [
        "verb8 noun8",
        "verb9 noun9",
        "verb10 noun10",
    ]

    sample = window_to_vlia_sample(
        first,
        sample_id="ego4d_test_0001",
        split="train",
    )

    validate_vlia_sample(
        sample,
        require_intention=False,
    )

    assert sample["intention"]["why"] == ""
    assert sample["provenance"]["trajectory_id"] == "clip_test_001"

    print("EGO4D LTA ADAPTER: PASS")
    print("observed:", len(sample["history"]["semantic_actions"]))
    print("future:", len(sample["future_actions"]))


if __name__ == "__main__":
    main()
