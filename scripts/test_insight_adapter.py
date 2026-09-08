"""Smoke test for the INSIGHT-to-VLIA adapter."""

from datasets.insight_adapter import adapt_insight_sample
from datasets.schema import validate_vlia_sample


def main() -> None:
    sample = {
        "messages": [
            {
                "role": "user",
                "content": "Observe the current context and anticipate future actions.",
            }
        ],
        "solution": (
            "<think>The person is interacting with a container.</think>"
            "<intention>retrieve an object from the container</intention>"
            "<answer>open container, take object, close container</answer>"
        ),
        "observed_actions": [
            "approach container",
            "touch handle",
        ],
        "dataset": "synthetic_insight_format_test",
        "trajectory_id": "test_0001",
    }

    result = adapt_insight_sample(
        sample,
        sample_id="insight_test_0001",
        split="train",
    )

    validate_vlia_sample(result)
    assert result["source"] == "insight"
    assert result["intention"]["why"] == (
        "retrieve an object from the container"
    )
    assert result["future_actions"] == [
        "open container",
        "take object",
        "close container",
    ]
    assert result["action"]["low_level"] is None
    assert result["provenance"]["trajectory_id"] == "test_0001"

    print("INSIGHT ADAPTER V0: PASS")
    print("intention:", result["intention"]["why"])
    print("future_actions:", result["future_actions"])


if __name__ == "__main__":
    main()
