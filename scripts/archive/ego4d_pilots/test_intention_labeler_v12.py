from datasets.intention_labeler import label_boundary_v12
from datasets.schema import validate_vlia_sample


cases = [
    (
        "puts clothes in the washing machine",
        "put in(clothes, washing machine)",
        "transfer",
    ),
    (
        "opens the bag",
        "open(bag)",
        "access",
    ),
    (
        "turns on the washing machine",
        "turn on(laundry machine)",
        "activate",
    ),
    (
        "takes bra from drawer",
        "take out of(bra, drawer)",
        "extract",
    ),
    (
        "closes the bag",
        "close(bag)",
        "restore",
    ),
    (
        "walks to the sink",
        "walk towards(sink)",
        "navigate",
    ),
]


for i, (task, action, expected_phase) in enumerate(cases):
    sample = {
        "sample_id": f"v12_test_{i}",
        "source": "ego4d",
        "split": "train",

        "observation": {
            "images": [],
            "state": None,
        },

        "language": {
            "task": task,
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

        "future_actions": [action],

        "action": {
            "low_level": None,
        },

        "provenance": {
            "dataset": "test",
            "trajectory_id": f"test_{i}",
            "step_id": 0,
        },
    }

    labeled = label_boundary_v12(sample)

    validate_vlia_sample(
        labeled,
        require_intention=True,
    )

    assert labeled["intention"]["phase"] == expected_phase

    print("=" * 70)
    print("ACTION:", action)
    print("PHASE: ", labeled["intention"]["phase"])
    print("WHY:   ", labeled["intention"]["why"])
    print("NEXT:  ", labeled["intention"]["next"])


print("INTENTION LABELER V1.2: PASS")
