from datasets.intention_alignment_dataset import (
    to_intention_alignment_sample,
)


sample = {
    "sample_id": "test_b2",
    "source": "ego4d",
    "split": "train",

    "observation": {
        "images": ["frame_001.jpg"],
        "state": None,
    },

    "language": {
        "task": "put clothes in the washing machine",
    },

    "history": {
        "semantic_actions": [
            "collect(clothes)",
            "open(washing machine)",
        ],
    },

    "intention": {
        "what": "Open washing machine completed.",
        "why": (
            "Relocate the clothes toward the washing machine "
            "to establish the required object-target relation."
        ),
        "next": "Put in clothes washing machine.",
        "phase": "transfer",
        "source": "pseudo",
    },

    "future_actions": [
        "put in(clothes, washing machine)",
        "close(washing machine)",
    ],

    "action": {
        "low_level": None,
    },

    "provenance": {
        "dataset": "test",
        "trajectory_id": "test_trajectory",
        "step_id": 2,
        "plan": "THIS MUST NOT ENTER STAGE A",
    },
}


stage_a = to_intention_alignment_sample(sample)

assert stage_a.sample_id == "test_b2"
assert stage_a.task == "put clothes in the washing machine"
assert stage_a.phase == "transfer"
assert len(stage_a.history) == 2

serialized = repr(stage_a)

# Leakage checks.
assert "put in(clothes, washing machine)" not in serialized
assert "close(washing machine)" not in serialized
assert "THIS MUST NOT ENTER STAGE A" not in serialized

print(stage_a)
print()
print("STAGE-A DATASET CONTRACT: PASS")