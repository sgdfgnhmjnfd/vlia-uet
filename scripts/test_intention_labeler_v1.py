from datasets.hf_ego4d_adapter import expand_to_boundary_samples
from datasets.intention_labeler import label_boundary_v1
from datasets.schema import validate_vlia_sample


row = {
    "id": "test",
    "video": "EGO_test.npy",
    "conversations": [
        {
            "from": "gpt",
            "value": """Task:
puts clothes in the washing machine
Plan:
collect the clothes. open the lid of the washing machine.
put the clothes in the washing machine. close the lid.
actions:
1. collect(clothes)
2. open(lid)
3. put in(clothes, washing machine)
4. close(lid)
"""
        }
    ]
}


samples = expand_to_boundary_samples(row)

expected_phases = [
    "acquire",
    "access",
    "transfer",
    "restore",
]

for sample, expected in zip(samples, expected_phases):
    labeled = label_boundary_v1(sample)

    validate_vlia_sample(
        labeled,
        require_intention=True,
    )

    assert labeled["intention"]["phase"] == expected
    assert labeled["intention"]["source"] == "pseudo"
    assert labeled["intention"]["why"]
    assert labeled["intention"]["next"]

    print("=" * 70)
    print("ID:   ", labeled["sample_id"])
    print("PHASE:", labeled["intention"]["phase"])
    print("WHAT: ", labeled["intention"]["what"])
    print("WHY:  ", labeled["intention"]["why"])
    print("NEXT: ", labeled["intention"]["next"])


print("INTENTION LABELER V1: PASS")
