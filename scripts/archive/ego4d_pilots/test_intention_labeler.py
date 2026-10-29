from datasets.hf_ego4d_adapter import expand_to_boundary_samples
from datasets.intention_labeler import label_boundary_v0
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
collect clothes. open lid. put clothes in machine. close lid.
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

for sample in samples:
    labeled = label_boundary_v0(sample)

    validate_vlia_sample(
        labeled,
        require_intention=True,
    )

    print("=" * 70)
    print("ID:   ", labeled["sample_id"])
    print("HIST: ", labeled["history"]["semantic_actions"])
    print("WHAT: ", labeled["intention"]["what"])
    print("WHY:  ", labeled["intention"]["why"])
    print("NEXT: ", labeled["intention"]["next"])

print("INTENTION LABELER V0: PASS")
