from datasets.hf_ego4d_adapter import expand_to_boundary_samples


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

assert len(samples) == 4

assert samples[0]["history"]["semantic_actions"] == []
assert len(samples[0]["future_actions"]) == 4

assert samples[1]["history"]["semantic_actions"] == [
    "collect(clothes)"
]

assert samples[1]["future_actions"][0] == "open(lid)"

assert samples[3]["history"]["semantic_actions"] == [
    "collect(clothes)",
    "open(lid)",
    "put in(clothes, washing machine)",
]

assert samples[3]["future_actions"] == [
    "close(lid)"
]

for sample in samples:
    print(
        sample["sample_id"],
        "| history =", sample["history"]["semantic_actions"],
        "| future =", sample["future_actions"],
    )

print("HF EGO4D BOUNDARY EXPANSION: PASS")
