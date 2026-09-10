import json
from collections import Counter

from datasets.intention_alignment_dataset import (
    to_intention_alignment_sample,
)


INPUT = (
    "/media/dhqg/d1/datasets/ego4d/pilot/"
    "hf_ego4d_intention_v12_100.jsonl"
)


samples = []
phase_counts = Counter()

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        raw = json.loads(line)
        sample = to_intention_alignment_sample(raw)

        samples.append(sample)
        phase_counts[sample.phase] += 1


assert len(samples) == 245

for sample in samples:
    assert sample.why.strip()
    assert sample.phase.strip()

    # Stage-A contract itself must expose only these fields.
    assert set(sample.__dict__) == {
        "sample_id",
        "task",
        "history",
        "observation",
        "why",
        "phase",
    }


print("=== STAGE-A REAL-METADATA PILOT ===")
print("samples:", len(samples))
print("phases:", len(phase_counts))

for phase, count in phase_counts.most_common():
    print(f"{phase:12s}: {count}")

print()
print("FIRST SAMPLE")
print(samples[0])

print()
print("STAGE-A PILOT LOADER: PASS")