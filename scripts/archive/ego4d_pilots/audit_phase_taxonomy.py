import json
from collections import Counter

from datasets.phase_taxonomy import classify_phase


PATH = (
    "/media/dhqg/d1/datasets/ego4d/"
    "pilot/intention_review_20.json"
)


with open(PATH, encoding="utf-8") as f:
    data = json.load(f)


counts = Counter()
unknown = []
total = 0

for traj in data:
    task = traj["task"]
    actions = traj["actions"]

    for i, action in enumerate(actions):
        history = actions[:i]
        future = actions[i:]

        phase = classify_phase(
            action=action,
            task=task,
            history=history,
            future=future,
        )

        counts[phase] += 1
        total += 1

        print(
            f"{traj['id']:>3} "
            f"b{i} | "
            f"{action:<40} "
            f"-> {phase}"
        )

        if phase == "unknown":
            unknown.append(
                (traj["id"], task, action)
            )


print("\n=== PHASE DISTRIBUTION ===")

for phase, count in counts.most_common():
    print(f"{phase:12s}: {count}")

print("\ntotal:", total)
print("unknown:", len(unknown))

if unknown:
    print("\n=== UNKNOWN ===")
    for item in unknown:
        print(item)

assert total == 56
assert not unknown

print("\nPHASE TAXONOMY V1: PASS")
