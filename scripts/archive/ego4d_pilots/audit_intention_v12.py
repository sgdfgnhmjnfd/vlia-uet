import argparse
import json
import re
from collections import Counter


def tokens(text):
    return set(
        re.findall(
            r"[a-z]+",
            (text or "").lower(),
        )
    )


def jaccard(a, b):
    a = tokens(a)
    b = tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    phase_counts = Counter()
    why_counts = Counter()

    why_next_scores = []
    why_task_scores = []

    high_next = []
    high_task = []

    n = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            task = x["language"]["task"]
            intention = x["intention"]

            why = intention["why"]
            next_text = intention["next"]
            phase = intention["phase"]

            wn = jaccard(why, next_text)
            wt = jaccard(why, task)

            why_next_scores.append(wn)
            why_task_scores.append(wt)

            phase_counts[phase] += 1
            why_counts[why] += 1

            if wn >= 0.5:
                high_next.append(
                    (x["sample_id"], wn, why, next_text)
                )

            if wt >= 0.5:
                high_task.append(
                    (x["sample_id"], wt, why, task)
                )

            n += 1

    print("=== V1.2 INTENTION AUDIT ===")
    print("samples:", n)
    print("unique WHY:", len(why_counts))

    if n:
        print(
            "unique WHY ratio:",
            f"{len(why_counts) / n:.3f}",
        )

        print(
            "mean WHY/NEXT Jaccard:",
            f"{sum(why_next_scores) / n:.3f}",
        )

        print(
            "mean WHY/TASK Jaccard:",
            f"{sum(why_task_scores) / n:.3f}",
        )

    print("\n=== PHASE DISTRIBUTION ===")
    for phase, count in phase_counts.most_common():
        print(f"{phase:12s}: {count}")

    print("\nWHY/NEXT >= 0.5:", len(high_next))
    for item in high_next[:10]:
        print(item)

    print("\nWHY/TASK >= 0.5:", len(high_task))
    for item in high_task[:10]:
        print(item)


if __name__ == "__main__":
    main()