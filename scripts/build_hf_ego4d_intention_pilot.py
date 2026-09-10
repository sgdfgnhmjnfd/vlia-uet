import argparse
import json
from collections import Counter

from datasets.hf_ego4d_adapter import expand_to_boundary_samples
from datasets.intention_labeler import label_boundary_v12
from datasets.schema import validate_vlia_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-trajectories", type=int, default=100)
    args = parser.parse_args()

    trajectories = 0
    boundaries = 0

    phase_counts = Counter()
    why_counts = Counter()
    unknown = []

    with open(args.input, "r", encoding="utf-8") as src, \
         open(args.output, "w", encoding="utf-8") as dst:

        for line in src:
            row = json.loads(line)

            samples = expand_to_boundary_samples(row)

            if not samples:
                continue

            # Only count trajectories that actually produce boundaries.
            trajectories += 1

            for sample in samples:
                try:
                    labeled = label_boundary_v12(sample)
                except ValueError as e:
                    unknown.append({
                        "sample_id": sample["sample_id"],
                        "action": (
                            sample["future_actions"][0]
                            if sample["future_actions"]
                            else None
                        ),
                        "error": str(e),
                    })
                    continue

                validate_vlia_sample(
                    labeled,
                    require_intention=True,
                )

                dst.write(
                    json.dumps(labeled, ensure_ascii=False) + "\n"
                )

                phase = labeled["intention"]["phase"]
                why = labeled["intention"]["why"]

                phase_counts[phase] += 1
                why_counts[why] += 1
                boundaries += 1

            if trajectories >= args.num_trajectories:
                break

    print("HF EGO4D INTENTION PILOT")
    print("trajectories:", trajectories)
    print("written boundaries:", boundaries)

    print("\n=== PHASE DISTRIBUTION ===")
    for phase, count in phase_counts.most_common():
        print(f"{phase:12s}: {count}")

    print("\nunique WHY:", len(why_counts))
    print("unknown:", len(unknown))

    if unknown:
        print("\n=== UNKNOWN EXAMPLES ===")
        for item in unknown[:20]:
            print(item)

    print("\noutput:", args.output)


if __name__ == "__main__":
    main()
