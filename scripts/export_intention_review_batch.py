import argparse
import json

from datasets.hf_ego4d_adapter import (
    parse_hf_ego4d_row,
    expand_to_boundary_samples,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-trajectories", type=int, default=20)
    args = parser.parse_args()

    trajectories = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            parsed = parse_hf_ego4d_row(row)

            if not parsed["actions"]:
                continue

            boundaries = expand_to_boundary_samples(row)

            trajectories.append({
                "id": parsed["id"],
                "video": parsed["video"],
                "task": parsed["task"],
                "plan": parsed["plan"],
                "actions": parsed["actions"],
                "boundaries": [
                    {
                        "boundary": s["provenance"]["step_id"],
                        "history": s["history"]["semantic_actions"],
                        "future": s["future_actions"],
                    }
                    for s in boundaries
                ],
            })

            if len(trajectories) >= args.num_trajectories:
                break

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            trajectories,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_boundaries = sum(
        len(x["boundaries"]) for x in trajectories
    )

    print("INTENTION REVIEW BATCH: PASS")
    print("trajectories:", len(trajectories))
    print("boundaries:", total_boundaries)
    print("output:", args.output)


if __name__ == "__main__":
    main()
