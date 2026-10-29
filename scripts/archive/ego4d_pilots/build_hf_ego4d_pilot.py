import argparse
import json

from datasets.hf_ego4d_adapter import to_vlia_bootstrap_sample
from datasets.schema import validate_vlia_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    written = 0
    skipped = 0

    with open(args.input, "r", encoding="utf-8") as src, \
         open(args.output, "w", encoding="utf-8") as dst:

        for line in src:
            row = json.loads(line)

            sample = to_vlia_bootstrap_sample(
                row,
                split=args.split,
            )

            # Skip malformed rows without semantic actions.
            if not sample["future_actions"]:
                skipped += 1
                continue

            validate_vlia_sample(
                sample,
                require_intention=False,
            )

            dst.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

            written += 1

            if written >= args.num_samples:
                break

    print("HF EGO4D PILOT BUILD: PASS")
    print("written:", written)
    print("skipped:", skipped)
    print("output:", args.output)


if __name__ == "__main__":
    main()
