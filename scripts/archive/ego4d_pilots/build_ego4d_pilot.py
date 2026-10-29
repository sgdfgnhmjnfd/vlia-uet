"""Build a VLIA bootstrap pilot from Ego4D LTA annotations."""

import argparse
import json
from pathlib import Path
from typing import Any

from datasets.ego4d_lta_adapter import (
    build_lta_windows,
    window_to_vlia_sample,
)
from datasets.schema import validate_vlia_sample


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ego4D-style annotation files may store entries under "clips".
    if isinstance(data, dict):
        if "clips" in data:
            data = data["clips"]
        elif "annotations" in data:
            data = data["annotations"]

    if not isinstance(data, list):
        raise ValueError(
            "Expected a list of annotations or a dictionary "
            "containing 'clips'/'annotations'."
        )

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--observed-window", type=int, default=8)
    parser.add_argument("--future-window", type=int, default=20)
    args = parser.parse_args()

    annotations = load_annotations(Path(args.input))

    windows = build_lta_windows(
        annotations,
        observed_window=args.observed_window,
        future_window=args.future_window,
    )

    selected = windows[: args.num_samples]

    if len(selected) < args.num_samples:
        print(
            f"WARNING: requested {args.num_samples} samples, "
            f"but only {len(selected)} windows are available."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, window in enumerate(selected):
            sample = window_to_vlia_sample(
                window,
                sample_id=f"ego4d_{args.split}_{idx:06d}",
                split=args.split,
            )

            validate_vlia_sample(
                sample,
                require_intention=False,
            )

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("EGO4D PILOT BUILD: PASS")
    print(f"annotations: {len(annotations)}")
    print(f"windows: {len(windows)}")
    print(f"written: {len(selected)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
