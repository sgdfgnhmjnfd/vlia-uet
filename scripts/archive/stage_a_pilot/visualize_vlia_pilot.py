"""Inspect VLIA pilot samples and optionally play associated video."""

import argparse
import json
from pathlib import Path

import cv2


def load_jsonl(path: Path) -> list[dict]:
    samples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def print_sample(sample: dict) -> None:
    print("=" * 72)
    print(f"Sample      : {sample['sample_id']}")
    print(f"Source      : {sample['source']}")
    print(f"Split       : {sample['split']}")
    print(f"Trajectory  : {sample['provenance']['trajectory_id']}")
    print(f"Boundary    : {sample['provenance']['step_id']}")

    print("\nOBSERVED ACTIONS")
    for idx, action in enumerate(
        sample["history"]["semantic_actions"], start=1
    ):
        print(f"  O{idx:02d}: {action}")

    print("\n---------- PREDICTION BOUNDARY ----------")

    print("\nINTENTION")
    why = sample["intention"]["why"]
    print(f"  WHY: {why if why else '[UNLABELED]'}")

    print("\nFUTURE ACTIONS")
    for idx, action in enumerate(sample["future_actions"], start=1):
        print(f"  F{idx:02d}: {action}")

    print("=" * 72)


def play_video(video_path: Path) -> None:
    if not video_path.exists():
        print(f"VIDEO NOT FOUND: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"FAILED TO OPEN VIDEO: {video_path}")
        return

    print("Playing video. Press q to stop.")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        cv2.imshow("VLIA Pilot", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    samples = load_jsonl(Path(args.input))

    if not samples:
        raise ValueError("Pilot file contains no samples")

    if args.index < 0 or args.index >= len(samples):
        raise IndexError(
            f"Index {args.index} outside [0, {len(samples) - 1}]"
        )

    sample = samples[args.index]
    print_sample(sample)

    if args.video:
        play_video(Path(args.video))
    else:
        print("\nNo video path supplied; semantic inspection only.")


if __name__ == "__main__":
    main()
