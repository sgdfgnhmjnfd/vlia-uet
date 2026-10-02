from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATASETS_DIR = (
    PROJECT_ROOT
    / "datasets"
)

if str(LOCAL_DATASETS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(LOCAL_DATASETS_DIR),
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from egointent_adapter import (
    load_egointent_split,
)

from egointent_video import (
    sample_video_frames,
)

from models.intention_encoder import (
    IntentionVisualEncoder,
)

DEFAULT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent"
)

DEFAULT_SPLIT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "data"
    / "egointent_pilot_split_v0.json"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
        ],
        required=True,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_paths(
    root,
    split_config,
):
    if not root.exists():
        raise FileNotFoundError(
            f"EgoIntent root does not exist: {root}"
        )

    if not split_config.exists():
        raise FileNotFoundError(
            "Split config does not exist: "
            f"{split_config}"
        )
def ensure_num_frames(
    frames,
    num_frames,
):
    """
    Ensure a fixed temporal length.

    If a short clip contains fewer decoded frames than
    requested, repeat frames using uniformly spaced indices
    while preserving temporal order.
    """

    if len(frames) == 0:
        raise ValueError(
            "Video produced zero frames."
        )

    if len(frames) == num_frames:
        return frames

    indices = torch.linspace(
        0,
        len(frames) - 1,
        steps=num_frames,
    ).round().long().tolist()

    return [
        frames[index]
        for index in indices
    ]

def main():
    args = parse_args()

    validate_paths(
        args.root,
        args.split_config,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 72
    )

    print(
        "EGOINTENT TEMPORAL FEATURE CACHE"
    )

    print(
        "=" * 72
    )

    print(
        "device:",
        device,
    )

    print(
        "split:",
        args.split,
    )

    print(
        "num_frames:",
        args.num_frames,
    )

    print(
        "root:",
        args.root,
    )

    print(
        "split_config:",
        args.split_config,
    )

    samples = load_egointent_split(
        root=args.root,
        split_config=args.split_config,
        split=args.split,
        require_videos=True,
    )

    if args.limit is not None:
        samples = samples[
            : args.limit
        ]

    print(
        "samples:",
        len(samples),
    )

    output_dir = (
        args.output_root
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "output:",
        output_dir,
    )

    print()

    print(
        "Loading frozen SmolVLM..."
    )

    visual_encoder = (
        IntentionVisualEncoder()
        .to(device)
        .eval()
    )

    hidden_size = (
        visual_encoder
        .vlm
        .config
        .text_config
        .hidden_size
    )

    if hidden_size != 960:
        raise ValueError(
            "Expected SmolVLM hidden size 960, "
            f"got {hidden_size}."
        )

    print(
        "visual feature dim:",
        hidden_size,
    )

    completed = 0
    skipped = 0
    failed = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        output_path = (
            output_dir
            / f"{sample.sample_id}.pt"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            skipped += 1

            print(
                f"[{index}/{len(samples)}] "
                f"SKIP {sample.sample_id}"
            )

            continue

        print(
            f"[{index}/{len(samples)}] "
            f"CACHE {sample.sample_id}"
        )

        try:
            video_path = (
                sample.observation[
                    "video_path"
                ]
            )

            frames = sample_video_frames(
                video_path,
                num_frames=args.num_frames,
                resize=None,
            )

            original_num_frames = len(
                frames
            )

            frames = ensure_num_frames(
                frames,
                args.num_frames,
            )

            if original_num_frames != args.num_frames:
                print(
                    "  temporal resample:",
                    original_num_frames,
                    "->",
                    args.num_frames,
                )

            with torch.inference_mode():
                frame_features = (
                    visual_encoder
                    .encode_frame_features(
                        frames
                    )
                )

            frame_features = (
                frame_features
                .detach()
                .float()
                .cpu()
            )

            expected_shape = (
                args.num_frames,
                hidden_size,
            )

            if tuple(
                frame_features.shape
            ) != expected_shape:
                raise ValueError(
                    "Unexpected frame feature "
                    "shape: "
                    f"{tuple(frame_features.shape)}, "
                    f"expected {expected_shape}."
                )

            if not torch.isfinite(
                frame_features
            ).all():
                raise ValueError(
                    "Non-finite frame features."
                )

            record = {
                "sample_id":
                    sample.sample_id,

                "split":
                    args.split,

                "task":
                    sample.task,

                "history":
                    list(
                        sample.history
                    ),

                "why":
                    sample.why,

                "video_path":
                    video_path,

                "num_frames":
                    args.num_frames,
                "original_num_frames":
                    original_num_frames,
                "frame_features":
                    frame_features,
            }

            temp_path = (
                output_path
                .with_suffix(
                    ".tmp"
                )
            )

            torch.save(
                record,
                temp_path,
            )

            temp_path.replace(
                output_path
            )

            completed += 1

        except Exception as exc:
            print(
                "FAILED:",
                sample.sample_id,
                str(exc),
            )

            failed.append(
                {
                    "sample_id":
                        sample.sample_id,

                    "error":
                        str(exc),
                }
            )

    print()
    print(
        "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "completed:",
        completed,
    )

    print(
        "skipped:",
        skipped,
    )

    print(
        "failed:",
        len(failed),
    )

    if failed:
        print()

        for item in failed:
            print(
                item[
                    "sample_id"
                ],
                "->",
                item[
                    "error"
                ],
            )

    print()

    if not failed:
        print(
            "TEMPORAL FEATURE CACHE: PASS"
        )


if __name__ == "__main__":
    main()