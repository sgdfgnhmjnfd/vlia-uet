from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_DATASETS_DIR = PROJECT_ROOT / "datasets"

if str(LOCAL_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DATASETS_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egointent_video import sample_video_frames
from models.intention_encoder import IntentionVisualEncoder

DEFAULT_SPLIT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent_full_split_v1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent_full/"
    "cache/temporal_intention_v1"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
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


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            rows.append(row)

    return rows


def ensure_num_frames(frames, num_frames):
    """
    Preserve the exact temporal-length handling used by the
    original EgoIntent temporal cache.
    """

    if len(frames) == 0:
        raise ValueError(
            "Video produced zero frames."
        )

    if len(frames) == num_frames:
        return frames

    indices = (
        torch.linspace(
            0,
            len(frames) - 1,
            steps=num_frames,
        )
        .round()
        .long()
        .tolist()
    )

    return [
        frames[index]
        for index in indices
    ]


def main():
    args = parse_args()

    split_path = (
        args.split_root
        / f"{args.split}.jsonl"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {split_path}"
        )

    samples = load_jsonl(split_path)

    if args.limit is not None:
        samples = samples[: args.limit]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = (
        args.output_root
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("EGOINTENT FULL TEMPORAL FEATURE CACHE")
    print("=" * 80)

    print("device:", device)
    print("split:", args.split)
    print("split_path:", split_path)
    print("samples:", len(samples))
    print("num_frames:", args.num_frames)
    print("output:", output_dir)

    print()
    print("Loading frozen SmolVLM...")

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

    print("visual feature dim:", hidden_size)
    print()

    completed = 0
    skipped = 0
    failed = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        sample_id = sample["sample_id"]
        video_path = sample["video_path"]

        # sample_id contains "/" in the full metadata.
        # Use a filesystem-safe cache filename.
        cache_id = sample_id.replace("/", "__")

        output_path = (
            output_dir
            / f"{cache_id}.pt"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            skipped += 1

            print(
                f"[{index}/{len(samples)}] "
                f"SKIP {sample_id}"
            )

            continue

        print(
            f"[{index}/{len(samples)}] "
            f"CACHE {sample_id}"
        )

        try:
            video_path_obj = Path(video_path)

            if not video_path_obj.exists():
                raise FileNotFoundError(
                    video_path_obj
                )

            # Keep the exact frame sampling used by the
            # original 771-sample temporal cache.
            frames = sample_video_frames(
                video_path,
                num_frames=args.num_frames,
                resize=None,
            )

            original_num_frames = len(frames)

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

            if tuple(frame_features.shape) != expected_shape:
                raise ValueError(
                    "Unexpected frame feature shape: "
                    f"{tuple(frame_features.shape)}, "
                    f"expected {expected_shape}."
                )

            if not torch.isfinite(
                frame_features
            ).all():
                raise ValueError(
                    "Non-finite frame features."
                )

            # Preserve the old cache keys while adding the
            # structured full-dataset metadata.
            record = {
                "sample_id": sample_id,
                "split": args.split,

                "task": sample["event"],
                "history": [],

                "why": sample["why"],
                "what": sample["what"],
                "next": sample["next"],

                "setting": sample["setting"],
                "scene": sample["scene"],
                "event": sample["event"],
                "video_uid": sample["video_uid"],
                "step_id": sample["step_id"],

                "video_path": video_path,

                "num_frames": args.num_frames,
                "original_num_frames":
                    original_num_frames,

                "frame_features":
                    frame_features,
            }

            temp_path = (
                output_path
                .with_suffix(".tmp")
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
                sample_id,
                repr(exc),
            )

            failed.append(
                {
                    "sample_id": sample_id,
                    "error": repr(exc),
                }
            )

    print()
    print("=" * 80)
    print("CACHE SUMMARY")
    print("=" * 80)

    print("requested:", len(samples))
    print("completed:", completed)
    print("skipped:", skipped)
    print("failed:", len(failed))

    if failed:
        failure_path = (
            args.output_root
            / f"failed_{args.split}.json"
        )

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                failed,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("failure log:", failure_path)

        raise RuntimeError(
            f"{len(failed)} samples failed."
        )

    print()
    print("Cache validation: PASS")
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()