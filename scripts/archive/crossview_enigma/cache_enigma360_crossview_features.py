from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATASETS_DIR = REPO_ROOT / "datasets"

if str(LOCAL_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DATASETS_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enigma360_adapter import (
    DEFAULT_ENIGMA_ROOT,
    Enigma360Sample,
    build_enigma360_samples,
)
from models.intention_encoder import IntentionVisualEncoder


DEFAULT_OUTPUT_ROOT = Path(
    "/media/dhqg/d1/datasets/enigma360/cache/crossview_v1"
)

DEFAULT_VIDEO_IDS = ("54", "55", "56", "57", "58")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache ENIGMA-360 ego/exo temporal visual features using the "
            "same frozen visual encoder as the EgoIntent pipeline."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ENIGMA_ROOT,
        help="ENIGMA-360 dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output cache root.",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=list(DEFAULT_VIDEO_IDS),
        help="ENIGMA video IDs to cache.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of uniformly sampled frames per action segment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already cached records.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for frozen visual feature extraction.",
    )
    return parser.parse_args()


def sample_timestamps(
    start_time: float,
    end_time: float,
    num_frames: int,
) -> list[float]:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")

    if end_time <= start_time:
        raise ValueError(
            f"Invalid segment interval: {start_time} -> {end_time}"
        )

    if num_frames == 1:
        return [(start_time + end_time) / 2.0]

    duration = end_time - start_time

    return [
        start_time + duration * i / (num_frames - 1)
        for i in range(num_frames)
    ]


def load_frame_at_timestamp(
    video_path: Path,
    timestamp: float,
) -> Image.Image:
    import av

    container = av.open(str(video_path))

    try:
        stream = container.streams.video[0]

        if stream.time_base is None:
            raise RuntimeError(
                f"Video stream has no time_base: {video_path}"
            )

        target_pts = int(timestamp / float(stream.time_base))

        container.seek(
            target_pts,
            stream=stream,
            any_frame=False,
            backward=True,
        )

        best_frame = None
        best_delta = None

        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            frame_time = float(frame.pts * stream.time_base)
            delta = abs(frame_time - timestamp)

            if best_delta is None or delta < best_delta:
                best_frame = frame
                best_delta = delta

            if frame_time >= timestamp:
                break

        if best_frame is None:
            raise RuntimeError(
                f"Could not decode frame at {timestamp:.3f}s "
                f"from {video_path}"
            )

        return best_frame.to_image().convert("RGB")

    finally:
        container.close()


def load_temporal_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    num_frames: int,
) -> list[Image.Image]:
    timestamps = sample_timestamps(
        start_time=start_time,
        end_time=end_time,
        num_frames=num_frames,
    )

    return [
        load_frame_at_timestamp(
            video_path=video_path,
            timestamp=timestamp,
        )
        for timestamp in timestamps
    ]


def filter_samples(
    samples: Iterable[Enigma360Sample],
    video_ids: set[str],
) -> list[Enigma360Sample]:
    selected = [
        sample
        for sample in samples
        if sample.video_uid in video_ids
        and sample.split == "train"
    ]

    selected.sort(
        key=lambda sample: (
            int(sample.video_uid),
            sample.start_time,
            sample.view_type,
        )
    )

    return selected


def save_record(
    output_path: Path,
    sample: Enigma360Sample,
    frame_features: torch.Tensor,
    num_frames: int,
) -> None:
    frame_features = frame_features.detach().float().cpu()

    if frame_features.ndim != 2:
        raise ValueError(
            f"Expected frame_features with shape [T,D], "
            f"got {tuple(frame_features.shape)}"
        )

    if frame_features.shape[0] != num_frames:
        raise ValueError(
            f"Expected T={num_frames}, "
            f"got T={frame_features.shape[0]}"
        )

    if not torch.isfinite(frame_features).all():
        raise ValueError("frame_features contains non-finite values")

    record = sample.to_dict()
    record["num_frames"] = num_frames
    record["frame_features"] = frame_features

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(record, output_path)


def main() -> None:
    args = parse_args()

    video_ids = {
        str(video_id)
        for video_id in args.video_ids
    }

    samples = build_enigma360_samples(
        root=args.root,
        splits=("train",),
        views=("ego", "exo"),
        require_video=False,
    )

    samples = filter_samples(
        samples=samples,
        video_ids=video_ids,
    )

    if not samples:
        raise RuntimeError(
            f"No train samples found for video IDs: "
            f"{sorted(video_ids)}"
        )

    print("=" * 72)
    print("ENIGMA-360 CROSS-VIEW FEATURE CACHE")
    print("=" * 72)
    print(f"root: {args.root}")
    print(f"output: {args.output_root}")
    print(f"video_ids: {sorted(video_ids, key=int)}")
    print(f"samples: {len(samples)}")
    print(f"num_frames: {args.num_frames}")
    print()

    # The existing IntentionVisualEncoder in this repository does not accept a
    # `device` keyword argument. It initializes the frozen visual backbone using
    # the same code path as the EgoIntent feature-cache pipeline.
    encoder = IntentionVisualEncoder()

    device = torch.device(args.device)

    # Move only the visual path used by encode_frame_features().
    # Do not move the entire VLM because the language model is not needed here.
    encoder.vision_model = encoder.vision_model.to(device)
    encoder.connector = encoder.connector.to(device)
    encoder.visual_projection = encoder.visual_projection.to(device)

    encoder.eval()

    print(
        "vision device:",
        next(encoder.vision_model.parameters()).device,
    )
    print(
        "connector device:",
        next(encoder.connector.parameters()).device,
    )
    print(
        "projection device:",
        next(encoder.visual_projection.parameters()).device,
    )

    cached = 0
    skipped = 0
    failed = 0

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            video_path = Path(sample.video_path)

            if not video_path.exists():
                print(
                    f"[MISSING] {sample.sample_id}: {video_path}"
                )
                failed += 1
                continue

            output_path = (
                args.output_root
                / sample.split
                / sample.video_uid
                / f"{sample.sample_id}.pt"
            )

            if output_path.exists() and not args.overwrite:
                skipped += 1

                if index % 25 == 0 or index == len(samples):
                    print(
                        f"[{index}/{len(samples)}] "
                        f"cached={cached} "
                        f"skipped={skipped} "
                        f"failed={failed}"
                    )

                continue

            try:
                frames = load_temporal_frames(
                    video_path=video_path,
                    start_time=sample.start_time,
                    end_time=sample.end_time,
                    num_frames=args.num_frames,
                )

                frame_features = encoder.encode_frame_features(frames)

                if frame_features.ndim == 3:
                    if frame_features.shape[0] != 1:
                        raise ValueError(
                            "Unexpected batched feature shape: "
                            f"{tuple(frame_features.shape)}"
                        )
                    frame_features = frame_features[0]

                save_record(
                    output_path=output_path,
                    sample=sample,
                    frame_features=frame_features,
                    num_frames=args.num_frames,
                )

                cached += 1

            except Exception as exc:
                failed += 1
                print(
                    f"[FAILED] {sample.sample_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if index % 25 == 0 or index == len(samples):
                print(
                    f"[{index}/{len(samples)}] "
                    f"cached={cached} "
                    f"skipped={skipped} "
                    f"failed={failed}"
                )

    print()
    print("=" * 72)
    print("CACHE COMPLETE")
    print("=" * 72)
    print(f"cached: {cached}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")
    print(f"total: {len(samples)}")


if __name__ == "__main__":
    main()
