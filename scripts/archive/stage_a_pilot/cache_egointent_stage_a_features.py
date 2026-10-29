import argparse
import json
from pathlib import Path

import torch

from datasets.egointent_adapter import load_egointent_split
from datasets.egointent_video import sample_video_frames
from models.intention_encoder import (
    IntentionTextEncoder,
    IntentionVisualEncoder,
)


DEFAULT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent"
)

DEFAULT_SPLIT_CONFIG = Path(
    "config/data/egointent_pilot_split_v0.json"
)

DEFAULT_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/cache/stage_a_v0"
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
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
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
        help="Optional smoke-test limit.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def extract_video_uid(sample_id):
    prefix = "egointent_"

    if not sample_id.startswith(prefix):
        raise ValueError(
            f"Unexpected sample_id: {sample_id}"
        )

    return (
        sample_id[len(prefix):]
        .rsplit("_", 1)[0]
    )


def encode_text(
    text_encoder,
    text,
):
    feature = text_encoder(
        [text]
    )

    return (
        feature[0]
        .detach()
        .float()
        .cpu()
    )


def build_record(
    sample,
    split,
    visual_encoder,
    text_encoder,
    num_frames,
    hidden_size,
):
    frames = sample_video_frames(
        sample.observation["video_path"],
        num_frames=num_frames,
        resize=None,
    )

    visual_feature = (
        visual_encoder
        .encode_video_features(frames)[0]
        .detach()
        .float()
        .cpu()
    )

    task_feature = encode_text(
        text_encoder,
        sample.task,
    )

    # Keep this exactly consistent with
    # IntentionEncoderV0.encode_history().
    if sample.history:
        history_text = " ; ".join(
            sample.history
        )

        history_feature = encode_text(
            text_encoder,
            history_text,
        )
    else:
        history_text = ""
        history_feature = torch.zeros(
            hidden_size,
            dtype=torch.float32,
        )

    why_feature = encode_text(
        text_encoder,
        sample.why,
    )

    return {
        "sample_id": sample.sample_id,
        "video_uid": extract_video_uid(
            sample.sample_id
        ),
        "split": split,
        "task": sample.task,
        "history": list(sample.history),
        "history_text": history_text,
        "why": sample.why,
        "video_path": sample.observation[
            "video_path"
        ],
        "num_frames": num_frames,

        # Frozen SmolVLM semantic-space features.
        "visual_feature": visual_feature,
        "task_feature": task_feature,
        "history_feature": history_feature,
        "why_feature": why_feature,
    }


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("split:", args.split)

    samples = load_egointent_split(
        root=args.root,
        split_config=args.split_config,
        split=args.split,
        require_videos=True,
    )

    if args.limit is not None:
        samples = samples[:args.limit]

    print("samples:", len(samples))

    output_dir = (
        args.cache_root / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("cache:", output_dir)

    print()
    print("Loading frozen SmolVLM...")

    visual_encoder = IntentionVisualEncoder()

    text_encoder = IntentionTextEncoder(
        vlm=visual_encoder.vlm,
        processor=visual_encoder.processor,
    )

    visual_encoder = visual_encoder.to(
        device
    )

    text_encoder = text_encoder.to(
        device
    )

    visual_encoder.eval()
    text_encoder.eval()

    hidden_size = (
        visual_encoder.vlm
        .config
        .text_config
        .hidden_size
    )

    print("hidden size:", hidden_size)
    print()

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
            with torch.inference_mode():
                record = build_record(
                    sample=sample,
                    split=args.split,
                    visual_encoder=visual_encoder,
                    text_encoder=text_encoder,
                    num_frames=args.num_frames,
                    hidden_size=hidden_size,
                )

            temp_path = output_path.with_suffix(
                ".tmp"
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
                f"FAILED: "
                f"{sample.sample_id}: "
                f"{exc}"
            )

            failed.append({
                "sample_id": sample.sample_id,
                "error": str(exc),
            })

    summary = {
        "split": args.split,
        "requested_samples": len(samples),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "num_frames": args.num_frames,
        "hidden_size": hidden_size,
        "cache_dir": str(output_dir),
    }

    summary_path = (
        args.cache_root
        / f"{args.split}_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("completed:", completed)
    print("skipped:", skipped)
    print("failed:", len(failed))
    print("summary:", summary_path)

    if failed:
        raise RuntimeError(
            f"{len(failed)} samples failed."
        )

    print("PASS")


if __name__ == "__main__":
    main()