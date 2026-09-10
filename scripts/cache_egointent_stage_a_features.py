from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from datasets.egointent_adapter import (
    build_egointent_alignment_samples,
)
from datasets.egointent_video import (
    sample_video_frames,
)
from models.intention_encoder import (
    IntentionTextEncoder,
    IntentionVisualEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/data/egointent_stage_a_pilot.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent/cache/stage_a_v0"
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--intention-dim",
        type=int,
        default=256,
    )

    return parser.parse_args()


def load_config(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def build_split_samples(
    dataset_root: Path,
    entries: list[dict],
):
    records = []

    for entry in entries:
        event_dir = (
            dataset_root
            / entry["path"]
        )

        label_path = (
            event_dir
            / "step_label.json"
        )

        samples = (
            build_egointent_alignment_samples(
                label_path
            )
        )

        expected_steps = int(
            entry["num_steps"]
        )

        if len(samples) != expected_steps:
            raise RuntimeError(
                f"{entry['path']}: "
                f"expected {expected_steps}, "
                f"got {len(samples)}"
            )

        for sample in samples:
            records.append(
                {
                    "sample": sample,
                    "video_uid": entry[
                        "video_uid"
                    ],
                    "scene": entry[
                        "scene"
                    ],
                    "event": entry[
                        "event"
                    ],
                }
            )

    return records


def encode_sample(
    record,
    visual_encoder,
    text_encoder,
    num_frames,
):
    sample = record["sample"]

    video_path = (
        sample.observation[
            "video_path"
        ]
    )

    if video_path is None:
        raise RuntimeError(
            f"{sample.sample_id}: "
            "missing video path"
        )

    frames = sample_video_frames(
        video_path,
        num_frames=num_frames,
        resize=None,
    )

    frames = list(frames)

    with torch.inference_mode():
        visual_feature = (
            visual_encoder
            .encode_video_features(
                frames
            )
        )

        task_feature = (
            text_encoder(
                [sample.task]
            )
        )

        if sample.history:
            history_text = (
                " ; ".join(
                    sample.history
                )
            )

            history_feature = (
                text_encoder(
                    [history_text]
                )
            )

        else:
            history_feature = (
                torch.zeros_like(
                    task_feature
                )
            )

        why_feature = (
            text_encoder(
                [sample.why]
            )
        )

    return {
        "sample_id": (
            sample.sample_id
        ),
        "video_uid": (
            record["video_uid"]
        ),
        "scene": (
            record["scene"]
        ),
        "event": (
            record["event"]
        ),
        "task": (
            sample.task
        ),
        "history": (
            sample.history
        ),
        "why": (
            sample.why
        ),
        "video_path": (
            video_path
        ),
        "visual": (
            visual_feature
            .detach()
            .float()
            .cpu()
        ),
        "task_feature": (
            task_feature
            .detach()
            .float()
            .cpu()
        ),
        "history_feature": (
            history_feature
            .detach()
            .float()
            .cpu()
        ),
        "why_feature": (
            why_feature
            .detach()
            .float()
            .cpu()
        ),
    }


def cache_split(
    split_name,
    records,
    output_dir,
    visual_encoder,
    text_encoder,
    num_frames,
):
    sample_dir = (
        output_dir
        / "samples"
        / split_name
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(records)

    print()
    print(
        f"=== CACHE {split_name.upper()} ==="
    )
    print(
        f"samples: {total}"
    )

    cached_paths = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        sample = record["sample"]

        cache_path = (
            sample_dir
            / f"{sample.sample_id}.pt"
        )

        cached_paths.append(
            cache_path
        )

        if cache_path.exists():
            print(
                f"[{index:04d}/{total:04d}] "
                f"SKIP {sample.sample_id}"
            )
            continue

        print(
            f"[{index:04d}/{total:04d}] "
            f"ENCODE {sample.sample_id}"
        )

        encoded = encode_sample(
            record=record,
            visual_encoder=(
                visual_encoder
            ),
            text_encoder=(
                text_encoder
            ),
            num_frames=num_frames,
        )

        torch.save(
            encoded,
            cache_path,
        )

    return cached_paths


def combine_split(
    split_name,
    cached_paths,
    output_dir,
):
    items = []

    print()
    print(
        f"Combining "
        f"{split_name} cache..."
    )

    for path in cached_paths:
        item = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        items.append(item)

    visual = torch.cat(
        [
            item["visual"]
            for item in items
        ],
        dim=0,
    )

    task = torch.cat(
        [
            item["task_feature"]
            for item in items
        ],
        dim=0,
    )

    history = torch.cat(
        [
            item["history_feature"]
            for item in items
        ],
        dim=0,
    )

    why = torch.cat(
        [
            item["why_feature"]
            for item in items
        ],
        dim=0,
    )

    combined = {
        "sample_ids": [
            item["sample_id"]
            for item in items
        ],
        "video_uids": [
            item["video_uid"]
            for item in items
        ],
        "scenes": [
            item["scene"]
            for item in items
        ],
        "events": [
            item["event"]
            for item in items
        ],
        "tasks": [
            item["task"]
            for item in items
        ],
        "histories": [
            item["history"]
            for item in items
        ],
        "whys": [
            item["why"]
            for item in items
        ],
        "video_paths": [
            item["video_path"]
            for item in items
        ],
        "visual": visual,
        "task": task,
        "history": history,
        "why": why,
    }

    output_path = (
        output_dir
        / f"{split_name}.pt"
    )

    torch.save(
        combined,
        output_path,
    )

    print(
        f"saved: {output_path}"
    )

    print(
        f"visual  : "
        f"{tuple(visual.shape)}"
    )

    print(
        f"task    : "
        f"{tuple(task.shape)}"
    )

    print(
        f"history : "
        f"{tuple(history.shape)}"
    )

    print(
        f"why     : "
        f"{tuple(why.shape)}"
    )

    return combined


def validate_combined(
    split_name,
    data,
    expected_count,
):
    count = len(
        data["sample_ids"]
    )

    if count != expected_count:
        raise RuntimeError(
            f"{split_name}: "
            f"expected {expected_count}, "
            f"got {count}"
        )

    if (
        len(
            set(
                data["sample_ids"]
            )
        )
        != count
    ):
        raise RuntimeError(
            f"{split_name}: "
            "duplicate sample IDs"
        )

    feature_names = [
        "visual",
        "task",
        "history",
        "why",
    ]

    for name in feature_names:
        tensor = data[name]

        if tensor.ndim != 2:
            raise RuntimeError(
                f"{split_name}/{name}: "
                f"expected 2-D tensor, "
                f"got {tensor.shape}"
            )

        if tensor.shape[0] != count:
            raise RuntimeError(
                f"{split_name}/{name}: "
                "sample count mismatch"
            )

        if tensor.shape[1] != 960:
            raise RuntimeError(
                f"{split_name}/{name}: "
                f"expected hidden size 960, "
                f"got {tensor.shape[1]}"
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                f"{split_name}/{name}: "
                "contains NaN or Inf"
            )

    print(
        f"PASS: {split_name} "
        f"cache validated."
    )


def main():
    args = parse_args()

    cfg = load_config(
        args.config
    )

    dataset_root = Path(
        cfg["dataset"]["root"]
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=== EgoIntent Stage-A "
        "Frozen Feature Cache ==="
    )

    print(
        f"dataset root : "
        f"{dataset_root}"
    )

    print(
        f"output dir   : "
        f"{output_dir}"
    )

    print(
        f"device       : "
        f"{device}"
    )

    print(
        f"num frames   : "
        f"{args.num_frames}"
    )

    train_records = (
        build_split_samples(
            dataset_root,
            cfg["split"]["train"],
        )
    )

    val_records = (
        build_split_samples(
            dataset_root,
            cfg["split"]["val"],
        )
    )

    print(
        f"train records: "
        f"{len(train_records)}"
    )

    print(
        f"val records  : "
        f"{len(val_records)}"
    )

    train_uids = {
        record["video_uid"]
        for record in train_records
    }

    val_uids = {
        record["video_uid"]
        for record in val_records
    }

    overlap = (
        train_uids
        & val_uids
    )

    if overlap:
        raise RuntimeError(
            "Train/val video_uid "
            f"leakage: {overlap}"
        )

    print(
        "Loading frozen "
        "SmolVLM backbone..."
    )

    visual_encoder = (
        IntentionVisualEncoder(
            intention_dim=(
                args.intention_dim
            ),
        )
        .to(device)
    )

    text_encoder = (
        IntentionTextEncoder(
            vlm=(
                visual_encoder.vlm
            ),
            processor=(
                visual_encoder.processor
            ),
        )
        .to(device)
    )

    visual_encoder.eval()
    text_encoder.eval()

    train_paths = cache_split(
        split_name="train",
        records=train_records,
        output_dir=output_dir,
        visual_encoder=(
            visual_encoder
        ),
        text_encoder=(
            text_encoder
        ),
        num_frames=(
            args.num_frames
        ),
    )

    val_paths = cache_split(
        split_name="val",
        records=val_records,
        output_dir=output_dir,
        visual_encoder=(
            visual_encoder
        ),
        text_encoder=(
            text_encoder
        ),
        num_frames=(
            args.num_frames
        ),
    )

    train_data = combine_split(
        split_name="train",
        cached_paths=train_paths,
        output_dir=output_dir,
    )

    val_data = combine_split(
        split_name="val",
        cached_paths=val_paths,
        output_dir=output_dir,
    )

    validate_combined(
        "train",
        train_data,
        int(
            cfg["counts"]["train"]
        ),
    )

    validate_combined(
        "val",
        val_data,
        int(
            cfg["counts"]["val"]
        ),
    )

    metadata = {
        "dataset": "EgoIntent",
        "config": str(
            args.config
        ),
        "num_frames": (
            args.num_frames
        ),
        "feature_dim": 960,
        "train_samples": len(
            train_records
        ),
        "val_samples": len(
            val_records
        ),
        "train_video_uids": sorted(
            train_uids
        ),
        "val_video_uids": sorted(
            val_uids
        ),
        "train_val_uid_overlap": (
            sorted(overlap)
        ),
        "features": [
            "visual",
            "task",
            "history",
            "why",
        ],
        "note": (
            "Frozen SmolVLM "
            "features for Stage-A "
            "alignment. No 256-D "
            "intention latent is "
            "cached because the "
            "960->256 projection is "
            "not trained by the "
            "current Stage-A "
            "alignment objective."
        ),
    }

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "=== CACHE COMPLETE ==="
    )
    print(
        f"metadata: "
        f"{metadata_path}"
    )
    print(
        "Train/val UID overlap: 0"
    )
    print(
        "PASS: Stage-A frozen "
        "feature cache built."
    )


if __name__ == "__main__":
    main()