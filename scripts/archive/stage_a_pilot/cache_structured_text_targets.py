from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    "/home/dhqg/vlia-uet"
)

DATASETS_DIR = (
    PROJECT_ROOT
    / "datasets"
)

if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(DATASETS_DIR),
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from models.intention_encoder import (
    IntentionTextEncoder,
    IntentionVisualEncoder,
)


DEFAULT_CACHE_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache structured WHAT / WHY / NEXT "
            "text target embeddings for EgoIntent."
        )
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
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
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def validate_record(
    record,
    path: Path,
):
    required = [
        "sample_id",
        "what",
        "why",
        "next",
        "frame_features",
    ]

    for key in required:
        if key not in record:
            raise KeyError(
                f"Missing '{key}' in {path}"
            )

    for key in [
        "what",
        "why",
        "next",
    ]:
        value = str(
            record[key]
        ).strip()

        if not value:
            raise ValueError(
                f"Empty '{key}' in {path}"
            )

    frame_features = record[
        "frame_features"
    ]

    if not torch.is_tensor(
        frame_features
    ):
        raise TypeError(
            f"'frame_features' is not a tensor: "
            f"{path}"
        )

    if (
        frame_features.ndim != 2
        or frame_features.shape[-1] != 960
    ):
        raise ValueError(
            "Unexpected frame feature shape "
            f"{tuple(frame_features.shape)} "
            f"in {path}"
        )


def normalize_text(
    text,
):
    return str(
        text
    ).strip()


def collect_unique_texts(
    records,
):
    texts = []
    seen = set()

    for record in records:
        for key in [
            "what",
            "why",
            "next",
        ]:
            text = normalize_text(
                record[key]
            )

            if text not in seen:
                seen.add(
                    text
                )

                texts.append(
                    text
                )

    return texts


def encode_text_batches(
    text_encoder,
    texts,
    batch_size,
    device,
):
    output = {}

    total = len(
        texts
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        batch_texts = texts[
            start:end
        ]

        with torch.inference_mode():
            features = text_encoder(
                batch_texts
            )

        features = (
            features
            .detach()
            .float()
            .cpu()
        )

        if features.ndim != 2:
            raise ValueError(
                "Text encoder returned shape "
                f"{tuple(features.shape)}"
            )

        if features.shape[0] != len(
            batch_texts
        ):
            raise ValueError(
                "Text encoder batch-size mismatch: "
                f"{features.shape[0]} vs "
                f"{len(batch_texts)}"
            )

        for text, feature in zip(
            batch_texts,
            features,
        ):
            output[
                text
            ] = feature.clone()

        print(
            f"encoded: {end}/{total}"
        )

    return output


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    cache_dir = (
        args.cache_root
        / args.split
    )

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found: "
            f"{cache_dir}"
        )

    paths = sorted(
        cache_dir.glob(
            "*.pt"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No .pt files found in "
            f"{cache_dir}"
        )

    print("=" * 72)
    print(
        "STRUCTURED TEXT TARGET CACHE"
    )
    print("=" * 72)

    print(
        "split:",
        args.split,
    )

    print(
        "cache:",
        cache_dir,
    )

    print(
        "records:",
        len(paths),
    )

    print(
        "batch size:",
        args.batch_size,
    )

    print(
        "overwrite:",
        args.overwrite,
    )

    print(
        "dry run:",
        args.dry_run,
    )

    print()

    records = []

    already_complete = 0

    for path in paths:
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        validate_record(
            record,
            path,
        )

        complete = all(
            key in record
            for key in [
                "what_feature",
                "why_feature",
                "next_feature",
            ]
        )

        if (
            complete
            and not args.overwrite
        ):
            already_complete += 1

        records.append(
            record
        )

    print(
        "already complete:",
        already_complete,
    )

    texts = collect_unique_texts(
        records
    )

    print(
        "unique structured texts:",
        len(texts),
    )

    print()

    if args.dry_run:
        print(
            "DRY-RUN PASS"
        )
        return

    print(
        "Loading frozen SmolVLM..."
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    visual_encoder = (
        IntentionVisualEncoder()
    )

    text_encoder = (
        IntentionTextEncoder(
            vlm=visual_encoder.vlm,
            processor=(
                visual_encoder.processor
            ),
        )
    )

    visual_encoder = (
        visual_encoder.to(
            device
        )
    )

    text_encoder = (
        text_encoder.to(
            device
        )
    )

    visual_encoder.eval()
    text_encoder.eval()

    hidden_size = (
        visual_encoder
        .vlm
        .config
        .text_config
        .hidden_size
    )

    print(
        "hidden size:",
        hidden_size,
    )

    if hidden_size != 960:
        raise ValueError(
            "Expected SmolVLM hidden size 960, "
            f"got {hidden_size}"
        )

    print()
    print(
        "Encoding unique texts..."
    )

    feature_lookup = (
        encode_text_batches(
            text_encoder=(
                text_encoder
            ),
            texts=texts,
            batch_size=(
                args.batch_size
            ),
            device=device,
        )
    )

    print()
    print(
        "Writing cache records..."
    )

    updated = 0
    skipped = 0

    for index, (
        path,
        record,
    ) in enumerate(
        zip(
            paths,
            records,
        ),
        start=1,
    ):
        complete = all(
            key in record
            for key in [
                "what_feature",
                "why_feature",
                "next_feature",
            ]
        )

        if (
            complete
            and not args.overwrite
        ):
            skipped += 1

            print(
                f"[{index}/{len(paths)}] "
                f"SKIP "
                f"{record['sample_id']}"
            )

            continue

        what_text = normalize_text(
            record[
                "what"
            ]
        )

        why_text = normalize_text(
            record[
                "why"
            ]
        )

        next_text = normalize_text(
            record[
                "next"
            ]
        )

        record[
            "what_feature"
        ] = feature_lookup[
            what_text
        ]

        record[
            "why_feature"
        ] = feature_lookup[
            why_text
        ]

        record[
            "next_feature"
        ] = feature_lookup[
            next_text
        ]

        for key in [
            "what_feature",
            "why_feature",
            "next_feature",
        ]:
            feature = record[
                key
            ]

            if feature.shape != (
                hidden_size,
            ):
                raise ValueError(
                    f"{key} has unexpected "
                    f"shape "
                    f"{tuple(feature.shape)} "
                    f"for "
                    f"{record['sample_id']}"
                )

            if not torch.isfinite(
                feature
            ).all():
                raise ValueError(
                    f"Non-finite values in "
                    f"{key} for "
                    f"{record['sample_id']}"
                )

        temp_path = (
            path.with_suffix(
                ".tmp"
            )
        )

        torch.save(
            record,
            temp_path,
        )

        temp_path.replace(
            path
        )

        updated += 1

        print(
            f"[{index}/{len(paths)}] "
            f"OK "
            f"{record['sample_id']}"
        )

    print()
    print("=" * 72)

    print(
        "updated:",
        updated,
    )

    print(
        "skipped:",
        skipped,
    )

    print(
        "total:",
        len(paths),
    )

    print(
        "PASS"
    )


if __name__ == "__main__":
    main()