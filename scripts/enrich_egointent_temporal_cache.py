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


from egointent_adapter import (
    load_egointent_split,
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

DEFAULT_CACHE_ROOT = Path(
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
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def build_sample_index(
    root: Path,
    split_config: Path,
    split: str,
):
    samples = load_egointent_split(
        root=root,
        split_config=split_config,
        split=split,
        require_videos=False,
    )

    index = {}

    for sample in samples:
        sample_id = str(
            sample.sample_id
        )

        if sample_id in index:
            raise ValueError(
                "Duplicate sample_id: "
                f"{sample_id}"
            )

        index[
            sample_id
        ] = sample

    return index


def validate_text(
    value,
    name: str,
    sample_id: str,
):
    value = str(
        value
    ).strip()

    if not value:
        raise ValueError(
            f"Empty {name} for "
            f"{sample_id}"
        )

    return value


def main():
    args = parse_args()

    cache_dir = (
        args.cache_root
        / args.split
    )

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found: "
            f"{cache_dir}"
        )

    cache_paths = sorted(
        cache_dir.glob(
            "*.pt"
        )
    )

    if not cache_paths:
        raise RuntimeError(
            f"No .pt cache files found in "
            f"{cache_dir}"
        )

    sample_index = build_sample_index(
        root=args.root,
        split_config=(
            args.split_config
        ),
        split=args.split,
    )

    print("=" * 72)
    print(
        "ENRICH TEMPORAL CACHE"
    )
    print("=" * 72)

    print(
        "split:",
        args.split,
    )

    print(
        "cache files:",
        len(cache_paths),
    )

    print(
        "alignment samples:",
        len(sample_index),
    )

    print(
        "dry run:",
        args.dry_run,
    )

    print()

    updated = 0
    missing = []
    mismatched_why = []

    for path in cache_paths:
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        sample_id = str(
            record[
                "sample_id"
            ]
        )

        sample = sample_index.get(
            sample_id
        )

        if sample is None:
            missing.append(
                sample_id
            )
            continue

        what = validate_text(
            sample.what,
            "what",
            sample_id,
        )

        why = validate_text(
            sample.why,
            "why",
            sample_id,
        )

        next_text = validate_text(
            sample.next,
            "next",
            sample_id,
        )

        cached_why = str(
            record.get(
                "why",
                "",
            )
        ).strip()

        if (
            cached_why
            and cached_why != why
        ):
            mismatched_why.append(
                {
                    "sample_id":
                        sample_id,

                    "cached":
                        cached_why,

                    "adapter":
                        why,
                }
            )

            continue

        record[
            "what"
        ] = what

        record[
            "why"
        ] = why

        record[
            "next"
        ] = next_text

        record[
            "structured_supervision"
        ] = {
            "what":
                what,

            "why":
                why,

            "next":
                next_text,
        }

        if not args.dry_run:
            torch.save(
                record,
                path,
            )

        updated += 1

    print(
        "updated:",
        updated,
    )

    print(
        "missing sample ids:",
        len(missing),
    )

    print(
        "WHY mismatches:",
        len(mismatched_why),
    )

    if missing:
        print()
        print(
            "First missing samples:"
        )

        for sample_id in missing[
            :10
        ]:
            print(
                "  ",
                sample_id,
            )

    if mismatched_why:
        print()
        print(
            "First WHY mismatches:"
        )

        for item in mismatched_why[
            :10
        ]:
            print(
                item
            )

    if (
        missing
        or mismatched_why
    ):
        raise RuntimeError(
            "Cache enrichment validation failed."
        )

    expected = len(
        sample_index
    )

    if updated != expected:
        raise RuntimeError(
            "Unexpected number of updated "
            f"samples: {updated} "
            f"vs expected {expected}"
        )

    print()
    print(
        "PASS"
        if not args.dry_run
        else "DRY-RUN PASS"
    )


if __name__ == "__main__":
    main()