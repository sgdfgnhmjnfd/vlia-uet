from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_STAGE_A_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/stage_a_v0"
)

DEFAULT_TEMPORAL_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage-a-root",
        type=Path,
        default=DEFAULT_STAGE_A_ROOT,
    )

    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=DEFAULT_TEMPORAL_ROOT,
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def resolve_stage_a_dir(
    root: Path,
    split: str,
):
    direct = root / split

    if direct.exists():
        return direct

    legacy = (
        root
        / "samples"
        / split
    )

    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        f"Stage-A cache not found for {split}"
    )


def load_index(
    directory: Path,
):
    index = {}

    for path in sorted(
        directory.glob("*.pt")
    ):
        record = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        sample_id = str(
            record["sample_id"]
        )

        if sample_id in index:
            raise ValueError(
                f"Duplicate sample_id: {sample_id}"
            )

        index[sample_id] = (
            path,
            record,
        )

    return index


def main():
    args = parse_args()

    stage_a_dir = resolve_stage_a_dir(
        args.stage_a_root,
        args.split,
    )

    temporal_dir = (
        args.temporal_root
        / args.split
    )

    if not temporal_dir.exists():
        raise FileNotFoundError(
            temporal_dir
        )

    old_index = load_index(
        stage_a_dir
    )

    temporal_index = load_index(
        temporal_dir
    )

    old_ids = set(
        old_index
    )

    temporal_ids = set(
        temporal_index
    )

    print("=" * 72)
    print("SYNC STAGE-A WHY TARGET")
    print("=" * 72)

    print("split:", args.split)
    print("stage-a:", stage_a_dir)
    print("temporal:", temporal_dir)
    print("stage-a samples:", len(old_ids))
    print(
        "temporal samples:",
        len(temporal_ids),
    )

    missing_old = sorted(
        temporal_ids - old_ids
    )

    missing_temporal = sorted(
        old_ids - temporal_ids
    )

    print(
        "missing in Stage-A:",
        len(missing_old),
    )

    print(
        "missing in temporal:",
        len(missing_temporal),
    )

    if (
        missing_old
        or missing_temporal
    ):
        raise RuntimeError(
            "Sample sets do not match."
        )

    synced = 0
    text_mismatches = []

    for sample_id in sorted(
        temporal_ids
    ):
        _, old = old_index[
            sample_id
        ]

        temporal_path, temporal = (
            temporal_index[
                sample_id
            ]
        )

        old_why = str(
            old["why"]
        ).strip()

        temporal_why = str(
            temporal["why"]
        ).strip()

        if old_why != temporal_why:
            text_mismatches.append(
                sample_id
            )
            continue

        old_feature = old[
            "why_feature"
        ].detach().float().cpu()

        if old_feature.shape != (
            960,
        ):
            raise ValueError(
                f"Unexpected WHY shape "
                f"{tuple(old_feature.shape)} "
                f"for {sample_id}"
            )

        # Preserve the freshly encoded version
        # only for provenance/debugging.
        if (
            "why_feature"
            in temporal
            and
            "why_feature_reencoded"
            not in temporal
        ):
            temporal[
                "why_feature_reencoded"
            ] = (
                temporal[
                    "why_feature"
                ]
                .detach()
                .float()
                .cpu()
            )

        # Main WHY supervision is restored
        # exactly from the original Stage-A cache.
        temporal[
            "why_feature"
        ] = old_feature.clone()

        temporal[
            "why_feature_source"
        ] = (
            "stage_a_v0_exact"
        )

        if not args.dry_run:
            temp_path = (
                temporal_path
                .with_suffix(".tmp")
            )

            torch.save(
                temporal,
                temp_path,
            )

            temp_path.replace(
                temporal_path
            )

        synced += 1

    print(
        "WHY text mismatches:",
        len(text_mismatches),
    )

    print(
        "synced:",
        synced,
    )

    if text_mismatches:
        print(
            "first mismatch:",
            text_mismatches[0],
        )

        raise RuntimeError(
            "WHY text mismatch."
        )

    if synced != len(
        temporal_ids
    ):
        raise RuntimeError(
            "Incomplete synchronization."
        )

    print()
    print(
        "DRY-RUN PASS"
        if args.dry_run
        else "PASS"
    )


if __name__ == "__main__":
    main()