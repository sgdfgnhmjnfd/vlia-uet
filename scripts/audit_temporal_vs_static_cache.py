from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


TEMPORAL_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/temporal_intention_v1"
)

STATIC_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent/"
    "cache/stage_a_v0"
)


def find_static(
    sample_id: str,
    split: str,
):
    candidates = [
        STATIC_ROOT
        / split
        / f"{sample_id}.pt",

        STATIC_ROOT
        / "samples"
        / split
        / f"{sample_id}.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_static_visual(
    record,
):
    if "visual_feature" in record:
        return record[
            "visual_feature"
        ]

    if "visual" in record:
        return record[
            "visual"
        ]

    raise KeyError(
        "No visual feature found."
    )


def audit_split(
    split: str,
):
    temporal_dir = (
        TEMPORAL_ROOT
        / split
    )

    paths = sorted(
        temporal_dir.glob("*.pt")
    )

    cosines = []
    l2_errors = []

    missing = 0

    for path in paths:
        temporal = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        sample_id = temporal[
            "sample_id"
        ]

        static_path = find_static(
            sample_id,
            split,
        )

        if static_path is None:
            missing += 1
            continue

        static = torch.load(
            static_path,
            map_location="cpu",
            weights_only=False,
        )

        frames = (
            temporal[
                "frame_features"
            ]
            .float()
        )

        temporal_mean = (
            frames.mean(
                dim=0
            )
        )

        static_visual = (
            get_static_visual(
                static
            )
            .float()
            .reshape(-1)
        )

        if temporal_mean.shape != (
            static_visual.shape
        ):
            raise ValueError(
                f"{sample_id}: "
                f"{temporal_mean.shape} "
                f"vs "
                f"{static_visual.shape}"
            )

        cosine = (
            F.cosine_similarity(
                temporal_mean.unsqueeze(0),
                static_visual.unsqueeze(0),
                dim=-1,
            )
            .item()
        )

        l2 = (
            temporal_mean
            - static_visual
        ).norm().item()

        cosines.append(
            cosine
        )

        l2_errors.append(
            l2
        )

    print("=" * 72)
    print(split.upper())
    print("=" * 72)

    print(
        "matched:",
        len(cosines),
    )

    print(
        "missing:",
        missing,
    )

    if cosines:
        print(
            "mean cosine:",
            sum(cosines)
            / len(cosines),
        )

        print(
            "min cosine:",
            min(cosines),
        )

        print(
            "max cosine:",
            max(cosines),
        )

        print(
            "mean L2:",
            sum(l2_errors)
            / len(l2_errors),
        )


def main():
    audit_split(
        "train"
    )

    audit_split(
        "val"
    )


if __name__ == "__main__":
    main()