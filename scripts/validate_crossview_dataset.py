from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_FIELDS = {
    "sample_id",
    "source_dataset",
    "task",
    "phase",
    "why",
    "view_type",
    "view_id",
    "subject_id",
    "scene_id",
    "video_uid",
    "video_path",
    "start_time",
    "end_time",
    "pair_group_id",
    "split",
}


VALID_VIEWS = {
    "ego",
    "exo",
    "robot",
}


VALID_SPLITS = {
    "train",
    "val",
    "test",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest",
        type=Path,
    )

    return parser.parse_args()


def load_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    if path.suffix == ".json":
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            if "samples" in data:
                data = data["samples"]
            else:
                raise ValueError(
                    "JSON object must contain "
                    "'samples'."
                )

        if not isinstance(data, list):
            raise ValueError(
                "Manifest must contain a list "
                "of samples."
            )

        return data

    if path.suffix == ".jsonl":
        samples = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    samples.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSONL at line "
                        f"{line_number}: {exc}"
                    ) from exc

        return samples

    raise ValueError(
        "Manifest must be .json or .jsonl"
    )


def main():
    args = parse_args()

    samples = load_manifest(
        args.manifest
    )

    print("=" * 72)
    print("CROSS-VIEW DATASET VALIDATION")
    print("=" * 72)

    print(
        "samples:",
        len(samples),
    )

    errors = []

    sample_ids = set()

    view_counter = Counter()
    split_counter = Counter()
    source_counter = Counter()

    groups = defaultdict(list)

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        missing = (
            REQUIRED_FIELDS
            - set(sample)
        )

        if missing:
            errors.append(
                (
                    index,
                    "missing fields",
                    sorted(missing),
                )
            )

            continue

        sample_id = sample[
            "sample_id"
        ]

        if sample_id in sample_ids:
            errors.append(
                (
                    index,
                    "duplicate sample_id",
                    sample_id,
                )
            )

        sample_ids.add(
            sample_id
        )

        view_type = sample[
            "view_type"
        ]

        if view_type not in VALID_VIEWS:
            errors.append(
                (
                    index,
                    "invalid view_type",
                    view_type,
                )
            )

        split = sample[
            "split"
        ]

        if split not in VALID_SPLITS:
            errors.append(
                (
                    index,
                    "invalid split",
                    split,
                )
            )

        if not str(
            sample["why"]
        ).strip():
            errors.append(
                (
                    index,
                    "empty why",
                    sample_id,
                )
            )

        if not str(
            sample["pair_group_id"]
        ).strip():
            errors.append(
                (
                    index,
                    "empty pair_group_id",
                    sample_id,
                )
            )

        view_counter[
            view_type
        ] += 1

        split_counter[
            split
        ] += 1

        source_counter[
            sample["source_dataset"]
        ] += 1

        groups[
            sample["pair_group_id"]
        ].append(
            sample
        )

    multi_view_groups = 0
    ego_exo_groups = 0

    for group_samples in groups.values():
        views = {
            x["view_type"]
            for x in group_samples
        }

        if len(views) >= 2:
            multi_view_groups += 1

        if (
            "ego" in views
            and "exo" in views
        ):
            ego_exo_groups += 1

    print()
    print(
        "views:",
        dict(view_counter),
    )

    print(
        "splits:",
        dict(split_counter),
    )

    print(
        "sources:",
        dict(source_counter),
    )

    print()

    print(
        "pair groups:",
        len(groups),
    )

    print(
        "multi-view groups:",
        multi_view_groups,
    )

    print(
        "ego-exo groups:",
        ego_exo_groups,
    )

    print()

    print(
        "errors:",
        len(errors),
    )

    if errors:
        for error in errors[:20]:
            print(
                "ERROR:",
                error,
            )

        raise SystemExit(1)

    print(
        "CROSS-VIEW DATASET VALIDATION: PASS"
    )


if __name__ == "__main__":
    main()