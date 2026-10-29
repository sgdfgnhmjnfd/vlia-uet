from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from datasets.egointent_adapter import build_egointent_alignment_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/data/egointent_stage_a_pilot.yaml"),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(
    dataset_root: Path,
    entries: list[dict],
):
    samples = []
    video_uids = set()

    for entry in entries:
        event_dir = dataset_root / entry["path"]
        label_path = event_dir / "step_label.json"

        assert event_dir.is_dir(), f"Missing event directory: {event_dir}"
        assert label_path.is_file(), f"Missing label file: {label_path}"

        event_samples = build_egointent_alignment_samples(label_path)

        expected_uid = entry["video_uid"]
        expected_steps = int(entry["num_steps"])

        assert len(event_samples) == expected_steps, (
            f"{entry['path']}: expected {expected_steps} samples, "
            f"got {len(event_samples)}"
        )

        # Verify the source label itself belongs to the expected video.
        import json

        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        actual_uid = labels["video_uid"]

        assert actual_uid == expected_uid, (
            f"{entry['path']}: expected video_uid={expected_uid}, "
            f"got {actual_uid}"
        )

        assert actual_uid not in video_uids, (
            f"Duplicate video_uid inside split: {actual_uid}"
        )
        video_uids.add(actual_uid)

        for sample in event_samples:
            assert sample.why.strip(), (
                f"{sample.sample_id}: empty WHY supervision"
            )

            video_path = sample.observation.get("video_path")
            assert video_path, (
                f"{sample.sample_id}: missing observation.video_path"
            )
            assert Path(video_path).is_file(), (
                f"{sample.sample_id}: video does not exist: {video_path}"
            )

            # Stage-A predictor contract:
            # future semantic labels must never appear in predictor input.
            observation_keys = set(sample.observation.keys())

            forbidden = {
                "observed_next_step",
                "plausible_next_steps",
                "future_actions",
                "next",
            }

            leaked = observation_keys & forbidden
            assert not leaked, (
                f"{sample.sample_id}: future leakage in observation: {leaked}"
            )

        samples.extend(event_samples)

        print(
            f"[OK] {entry['path']}: "
            f"{len(event_samples)} samples, uid={actual_uid}"
        )

    return samples, video_uids


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_root = Path(cfg["dataset"]["root"])

    assert dataset_root.is_dir(), (
        f"Dataset root does not exist: {dataset_root}"
    )

    train_samples, train_uids = load_split(
        dataset_root,
        cfg["split"]["train"],
    )

    val_samples, val_uids = load_split(
        dataset_root,
        cfg["split"]["val"],
    )

    overlap = train_uids & val_uids
    assert not overlap, (
        f"Train/val video_uid leakage detected: {sorted(overlap)}"
    )

    expected_train = int(cfg["counts"]["train"])
    expected_val = int(cfg["counts"]["val"])
    expected_total = int(cfg["counts"]["total"])

    assert len(train_samples) == expected_train, (
        f"Expected {expected_train} train samples, "
        f"got {len(train_samples)}"
    )

    assert len(val_samples) == expected_val, (
        f"Expected {expected_val} val samples, "
        f"got {len(val_samples)}"
    )

    total = len(train_samples) + len(val_samples)

    assert total == expected_total, (
        f"Expected {expected_total} total samples, got {total}"
    )

    sample_ids = [
        sample.sample_id
        for sample in train_samples + val_samples
    ]

    assert len(sample_ids) == len(set(sample_ids)), (
        "Duplicate sample_id detected"
    )

    print()
    print("=== EgoIntent Stage-A Pilot Split ===")
    print(f"Train samples : {len(train_samples)}")
    print(f"Val samples   : {len(val_samples)}")
    print(f"Total samples : {total}")
    print(f"Train videos  : {len(train_uids)}")
    print(f"Val videos    : {len(val_uids)}")
    print(f"UID overlap   : {len(overlap)}")
    print()
    print("PASS: leakage-free video_uid split validated.")


if __name__ == "__main__":
    main()