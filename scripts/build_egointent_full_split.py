import json
from pathlib import Path
from collections import Counter, defaultdict


DATA_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent_full"
)

OUTPUT_ROOT = Path(
    "/media/dhqg/d1/datasets/egointent_full_split_v1"
)

VAL_SCENES = {
    "living_room",
    "farm",
}

TEST_SCENES = {
    "study_room",
    "garden",
}


def get_split(scene):
    if scene in TEST_SCENES:
        return "test"

    if scene in VAL_SCENES:
        return "val"

    return "train"


def main():
    rows = []

    for annotation_path in sorted(
        DATA_ROOT.rglob("step_label.json")
    ):
        event_dir = annotation_path.parent
        relative = event_dir.relative_to(DATA_ROOT)

        if len(relative.parts) != 3:
            raise RuntimeError(
                f"Unexpected event path: {event_dir}"
            )

        setting, scene, event = relative.parts

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as f:
            annotation = json.load(f)

        video_uid = str(annotation["video_uid"])
        split = get_split(scene)

        for step in annotation["steps"]:
            step_id = int(step["step_id"])

            video_path = (
                event_dir
                / f"step{step_id}.mp4"
            )

            if not video_path.exists():
                raise FileNotFoundError(video_path)

            sample_id = (
                f"{setting}/{scene}/{event}/"
                f"step{step_id}"
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "setting": setting,
                    "scene": scene,
                    "event": event,
                    "video_uid": video_uid,
                    "step_id": step_id,
                    "video_path": str(video_path),
                    "start_time": step.get(
                        "start_time"
                    ),
                    "obs_end_time": step.get(
                        "obs_end_time"
                    ),
                    "what": step["local_intent"],
                    "why": step["procedural_intent"],
                    "next": step[
                        "observed_next_step"
                    ],
                    "plausible_next_steps": step.get(
                        "plausible_next_steps",
                        [],
                    ),
                }
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_rows = defaultdict(list)

    for row in rows:
        split_rows[row["split"]].append(row)

    for split in [
        "train",
        "val",
        "test",
    ]:
        output_path = (
            OUTPUT_ROOT
            / f"{split}.jsonl"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            for row in split_rows[split]:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    all_path = OUTPUT_ROOT / "all.jsonl"

    with open(
        all_path,
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 80)
    print("EGOINTENT FULL SPLIT V1")
    print("=" * 80)

    for split in [
        "train",
        "val",
        "test",
    ]:
        x = split_rows[split]

        scenes = sorted(
            set(r["scene"] for r in x)
        )

        events = sorted(
            set(r["event"] for r in x)
        )

        uids = set(
            r["video_uid"] for r in x
        )

        settings = Counter(
            r["setting"] for r in x
        )

        print()
        print(split.upper())
        print("samples:", len(x))
        print("scenes:", len(scenes), scenes)
        print("events:", len(events))
        print("video_uid:", len(uids))
        print("settings:", dict(settings))

    train_uids = {
        r["video_uid"]
        for r in split_rows["train"]
    }

    val_uids = {
        r["video_uid"]
        for r in split_rows["val"]
    }

    test_uids = {
        r["video_uid"]
        for r in split_rows["test"]
    }

    train_scenes = {
        r["scene"]
        for r in split_rows["train"]
    }

    val_scenes = {
        r["scene"]
        for r in split_rows["val"]
    }

    test_scenes = {
        r["scene"]
        for r in split_rows["test"]
    }

    print()
    print("=" * 80)
    print("LEAKAGE CHECK")
    print("=" * 80)

    print(
        "train/val UID overlap:",
        len(train_uids & val_uids),
    )

    print(
        "train/test UID overlap:",
        len(train_uids & test_uids),
    )

    print(
        "val/test UID overlap:",
        len(val_uids & test_uids),
    )

    print(
        "train/val scene overlap:",
        len(train_scenes & val_scenes),
    )

    print(
        "train/test scene overlap:",
        len(train_scenes & test_scenes),
    )

    print(
        "val/test scene overlap:",
        len(val_scenes & test_scenes),
    )

    print()
    print("total:", len(rows))

    if len(rows) != 3014:
        raise RuntimeError(
            f"Expected 3014 samples, got {len(rows)}"
        )

    if train_uids & val_uids:
        raise RuntimeError(
            "Train/val video_uid leakage"
        )

    if train_uids & test_uids:
        raise RuntimeError(
            "Train/test video_uid leakage"
        )

    if val_uids & test_uids:
        raise RuntimeError(
            "Val/test video_uid leakage"
        )

    if train_scenes & val_scenes:
        raise RuntimeError(
            "Train/val scene leakage"
        )

    if train_scenes & test_scenes:
        raise RuntimeError(
            "Train/test scene leakage"
        )

    if val_scenes & test_scenes:
        raise RuntimeError(
            "Val/test scene leakage"
        )

    print()
    print("Split validation: PASS")
    print("Saved:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()