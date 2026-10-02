from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def maybe_download(dataset_root: Path) -> None:
    if dataset_root.exists() and any(dataset_root.rglob("step_label.json")):
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download. "
            "Install it in the active LeRobot environment first."
        ) from exc

    dataset_root.mkdir(parents=True, exist_ok=True)

    print("Downloading full EgoIntent repository...")
    print("Target:", dataset_root)

    snapshot_download(
        repo_id="py2279105943/EgoIntent",
        repo_type="dataset",
        local_dir=str(dataset_root),
    )


def parse_dataset(dataset_root: Path):
    rows = []

    label_files = sorted(dataset_root.rglob("step_label.json"))

    if not label_files:
        raise FileNotFoundError(
            f"No step_label.json found under {dataset_root}. "
            "Run with --download or point --dataset-root to the full EgoIntent repository."
        )

    for label_path in label_files:
        event_dir = label_path.parent
        rel = event_dir.relative_to(dataset_root)

        if len(rel.parts) < 3:
            raise RuntimeError(
                f"Unexpected EgoIntent path layout: {event_dir}"
            )

        setting, scene, event = rel.parts[:3]

        payload = json.loads(
            label_path.read_text(encoding="utf-8")
        )

        video_uid = str(payload["video_uid"])

        for step in payload["steps"]:
            step_id = int(step["step_id"])
            video_path = event_dir / f"step{step_id}.mp4"

            if not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video for annotation: {video_path}"
                )

            sample_id = f"{setting}/{scene}/{event}/step{step_id}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "setting": setting,
                    "scene": scene,
                    "event": event,
                    "video_uid": video_uid,
                    "step_id": step_id,
                    "start_time": float(step["start_time"]),
                    "obs_end_time": float(step["obs_end_time"]),
                    "what": str(step["local_intent"]),
                    "why": str(step["procedural_intent"]),
                    "next": str(step["observed_next_step"]),
                    "plausible_next_steps": list(
                        step.get("plausible_next_steps", [])
                    ),
                    "video_path": str(video_path.resolve()),
                }
            )

    return rows


def summarize_scenes(rows):
    scene_stats = {}

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["scene"]].append(row)

    for scene, items in grouped.items():
        settings = sorted({x["setting"] for x in items})

        if len(settings) != 1:
            raise RuntimeError(
                f"Scene {scene} appears in multiple settings: {settings}"
            )

        scene_stats[scene] = {
            "setting": settings[0],
            "num_samples": len(items),
            "num_events": len({x["event"] for x in items}),
            "num_videos": len({x["video_uid"] for x in items}),
            "events": sorted({x["event"] for x in items}),
        }

    return scene_stats


def choose_scene_split(scene_stats):
    """
    Fixed protocol:
      - 10 train scenes: 7 indoor + 3 outdoor
      - 2 validation scenes: 1 indoor + 1 outdoor
      - 3 test scenes: 2 indoor + 1 outdoor

    Among all scene-disjoint assignments that satisfy those setting counts,
    choose the assignment whose sample proportions are closest to
    10/15, 2/15, 3/15.

    This uses only scene identity, indoor/outdoor setting, and sample counts.
    It does NOT use labels, model predictions, or evaluation scores.
    """
    indoor = sorted(
        s
        for s, stat in scene_stats.items()
        if stat["setting"] == "indoor"
    )

    outdoor = sorted(
        s
        for s, stat in scene_stats.items()
        if stat["setting"] == "outdoor"
    )

    if len(indoor) != 10 or len(outdoor) != 5:
        raise RuntimeError(
            "Expected EgoIntent to contain 10 indoor and 5 outdoor scenes, "
            f"but found {len(indoor)} indoor and {len(outdoor)} outdoor."
        )

    count = {
        scene: int(scene_stats[scene]["num_samples"])
        for scene in scene_stats
    }

    total = sum(count.values())

    target = {
        "train": 10 / 15,
        "val": 2 / 15,
        "test": 3 / 15,
    }

    best = None

    # val = 1 indoor + 1 outdoor
    for val_in in itertools.combinations(indoor, 1):
        for val_out in itertools.combinations(outdoor, 1):
            val = set(val_in + val_out)

            remaining_in = [s for s in indoor if s not in val]
            remaining_out = [s for s in outdoor if s not in val]

            # test = 2 indoor + 1 outdoor
            for test_in in itertools.combinations(remaining_in, 2):
                for test_out in itertools.combinations(remaining_out, 1):
                    test = set(test_in + test_out)
                    train = set(scene_stats) - val - test

                    train_indoor = sum(
                        scene_stats[s]["setting"] == "indoor"
                        for s in train
                    )
                    train_outdoor = sum(
                        scene_stats[s]["setting"] == "outdoor"
                        for s in train
                    )

                    if train_indoor != 7 or train_outdoor != 3:
                        continue

                    sizes = {
                        "train": sum(count[s] for s in train),
                        "val": sum(count[s] for s in val),
                        "test": sum(count[s] for s in test),
                    }

                    ratios = {
                        k: sizes[k] / total
                        for k in sizes
                    }

                    # Main objective: sample-ratio balance.
                    objective = sum(
                        (ratios[k] - target[k]) ** 2
                        for k in target
                    )

                    # Deterministic lexical tie-breaker.
                    tie_key = (
                        tuple(sorted(val)),
                        tuple(sorted(test)),
                    )

                    candidate = (
                        objective,
                        tie_key,
                        train,
                        val,
                        test,
                        sizes,
                        ratios,
                    )

                    if best is None or candidate[:2] < best[:2]:
                        best = candidate

    if best is None:
        raise RuntimeError("Could not construct a valid scene-disjoint split.")

    (
        objective,
        _,
        train,
        val,
        test,
        sizes,
        ratios,
    ) = best

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "sizes": sizes,
        "ratios": ratios,
        "objective": objective,
    }


def attach_splits(rows, split_spec):
    scene_to_split = {}

    for split in ["train", "val", "test"]:
        for scene in split_spec[split]:
            if scene in scene_to_split:
                raise RuntimeError(
                    f"Scene assigned twice: {scene}"
                )

            scene_to_split[scene] = split

    if len(scene_to_split) != 15:
        raise RuntimeError(
            f"Expected 15 assigned scenes, got {len(scene_to_split)}"
        )

    output = []

    for row in rows:
        item = dict(row)
        item["split"] = scene_to_split[row["scene"]]
        output.append(item)

    return output


def audit(rows):
    by_split = defaultdict(list)

    for row in rows:
        by_split[row["split"]].append(row)

    audit_out = {}

    for split in ["train", "val", "test"]:
        items = by_split[split]

        audit_out[split] = {
            "num_samples": len(items),
            "num_scenes": len({x["scene"] for x in items}),
            "num_events": len({x["event"] for x in items}),
            "num_videos": len({x["video_uid"] for x in items}),
            "scenes": sorted({x["scene"] for x in items}),
            "events": sorted({x["event"] for x in items}),
            "settings": dict(
                Counter(x["setting"] for x in items)
            ),
        }

    scenes = {
        split: {x["scene"] for x in by_split[split]}
        for split in by_split
    }

    videos = {
        split: {x["video_uid"] for x in by_split[split]}
        for split in by_split
    }

    audit_out["overlap"] = {
        "scene_train_val": sorted(
            scenes["train"] & scenes["val"]
        ),
        "scene_train_test": sorted(
            scenes["train"] & scenes["test"]
        ),
        "scene_val_test": sorted(
            scenes["val"] & scenes["test"]
        ),
        "video_train_val": sorted(
            videos["train"] & videos["val"]
        ),
        "video_train_test": sorted(
            videos["train"] & videos["test"]
        ),
        "video_val_test": sorted(
            videos["val"] & videos["test"]
        ),
    }

    overlaps = audit_out["overlap"]

    if any(overlaps.values()):
        raise RuntimeError(
            f"Leakage detected across splits: {overlaps}"
        )

    return audit_out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fixed scene-disjoint full EgoIntent protocol "
            "for final intention-understanding experiments."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent_full/EgoIntent"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/media/dhqg/d1/datasets/"
            "egointent_full/protocol_v1"
        ),
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download the complete Hugging Face EgoIntent repository "
            "if it is not already present."
        ),
    )

    args = parser.parse_args()

    if args.download:
        maybe_download(args.dataset_root)

    rows = parse_dataset(
        args.dataset_root
    )

    scene_stats = summarize_scenes(
        rows
    )

    split_spec = choose_scene_split(
        scene_stats
    )

    rows = attach_splits(
        rows,
        split_spec,
    )

    audit_out = audit(
        rows
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        args.output_dir
        / "manifest.jsonl"
    )

    with manifest_path.open(
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

    summary = {
        "protocol_name": "egointent_scene_disjoint_v1",
        "dataset_root": str(
            args.dataset_root.resolve()
        ),
        "num_samples": len(rows),
        "num_scenes": len(scene_stats),
        "num_events": len(
            {x["event"] for x in rows}
        ),
        "num_videos": len(
            {x["video_uid"] for x in rows}
        ),
        "selection_rule": {
            "train_scenes": 10,
            "val_scenes": 2,
            "test_scenes": 3,
            "train_setting_counts": {
                "indoor": 7,
                "outdoor": 3,
            },
            "val_setting_counts": {
                "indoor": 1,
                "outdoor": 1,
            },
            "test_setting_counts": {
                "indoor": 2,
                "outdoor": 1,
            },
            "objective": (
                "Among assignments satisfying the fixed scene/setting counts, "
                "minimize squared deviation of sample proportions from "
                "10/15 train, 2/15 val, 3/15 test. "
                "No semantic labels or model scores are used."
            ),
        },
        "scene_stats": scene_stats,
        "split_spec": split_spec,
        "audit": audit_out,
    }

    summary_path = (
        args.output_dir
        / "split_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    locked_test_path = (
        args.output_dir
        / "LOCKED_TEST_SCENES.txt"
    )

    locked_test_path.write_text(
        "\n".join(split_spec["test"])
        + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("EGOINTERNT FULL FINAL PROTOCOL")
    print("=" * 88)
    print("dataset_root:", args.dataset_root)
    print("total samples:", len(rows))
    print("scenes:", len(scene_stats))
    print("events:", len({x['event'] for x in rows}))
    print("source videos:", len({x['video_uid'] for x in rows}))

    print()
    for split in ["train", "val", "test"]:
        info = audit_out[split]
        print(
            f"{split.upper():5s} | "
            f"samples={info['num_samples']:4d} | "
            f"scenes={info['num_scenes']} | "
            f"events={info['num_events']:2d} | "
            f"videos={info['num_videos']:2d} | "
            f"scenes={info['scenes']}"
        )

    print()
    print("OVERLAP AUDIT")
    print(json.dumps(
        audit_out["overlap"],
        indent=2,
    ))

    print()
    print("Manifest:", manifest_path)
    print("Summary :", summary_path)
    print("Locked test scenes:", locked_test_path)
    print()
    print(
        "IMPORTANT: after this protocol is accepted, do not tune "
        "model choices using the test split."
    )


if __name__ == "__main__":
    main()
