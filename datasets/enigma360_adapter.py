from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_ENIGMA_ROOT = Path("/media/dhqg/d1/datasets/enigma360")


@dataclass
class Enigma360Sample:
    sample_id: str
    source_dataset: str
    task: str
    phase: str
    what: str
    why: str | None
    next: str | None

    view_type: str
    view_id: str

    subject_id: str | None
    scene_id: str | None

    video_uid: str
    video_path: str

    start_time: float
    end_time: float

    pair_group_id: str
    split: str

    action_id: str
    action_description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_split_ids(
    root: Path,
    split: str,
) -> list[str]:
    split_path = root / "annotations" / f"{split}.txt"

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return ids


def _load_raw_annotation(
    root: Path,
    video_uid: str,
) -> dict[str, Any]:
    path = root / "annotations" / "raw_json" / f"{video_uid}.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing ENIGMA annotation: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _video_path(
    root: Path,
    video_uid: str,
    view_type: str,
) -> Path:
    if view_type == "ego":
        suffix = "first_person"
    elif view_type == "exo":
        suffix = "third_person"
    else:
        raise ValueError(f"Unsupported view_type: {view_type}")

    return (
        root
        / "data"
        / "videos"
        / video_uid
        / f"{video_uid}_{suffix}.mp4"
    )


def _normalize_action_id(action_id: Any) -> str:
    action_id = str(action_id)

    if action_id.isdigit():
        return action_id.zfill(3)

    return action_id


def build_enigma360_samples(
    root: str | Path = DEFAULT_ENIGMA_ROOT,
    splits: tuple[str, ...] = ("train", "val", "test"),
    views: tuple[str, ...] = ("ego", "exo"),
    require_video: bool = False,
) -> list[Enigma360Sample]:
    """
    Build synchronized cross-view metadata from ENIGMA-360 annotations.

    ENIGMA-360 provides procedural action segments for the egocentric sequence.
    The dataset downloader uses the same video ID for the first-person and
    third-person recordings. For the cross-view pilot, the same annotated time
    interval is assigned to both views.

    Important:
        - `what` is the current action description.
        - `phase` is the action ID.
        - `next` is the next annotated action description when available.
        - `why` is intentionally left unset because ENIGMA-360 does not provide
          the same procedural-intention supervision used by EgoIntent.
        - ego/exo samples from the same temporal segment share `pair_group_id`.
        - `next` is metadata/supervision only and must never be used as predictor
          input.
    """

    root = Path(root)
    samples: list[Enigma360Sample] = []
    skipped_invalid_duration = 0
    for split in splits:
        video_ids = _read_split_ids(root, split)

        for video_uid in video_ids:
            annotation = _load_raw_annotation(root, video_uid)

            clips = annotation.get("clips", [])
            if not isinstance(clips, list):
                raise ValueError(
                    f"Expected list of clips for video {video_uid}, "
                    f"got {type(clips)}"
                )

            for clip_index, clip in enumerate(clips):
                action_time = clip.get("action_time")
                action_label = clip.get("action_label", {})

                if not action_time or len(action_time) != 2:
                    raise ValueError(
                        f"Invalid action_time in video {video_uid}, "
                        f"clip {clip_index}: {action_time}"
                    )

                start_time = float(action_time[0])
                end_time = float(action_time[1])

                if end_time <= start_time:
                    continue

                action_id = _normalize_action_id(
                    action_label.get("id_action", "")
                )
                action_description = str(
                    action_label.get("description", "")
                ).strip()

                if not action_description:
                    raise ValueError(
                        f"Missing action description in video {video_uid}, "
                        f"clip {clip_index}"
                    )

                next_description: str | None = None

                if clip_index + 1 < len(clips):
                    next_label = clips[clip_index + 1].get(
                        "action_label",
                        {},
                    )
                    candidate = str(
                        next_label.get("description", "")
                    ).strip()

                    if candidate:
                        next_description = candidate

                pair_group_id = (
                    f"enigma_{video_uid}_{clip_index:04d}"
                )

                for view_type in views:
                    path = _video_path(
                        root=root,
                        video_uid=video_uid,
                        view_type=view_type,
                    )

                    if require_video and not path.exists():
                        raise FileNotFoundError(
                            f"Missing {view_type} video: {path}"
                        )

                    sample_id = (
                        f"{pair_group_id}_{view_type}"
                    )

                    samples.append(
                        Enigma360Sample(
                            sample_id=sample_id,
                            source_dataset="enigma360",
                            task="enigma360_procedural_task",
                            phase=action_id,
                            what=action_description,
                            why=None,
                            next=next_description,
                            view_type=view_type,
                            view_id=view_type,
                            subject_id=None,
                            scene_id=None,
                            video_uid=video_uid,
                            video_path=str(path),
                            start_time=start_time,
                            end_time=end_time,
                            pair_group_id=pair_group_id,
                            split=split,
                            action_id=action_id,
                            action_description=action_description,
                        )
                    )

    return samples


def group_cross_view_pairs(
    samples: list[Enigma360Sample],
) -> dict[str, dict[str, Enigma360Sample]]:
    groups: dict[str, dict[str, Enigma360Sample]] = {}

    for sample in samples:
        group = groups.setdefault(
            sample.pair_group_id,
            {},
        )

        if sample.view_type in group:
            raise ValueError(
                f"Duplicate view '{sample.view_type}' for "
                f"{sample.pair_group_id}"
            )

        group[sample.view_type] = sample

    return groups


def audit_cross_view_pairs(
    samples: list[Enigma360Sample],
) -> dict[str, int]:
    groups = group_cross_view_pairs(samples)

    complete = 0
    incomplete = 0

    for group in groups.values():
        if "ego" in group and "exo" in group:
            complete += 1
        else:
            incomplete += 1

    return {
        "samples": len(samples),
        "pair_groups": len(groups),
        "complete_pairs": complete,
        "incomplete_pairs": incomplete,
    }


if __name__ == "__main__":
    samples = build_enigma360_samples(
        root=DEFAULT_ENIGMA_ROOT,
        splits=("train", "val", "test"),
        views=("ego", "exo"),
        require_video=False,
    )

    audit = audit_cross_view_pairs(samples)

    print("ENIGMA-360 cross-view metadata audit")
    print("=" * 60)

    for key, value in audit.items():
        print(f"{key}: {value}")

    print()
    print("First sample:")
    print(samples[0].to_dict())

    print()
    print("First pair:")

    first_group_id = samples[0].pair_group_id
    groups = group_cross_view_pairs(samples)

    for view, sample in groups[first_group_id].items():
        print(view, sample.to_dict())