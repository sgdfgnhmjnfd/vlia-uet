#!/usr/bin/env python3
"""Validate an own-dataset annotation CSV against its MP4 files."""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path


SCHEMA = [
    "video_id",
    "participant_id",
    "task_id",
    "rep_id",
    "duration",
    "observation_end",
    "outcome_start",
    "object",
    "target",
    "what",
    "why",
    "next",
]
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Annotation CSV.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing <video_id>.mp4 files.",
    )
    return parser.parse_args()


def read_rows(path: Path, violations: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        violations.append(f"CSV file does not exist: {path}")
        return []
    try:
        handle = path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        violations.append(f"Cannot open CSV {path}: {exc}")
        return []
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCHEMA:
            violations.append(
                f"CSV header mismatch: expected {SCHEMA!r}, got {reader.fieldnames!r}"
            )
            return []
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                violations.append(f"line {line_number}: too many CSV fields")
                continue
            rows.append({"__line__": str(line_number), **row})
    return rows


def video_duration(path: Path) -> tuple[float | None, str | None]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None, "OpenCV cannot open the video"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
            return None, f"invalid video metadata (frames={frame_count}, fps={fps})"
        return frame_count / fps, None
    finally:
        capture.release()


def validate(csv_path: Path, video_dir: Path) -> tuple[list[str], Counter, Counter]:
    violations: list[str] = []
    rows = read_rows(csv_path, violations)
    if not video_dir.is_dir():
        violations.append(f"Video directory does not exist: {video_dir}")

    seen_ids: set[str] = set()
    task_counts: Counter = Counter()
    participant_counts: Counter = Counter()
    for row in rows:
        line = row["__line__"]
        values = {field: (row.get(field) or "").strip() for field in SCHEMA}
        for field, value in values.items():
            if not value:
                violations.append(f"line {line}: {field} must not be empty")

        video_id = values["video_id"]
        safe_id = bool(VIDEO_ID_PATTERN.fullmatch(video_id))
        if video_id and not safe_id:
            violations.append(
                f"line {line}: video_id must be a filename stem using letters, "
                "digits, '.', '_' or '-'"
            )
        if video_id:
            if video_id in seen_ids:
                violations.append(f"line {line}: duplicate video_id {video_id!r}")
            seen_ids.add(video_id)
        if values["task_id"]:
            task_counts[values["task_id"]] += 1
        if values["participant_id"]:
            participant_counts[values["participant_id"]] += 1

        numeric: dict[str, float] = {}
        for field in ("duration", "observation_end", "outcome_start"):
            raw = values[field]
            if not raw:
                continue
            try:
                number = float(raw)
            except ValueError:
                violations.append(f"line {line}: {field} is not numeric: {raw!r}")
                continue
            if not math.isfinite(number):
                violations.append(f"line {line}: {field} must be finite")
                continue
            numeric[field] = number
        if len(numeric) == 3:
            duration = numeric["duration"]
            observation_end = numeric["observation_end"]
            outcome_start = numeric["outcome_start"]
            if not (0 <= observation_end <= outcome_start - 0.5 <= duration):
                violations.append(
                    f"line {line}: timing rule failed: expected "
                    "0 <= observation_end <= outcome_start - 0.5 <= duration"
                )

        if not safe_id or not video_dir.is_dir():
            continue
        path = video_dir / f"{video_id}.mp4"
        if not path.is_file():
            violations.append(f"line {line}: video file does not exist: {path}")
            continue
        measured, error = video_duration(path)
        if error is not None:
            violations.append(f"line {line}: {path}: {error}")
            continue
        if "duration" in numeric and measured is not None:
            if abs(measured - numeric["duration"]) > 0.5:
                violations.append(
                    f"line {line}: annotated duration {numeric['duration']:.3f}s "
                    f"differs from video duration {measured:.3f}s by more than 0.5s"
                )

    return violations, task_counts, participant_counts


def main() -> int:
    args = parse_args()
    violations, task_counts, participant_counts = validate(args.csv, args.video_dir)
    print("Record counts per task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print("Record counts per participant:")
    for participant, count in sorted(participant_counts.items()):
        print(f"  {participant}: {count}")
    if violations:
        print(f"Validation failed with {len(violations)} violation(s):", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
