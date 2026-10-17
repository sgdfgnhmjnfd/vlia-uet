from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_own_dataset.py"
FIELDS = [
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


class OwnDatasetValidatorTests(unittest.TestCase):
    def write_video(self, path: Path, seconds: float = 2.0) -> None:
        fps = 10
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (64, 48),
        )
        if not writer.isOpened():
            self.fail(f"OpenCV VideoWriter could not create {path}")
        for index in range(round(seconds * fps)):
            frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def valid_rows(self) -> list[dict[str, str]]:
        return [
            {
                "video_id": f"clip_{index}",
                "participant_id": "p001",
                "task_id": "sort_books",
                "rep_id": str(index),
                "duration": "2.0",
                "observation_end": "0.5",
                "outcome_start": "1.2",
                "object": "book",
                "target": "shelf",
                "what": "pick up",
                "why": "put on shelf",
                "next": "move toward shelf",
            }
            for index in (1, 2, 3)
        ]

    def invoke(self, csv_path: Path, video_dir: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--csv",
                str(csv_path),
                "--video-dir",
                str(video_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_accepts_three_synthetic_videos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="own-dataset-valid-") as temp:
            root = Path(temp)
            video_dir = root / "videos"
            video_dir.mkdir()
            rows = self.valid_rows()
            for row in rows:
                self.write_video(video_dir / f"{row['video_id']}.mp4")
            csv_path = root / "annotations.csv"
            self.write_csv(csv_path, rows)

            result = self.invoke(csv_path, video_dir)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("sort_books: 3", result.stdout)
            self.assertIn("p001: 3", result.stdout)

    def test_reports_timing_duplicate_and_missing_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="own-dataset-invalid-") as temp:
            root = Path(temp)
            video_dir = root / "videos"
            video_dir.mkdir()
            rows = self.valid_rows()
            self.write_video(video_dir / "clip_1.mp4")
            rows[1]["video_id"] = "clip_1"
            rows[1]["observation_end"] = "1.0"
            rows[2]["video_id"] = "missing_video"
            csv_path = root / "annotations.csv"
            self.write_csv(csv_path, rows)

            result = self.invoke(csv_path, video_dir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate video_id", result.stderr)
            self.assertIn("timing rule failed", result.stderr)
            self.assertIn("video file does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
