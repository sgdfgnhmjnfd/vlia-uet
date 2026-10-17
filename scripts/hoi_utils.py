#!/usr/bin/env python3
"""Frame sampling, MediaPipe hand detection, and fixed HOI display geometry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datasets"))

from egointent_video import sample_video_frames  # noqa: E402


NUM_FRAMES = 8
MAX_DETECTION_SIDE = 960


def ensure_num_frames(frames: np.ndarray, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Match torch.linspace(...).round().long() used by the cache writer."""
    if len(frames) == 0:
        raise ValueError("Video produced zero frames.")
    if len(frames) == num_frames:
        return frames
    indices = np.linspace(
        0, len(frames) - 1, num=num_frames, dtype=np.float32
    ).round().astype(np.int64)
    return frames[indices]


def sample_exact_video_frames(video_path: str | Path) -> np.ndarray:
    sampled = sample_video_frames(video_path, num_frames=NUM_FRAMES, resize=None)
    return ensure_num_frames(sampled, NUM_FRAMES)


def hand_bbox(detections: list[Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not detections:
        return None
    points = np.concatenate(
        [
            np.asarray([[landmark.x * width, landmark.y * height] for landmark in hand])
            for hand in detections
        ],
        axis=0,
    )
    x1, y1 = np.floor(points.min(axis=0)).astype(int)
    x2, y2 = np.ceil(points.max(axis=0)).astype(int)
    return (
        int(np.clip(x1, 0, width - 1)),
        int(np.clip(y1, 0, height - 1)),
        int(np.clip(x2, x1 + 1, width)),
        int(np.clip(y2, y1 + 1, height)),
    )


def square_crop(
    box: tuple[int, int, int, int] | None,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """Return fixed pre-registered expanded hand or center crop coordinates."""
    if box is None:
        return (0, 0, width, height)
    x1, y1, x2, y2 = box
    side = int(round(2.0 * max(x2 - x1, y2 - y1)))
    side = int(np.clip(side, round(0.30 * min(height, width)), min(height, width)))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = int(np.clip(left, 0, width - side))
    top = int(np.clip(top, 0, height - side))
    return (left, top, left + side, top + side)


def center_crop_for_side(crop: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int, int, int]:
    side = crop[2] - crop[0]
    left = (width - side) // 2
    top = (height - side) // 2
    return (left, top, left + side, top + side)


def crop_from_box_or_full(frame: np.ndarray, box: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """Return hand crop and matched-size center crop; missing hand means full frame for both."""
    h, w = frame.shape[:2]
    hand_crop = square_crop(box, h, w)
    center_crop = (
        (0, 0, w, h)
        if box is None
        else center_crop_for_side(hand_crop, h, w)
    )
    return frame_crop(frame, hand_crop), frame_crop(frame, center_crop)


def frame_crop(frame: np.ndarray, coords: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = coords
    return frame[y1:y2, x1:x2].copy()


class HandDetector:
    """Static independent-frame hand detection for either supported MediaPipe API."""

    def __init__(self, model_path: Path | None = None) -> None:
        import mediapipe as mp

        self.mp = mp
        self._legacy = None
        self._landmarker = None
        self._tasks = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
            self.api = "solutions.hands"
        elif hasattr(mp, "tasks"):
            if model_path is None or not model_path.is_file():
                raise FileNotFoundError(
                    "MediaPipe Tasks API requires --hand-model pointing to "
                    "hand_landmarker.task"
                )
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._tasks = mp.tasks.vision.HandLandmarker.create_from_options(options)
            self.api = "tasks.HandLandmarker"
        else:
            raise RuntimeError("Installed MediaPipe exposes neither supported Hands API.")

    def detect(self, rgb_frame: np.ndarray) -> tuple[list[Any], tuple[int, int, int, int] | None]:
        h, w = rgb_frame.shape[:2]
        scale = min(1.0, MAX_DETECTION_SIDE / max(h, w))
        if scale < 1.0:
            resized_w, resized_h = max(1, round(w * scale)), max(1, round(h * scale))
            resized = cv2.resize(
                rgb_frame,
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = rgb_frame
            resized_w, resized_h = w, h
        if self._legacy is not None:
            result = self._legacy.process(resized)
            points = result.multi_hand_landmarks or []
            box_small = hand_bbox(points, resized.shape[1], resized.shape[0])
        else:
            image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(resized),
            )
            result = self._tasks.detect(image)
            points = result.hand_landmarks or []
            box_small = hand_bbox(points, resized.shape[1], resized.shape[0])
        if box_small is None:
            return points, None
        if scale < 1.0:
            sx, sy = resized_w / w, resized_h / h
            x1, y1, x2, y2 = box_small
            x1, x2 = int(round(x1 / sx)), int(round(x2 / sx))
            y1, y2 = int(round(y1 / sy)), int(round(y2 / sy))
            box = (
                int(np.clip(x1, 0, w - 1)),
                int(np.clip(y1, 0, h - 1)),
                int(np.clip(x2, x1 + 1, w)),
                int(np.clip(y2, y1 + 1, h)),
            )
        else:
            box = box_small
        return points, box

    def close(self) -> None:
        if self._legacy is not None:
            self._legacy.close()
        if self._tasks is not None:
            self._tasks.close()


def selftest() -> None:
    for count in range(1, 257):
        array = np.arange(count, dtype=np.int64)
        actual = ensure_num_frames(array, 8)
        expected_idx = np.linspace(
            0, count - 1, num=8, dtype=np.float32
        ).round().astype(np.int64)
        assert actual.tolist() == array[expected_idx].tolist(), count
    assert square_crop((40, 40, 60, 60), 100, 120) == (30, 30, 70, 70)
    assert square_crop(None, 100, 120) == (0, 0, 120, 100)
    hand, center = crop_from_box_or_full(np.zeros((100, 120, 3), np.uint8), None)
    assert hand.shape == center.shape == (100, 120, 3)
    box = (2, 2, 18, 18)
    crop = square_crop(box, 100, 120)
    center_box = center_crop_for_side(crop, 100, 120)
    assert crop[0] == 0 and crop[1] == 0
    assert center_box[2] - center_box[0] == crop[2] - crop[0]
    print("selftest: PASS (frame sampling, square geometry, full-frame fallbacks)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    parser.error("Use this module through viz_hoi_monitor.py.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
