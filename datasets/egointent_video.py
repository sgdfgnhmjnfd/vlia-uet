from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(
    video_path,
    num_frames=8,
    resize=(224, 224),
):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(
            f"Video has no readable frames: {video_path}"
        )

    indices = np.linspace(
        0,
        frame_count - 1,
        num=min(num_frames, frame_count),
        dtype=int,
    )

    frames = []

    for index in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        ok, frame = cap.read()

        if not ok:
            continue

        # OpenCV uses BGR.
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        if resize is not None:
            frame = cv2.resize(
                frame,
                resize,
            )

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded from: {video_path}"
        )

    return np.stack(
        frames,
        axis=0,
    )
def frames_to_tensor(frames):
    import torch

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected [T, H, W, 3], got {frames.shape}"
        )

    tensor = torch.from_numpy(
        frames.copy()
    ).float()

    # [T, H, W, C] -> [T, C, H, W]
    tensor = tensor.permute(0, 3, 1, 2)

    # uint8 RGB -> float32 [0, 1]
    tensor = tensor / 255.0

    return tensor