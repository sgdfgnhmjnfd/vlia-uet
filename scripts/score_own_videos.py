#!/usr/bin/env python3
"""Score validated own-dataset videos with the frozen SigLIP E1 baseline."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/siglip-base-patch16-224"
VIEWER = ROOT / "scripts/viz_hoi_monitor.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/dhqg/d3/vlia_outputs/own_video_scores.json"),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def validate_own_dataset(csv_path: Path, video_dir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_own_dataset.py"),
            "--csv",
            str(csv_path),
            "--video-dir",
            str(video_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise ValueError("Own-dataset validation failed; refusing to score videos.")


def read_annotations(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Own-dataset annotation CSV contains no records.")
    return rows


def sample_observation_frames(video_path: Path, observation_end: float) -> list[np.ndarray]:
    if not math.isfinite(observation_end) or observation_end < 0:
        raise ValueError(f"Invalid observation_end={observation_end!r} for {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"OpenCV cannot open video: {video_path}")
        timestamps = np.linspace(0.0, observation_end, num=8, dtype=np.float64)
        frames: list[np.ndarray] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000.0))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(
                    f"Could not decode frame at {timestamp:.3f}s in {video_path}"
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        capture.release()


def resolve_local_snapshot(
    model_id: str = MODEL_ID, cache_dir: Path = Path("/media/dhqg/d3/hf_cache")
) -> Path:
    repo_id = model_id.replace("/", "--")
    for root in (
        cache_dir / "hub" / f"models--{repo_id}",
        cache_dir / f"models--{repo_id}",
    ):
        snapshots = root / "snapshots"
        if snapshots.is_dir():
            candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
            for candidate in reversed(candidates):
                if (candidate / "config.json").is_file() and (
                    (candidate / "model.safetensors").is_file()
                    or (candidate / "pytorch_model.bin").is_file()
                ):
                    return candidate
    raise FileNotFoundError(f"No complete local model snapshot found for {model_id}")


def load_model(device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoProcessor

    snapshot = resolve_local_snapshot()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
    return model, processor


def _features(output: Any) -> torch.Tensor:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    return F.normalize(output.float(), dim=-1)


def _score_record(
    sample_id: str,
    task_id: str,
    k: int,
    candidate_texts: list[str],
    scores: np.ndarray,
) -> dict[str, Any]:
    order = np.argsort(-scores, kind="mergesort")
    probs = np.exp((scores - scores.max()) / 0.07)
    probs /= probs.sum()
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "k": k,
        "candidates": [
            {"text": candidate_texts[int(i)], "cosine": float(scores[i])}
            for i in order
        ],
        "confidence": float(probs.max()),
        "margin": float(scores[order[0]] - scores[order[1]]),
        "normalized_entropy": entropy / math.log(len(scores)),
    }


def score_annotations(
    rows: list[dict[str, str]],
    video_dir: Path,
    model: Any,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    by_task: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_task[row["task_id"]].add(row["why"])
    samples = []
    for row in rows:
        sample_id = row["video_id"]
        task_id = row["task_id"]
        candidates = sorted(by_task[task_id])
        if len(candidates) < 2:
            print(
                f"Skipping {sample_id}: task {task_id!r} has fewer than 2 distinct WHY candidates."
            )
            continue
        frames = sample_observation_frames(
            video_dir / f"{sample_id}.mp4", float(row["observation_end"])
        )
        tokens = processor(
            text=candidates,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        images = processor(
            images=[Image.fromarray(frame) for frame in frames],
            return_tensors="pt",
        )
        pixel_values = images["pixel_values"].to(device)
        with torch.inference_mode():
            text_features = _features(model.get_text_features(**tokens))
            frame_features = _features(
                model.get_image_features(pixel_values=pixel_values)
            )
        text_np = text_features.cpu().numpy()
        frame_np = frame_features.cpu().numpy()
        by_k: dict[str, dict[str, Any]] = {}
        for k in (2, 4, 6, 8):
            query = frame_np[:k].mean(axis=0)
            query /= max(float(np.linalg.norm(query)), 1e-12)
            cosines = query @ text_np.T
            by_k[str(k)] = _score_record(
                sample_id, task_id, k, candidates, cosines
            )
        samples.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "scores_by_k": by_k,
                **{
                    key: value
                    for key, value in by_k["8"].items()
                    if key not in {"sample_id", "task_id", "k"}
                },
            }
        )
    return {
        "model": MODEL_ID,
        "sampling": "8 uniform timestamps in [0, observation_end], inclusive",
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    validate_own_dataset(args.csv, args.video_dir)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite scores file: {args.output}")
    rows = read_annotations(args.csv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_model(device)
    scores = score_annotations(rows, args.video_dir, model, processor, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2)
        handle.write("\n")
    print(f"Wrote scores to {args.output}")
    print(
        "Render command: "
        f"{sys.executable} {VIEWER} --own-dataset --csv {args.csv} "
        f"--video-dir {args.video_dir} --scores {args.output} "
        f"--out /media/dhqg/d3/vlia_outputs/own_video_viewer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
