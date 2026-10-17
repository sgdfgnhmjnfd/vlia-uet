#!/usr/bin/env python3
"""Build a train/validation-only manifest for the offline EgoIntent viewer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datasets"))
sys.path.insert(0, str(ROOT / "scripts"))

from cache_egointent_full_temporal_features import ensure_num_frames  # noqa: E402
from egointent_video import sample_video_frames  # noqa: E402


ALLOWED_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/hoi_detection_v1/manifest.jsonl",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def load_manifest_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in ALLOWED_SPLITS:
        split_dir = args.temporal_cache / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Required non-test split cache missing: {split_dir}")
        for cache_path in sorted(split_dir.glob("*.pt")):
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if str(cached.get("split", split)) != split:
                raise ValueError(f"Split metadata mismatch in {cache_path}")
            sample_id = str(cached["sample_id"])
            video_path = (args.video_root / f"{sample_id}.mp4").resolve()
            if not video_path.is_file():
                missing.append(f"{sample_id}: {video_path}")
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "video_uid": str(cached["video_uid"]),
                    "scene": str(cached["scene"]),
                    "event": str(cached.get("event", "")),
                    "step_id": str(cached.get("step_id", "")),
                    "video_path": str(video_path),
                    "why": str(cached["why"]),
                    "what": str(cached["what"]),
                    "num_frames": int(cached["num_frames"]),
                    "original_num_frames": int(cached["original_num_frames"]),
                }
            )
    return records, missing


def verify_decoded_frames(
    records: list[dict[str, Any]], num_frames: int
) -> list[str]:
    mismatches = []
    for index, record in enumerate(records, start=1):
        raw_frames = sample_video_frames(
            record["video_path"], num_frames=num_frames, resize=None
        )
        decoded = len(raw_frames)
        if decoded != record["original_num_frames"]:
            mismatches.append(
                f"{record['sample_id']}: cache original_num_frames="
                f"{record['original_num_frames']}, decoded={decoded}"
            )
            continue
        fixed_frames = ensure_num_frames(raw_frames, num_frames)
        if len(fixed_frames) != num_frames:
            mismatches.append(
                f"{record['sample_id']}: ensure_num_frames returned "
                f"{len(fixed_frames)}, expected {num_frames}"
            )
        if index % 250 == 0:
            print(f"Verified sampled frame counts: {index}/{len(records)}", flush=True)
    return mismatches


def main() -> int:
    args = parse_args()
    if args.num_frames != 8:
        raise ValueError("This manifest is tied to the cached 8-frame features.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {args.output}")

    records, missing = load_manifest_records(args)
    if missing:
        print(f"Missing video files: {len(missing)}", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Duplicate sample_id found across train/val caches.")

    mismatches = verify_decoded_frames(records, args.num_frames)
    mismatch_fraction = len(mismatches) / max(1, len(records))
    print(
        f"samples={len(records)} videos={len({r['video_uid'] for r in records})} "
        f"scenes={len({r['scene'] for r in records})}"
    )
    print(f"missing video files={len(missing)}")
    print(
        f"original_num_frames mismatches={len(mismatches)}/{len(records)} "
        f"({mismatch_fraction:.2%})"
    )
    for item in mismatches:
        print(f"  {item}", file=sys.stderr)
    if mismatch_fraction > 0.02:
        print("Mismatch rate exceeds the 2% stop threshold; manifest not written.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
