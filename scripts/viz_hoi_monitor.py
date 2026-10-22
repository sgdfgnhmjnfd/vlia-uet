#!/usr/bin/env python3
"""Offline MediaPipe hand detection and contact-sheet viewer for EgoIntent."""
from __future__ import annotations

import argparse
import csv
import functools
import html
import json
import os
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hoi_utils import (
    HandDetector,
    sample_exact_video_frames,
    square_crop,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTIONS = ROOT / "results/hoi_detection_v1/detections.jsonl"
DEFAULT_HAND_MODEL = Path("/media/dhqg/d3/models/hand_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples-from", type=Path)
    parser.add_argument("--n-per-video", type=int, default=3)
    parser.add_argument("--videos", nargs="*", default=["auto6"])
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--tau-conf", type=float)
    parser.add_argument("--tau-margin", type=float)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL)
    parser.add_argument("--own-dataset", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--video-dir", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    allowed = {"train", "val"}
    forbidden = sorted({row.get("split") for row in rows} - allowed)
    if forbidden:
        raise ValueError(f"Manifest contains disallowed splits: {forbidden}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate sample_id values.")
    return rows


def summarize_detections(path: Path, out_dir: Path) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    rows = read_jsonl(path)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[("overall", "all")].append(row)
        groups[("scene", row["scene"])].append(row)
        groups[("video", row["video_uid"])].append(row)
        groups[("frame_position", str(row["frame_index"] + 1))].append(row)
        by_sample[row["sample_id"]][int(row["frame_index"])] = row

    output_rows = []
    for (group_type, group), members in sorted(groups.items()):
        detected = sum(bool(row["detected"]) for row in members)
        output_rows.append(
            {
                "group_type": group_type,
                "group": group,
                "frames": len(members),
                "detected_frames": detected,
                "detection_rate": detected / len(members) if members else 0.0,
            }
        )
    summary_csv = out_dir / "detection_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_type",
                "group",
                "frames",
                "detected_frames",
                "detection_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    overall = next(row for row in output_rows if row["group_type"] == "overall")
    scene_rows = [row for row in output_rows if row["group_type"] == "scene"]
    position_rows = sorted(
        (row for row in output_rows if row["group_type"] == "frame_position"),
        key=lambda row: int(row["group"]),
    )
    mean_rate = float(overall["detection_rate"])
    gate_pass = mean_rate >= 0.60 and all(
        float(row["detection_rate"]) >= 0.30 for row in scene_rows
    )
    lines = [
        "# MediaPipe hand-detection summary",
        "",
        f"- Overall: {mean_rate:.4f} ({overall['detected_frames']}/{overall['frames']})",
        f"- Gate: **{'PASS' if gate_pass else 'FAIL'}** "
        "(overall >= 0.60 and every scene >= 0.30)",
        "",
        "## Per scene",
        "",
        "| Scene | Detected frames / frames | Rate |",
        "|---|---:|---:|",
    ]
    for row in scene_rows:
        lines.append(
            f"| {row['group']} | {row['detected_frames']} / {row['frames']} "
            f"| {row['detection_rate']:.4f} |"
        )
    video_rows = [row for row in output_rows if row["group_type"] == "video"]
    lines += [
        "",
        "## Per video",
        "",
        "| Video UID | Detected frames / frames | Rate |",
        "|---|---:|---:|",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['group']} | {row['detected_frames']} / {row['frames']} "
            f"| {row['detection_rate']:.4f} |"
        )
    lines += [
        "",
        "## By sampled frame position",
        "",
        "| Position | Detected frames / frames | Rate |",
        "|---:|---:|---:|",
    ]
    for row in position_rows:
        lines.append(
            f"| {row['group']} | {row['detected_frames']} / {row['frames']} "
            f"| {row['detection_rate']:.4f} |"
        )
    (out_dir / "detection_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    gate = [
        "# Hand-detection gate",
        "",
        f"- Mean frame detection rate: {mean_rate:.4f} "
        f"({overall['detected_frames']}/{overall['frames']})",
        f"- Minimum per-scene rate: "
        f"{min(float(row['detection_rate']) for row in scene_rows):.4f}",
        f"- Verdict: **{'PASS' if gate_pass else 'FAIL'}**",
        "- Preregistered rule: PASS iff overall rate >= 0.60 and every scene >= 0.30.",
        "",
        "## Per-scene rates",
        "",
    ]
    gate.extend(
        f"- `{row['group']}`: {row['detection_rate']:.4f} "
        f"({row['detected_frames']}/{row['frames']})"
        for row in scene_rows
    )
    if not gate_pass:
        gate += [
            "The hand and matched-center branches are skipped in Stage 4. Hand "
            "regions remain display-only; Stage 4 uses full frames only.",
            "",
        ]
    (out_dir / "GATE.md").write_text("\n".join(gate), encoding="utf-8")
    return {
        "overall_rate": mean_rate,
        "gate_pass": gate_pass,
        "scene_rates": {row["group"]: row["detection_rate"] for row in scene_rows},
        "frame_position_rates": {
            int(row["group"]): row["detection_rate"] for row in position_rows
        },
    }, by_sample


def index_detections(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    rows = read_jsonl(path)
    by_sample: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][int(row["frame_index"])] = row
    return rows, by_sample


def run_detection(args: argparse.Namespace) -> int:
    if args.manifest is None:
        raise ValueError("--manifest is required with --detect-only.")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    detections_path = args.out / "detections.jsonl"
    for existing in (
        detections_path,
        args.out / "detection_summary.csv",
        args.out / "detection_summary.md",
        args.out / "GATE.md",
        args.out / "detector_metadata.json",
    ):
        if existing.exists():
            raise FileExistsError(f"Refusing to overwrite {existing}")
    rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(args.hand_model)
    detector_api = detector.api
    frame_count = 0
    detected_count = 0
    start = time.monotonic()
    try:
        with detections_path.open("x", encoding="utf-8") as output:
            for sample_index, record in enumerate(rows, start=1):
                frames = sample_exact_video_frames(record["video_path"])
                if len(frames) != 8:
                    raise ValueError(
                        f"{record['sample_id']}: expected 8 frames, got {len(frames)}"
                    )
                for frame_index, rgb in enumerate(frames):
                    landmarks, box = detector.detect(rgb)
                    n_hands = len(landmarks)
                    detected = n_hands > 0 and box is not None
                    entry = {
                        "sample_id": record["sample_id"],
                        "split": record["split"],
                        "video_uid": record["video_uid"],
                        "scene": record["scene"],
                        "video_path": record["video_path"],
                        "frame_index": frame_index,
                        "detected": detected,
                        "n_hands": n_hands,
                        "box": list(box) if box is not None else None,
                    }
                    output.write(json.dumps(entry) + "\n")
                    frame_count += 1
                    detected_count += int(detected)
                    if frame_count == 200:
                        elapsed = time.monotonic() - start
                        rate = frame_count / max(elapsed, 1e-9)
                        eta = (len(rows) * 8 - frame_count) / rate
                        print(
                            f"first 200 frames: {rate:.2f} frames/s; "
                            f"ETA {eta / 60:.1f} min",
                            flush=True,
                        )
                if sample_index % 100 == 0:
                    output.flush()
                    print(
                        f"samples={sample_index}/{len(rows)} frames={frame_count} "
                        f"running_rate={detected_count / frame_count:.4f}",
                        flush=True,
                    )
    finally:
        detector.close()
    import mediapipe as mp

    (args.out / "detector_metadata.json").write_text(
        json.dumps(
            {
                "mediapipe_version": mp.__version__,
                "api": detector_api,
                "running_mode": "IMAGE",
                "max_hands": 2,
                "min_detection_confidence": 0.5,
                "max_detection_side": 960,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    rates, _ = summarize_detections(detections_path, args.out)
    elapsed = time.monotonic() - start
    print(
        f"Completed {frame_count} frames in {elapsed / 60:.1f} min "
        f"({frame_count / max(elapsed, 1e-9):.2f} frames/s); "
        f"detection rate={rates['overall_rate']:.4f}"
    )
    summary_rows = read_jsonl(detections_path)
    video_groups: dict[str, list[bool]] = defaultdict(list)
    for row in summary_rows:
        video_groups[row["video_uid"]].append(bool(row["detected"]))
    for video_uid, flags in sorted(video_groups.items()):
        print(
            f"video {video_uid}: {sum(flags) / len(flags):.4f} "
            f"({sum(flags)}/{len(flags)})"
        )
    return 0


def choose_samples(rows: list[dict[str, Any]], video_arg: list[str], n_per_video: int) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_uid"]].append(row)
    scenes_by_video = {
        video: sorted({row["scene"] for row in members})[0]
        for video, members in by_video.items()
    }
    if video_arg == ["auto6"]:
        chosen: list[str] = []
        scenes = set()
        for video in sorted(by_video):
            scene = scenes_by_video[video]
            if scene not in scenes:
                chosen.append(video)
                scenes.add(scene)
            if len(chosen) == 6:
                break
        if len(chosen) < 6:
            raise ValueError("Fewer than six distinct scenes are available.")
    else:
        chosen = video_arg
        missing = sorted(set(chosen) - set(by_video))
        if missing:
            raise ValueError(f"Unknown video_uid values: {missing}")
    selected = []
    for video in chosen:
        selected.extend(
            sorted(by_video[video], key=lambda row: row["sample_id"])[:n_per_video]
        )
    return selected


HAND_COLOR = (0, 200, 0)      # RGB: detected hand box
HOI_COLOR = (255, 150, 0)     # RGB: HOI proxy square (display only)
FONT_ENV = "VLIA_FONT"


@functools.lru_cache(maxsize=None)
def get_font(size: int):
    """A TrueType font with Vietnamese coverage (DejaVu Sans), falling back safely."""
    paths = [os.environ.get(FONT_ENV, "")]
    try:
        import matplotlib

        paths.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
    except Exception:
        pass
    paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


class TextLayer:
    """Collect text, then draw it once with PIL so Vietnamese diacritics render."""

    def __init__(self) -> None:
        self.items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []

    def add(self, text: Any, xy: tuple[int, int], size: int = 16, color: tuple[int, int, int] = (15, 15, 15)) -> None:
        self.items.append((str(text), (int(xy[0]), int(xy[1])), size, color))

    def apply(self, canvas: np.ndarray) -> np.ndarray:
        if not self.items:
            return canvas
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for text, (x, y), size, color in self.items:
            font = get_font(size)
            try:
                draw.text((x, y), text, font=font, fill=color, anchor="ls")
            except (UnicodeEncodeError, ValueError):
                safe = text.encode("ascii", "replace").decode("ascii")
                draw.text((x, y - size), safe, font=font, fill=color)
        return np.array(image)


def add_wrapped(layer: TextLayer, text: str, x: int, y: int, width_chars: int, size: int = 16,
                color: tuple[int, int, int] = (15, 15, 15), line_h: int = 22) -> int:
    for part in textwrap.wrap(str(text), width=width_chars, break_long_words=True) or [""]:
        layer.add(part, (x, y), size, color)
        y += line_h
    return y


def pick_scores(record: dict[str, Any] | None, k: int) -> dict[str, Any] | None:
    """Scores for prefix k from either {'scores_by_k': {...}} or a flat record."""
    if not record:
        return None
    by_k = record.get("scores_by_k")
    if by_k is not None:
        return by_k.get(str(k))
    return record


def render_tile(rgb: np.ndarray, box: list[int] | None, tile_w: int, tile_h: int, grey: bool = False) -> np.ndarray:
    """Resize first, then draw boxes with visible thickness (full-res lines vanish after downscaling)."""
    h, w = rgb.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    tile = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    if grey:
        tile = cv2.cvtColor(cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if box is not None:
        ibox = tuple(int(v) for v in box)
        x1, y1, x2, y2 = (int(round(v * scale)) for v in ibox)
        cv2.rectangle(tile, (x1, y1), (x2, y2), HAND_COLOR, 2)
        cx1, cy1, cx2, cy2 = (int(round(v * scale)) for v in square_crop(ibox, h, w))
        cv2.rectangle(tile, (cx1, cy1), (cx2, cy2), HOI_COLOR, 2)
    return tile


def paste(canvas: np.ndarray, tile: np.ndarray, x0: int, y0: int) -> None:
    canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile



def load_scores(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {row["sample_id"]: row for row in data}
    samples = data.get("samples", data)
    if isinstance(samples, list):
        return {row["sample_id"]: row for row in samples}
    return samples


def draw_sample(
    record: dict[str, Any],
    detections: dict[int, dict[str, Any]],
    out_path: Path,
    k: int,
    score_record: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    frames = sample_exact_video_frames(record["video_path"])
    if len(frames) != 8:
        raise ValueError(f"{record['sample_id']}: viewer expected 8 sampled frames")
    tile_w, tile_h, panel_w = 320, 240, 520
    canvas = np.full((tile_h * 2, tile_w * 4 + panel_w, 3), 248, dtype=np.uint8)
    layer = TextLayer()
    tiles = []
    for index, rgb in enumerate(frames):
        detection = detections.get(index)
        box = detection.get("box") if detection is not None else None
        tile = render_tile(rgb, box, tile_w, tile_h, grey=index >= k)
        slot_x, slot_y = (index % 4) * tile_w, (index // 4) * tile_h
        paste(canvas, tile, slot_x, slot_y)
        status = "hand" if box is not None else "no hand"
        label = f"frame {index + 1}: {status}" + (" | after prefix" if index >= k else "")
        layer.add(label, (slot_x + 6, slot_y + tile_h - 8), 15, (25, 25, 25))
        tiles.append(tile)

    x, y = tile_w * 4 + 18, 28
    for line in (
        f"sample: {record['sample_id']}",
        f"scene: {record['scene']} | prefix k={k}",
    ):
        y = add_wrapped(layer, line, x, y, 54)
    y = add_wrapped(layer, f"GT WHAT: {record.get('what', '')}", x, y, 54)
    y = add_wrapped(layer, f"GT WHY: {record.get('why', '')}", x, y, 54)
    y += 6
    y = add_wrapped(layer, "Green box: detected hand. Orange square: HOI proxy.", x, y, 54, 15, (60, 60, 60), 20)
    y = add_wrapped(layer, "Display only; not model input. Greyed frames come after the prefix.", x, y, 54, 15, (60, 60, 60), 20)
    y += 8
    picked = pick_scores(score_record, k)
    if score_record is not None and picked is None:
        y = add_wrapped(layer, f"(no scores stored for k={k})", x, y, 54)
    if picked:
        gt = str(record.get("why", ""))
        for item in picked.get("candidates", [])[:3]:
            text = str(item.get("text", ""))
            score = float(item.get("cosine", item.get("score", 0)))
            mark = " [GT]" if text == gt else ""
            y = add_wrapped(layer, f"WHY: {text}{mark} ({score:.3f})", x, y, 54)
        for key in ("confidence", "margin", "normalized_entropy"):
            if key in picked:
                y = add_wrapped(layer, f"{key}: {float(picked[key]):.3f}", x, y, 54)
        decisions = []
        if args.tau_conf is not None and picked.get("confidence") is not None:
            decisions.append(float(picked["confidence"]) >= args.tau_conf)
        if args.tau_margin is not None and picked.get("margin") is not None:
            decisions.append(float(picked["margin"]) >= args.tau_margin)
        if decisions:
            y = add_wrapped(layer, f"display decision: {'PREDICT' if all(decisions) else 'DEFER'}", x, y, 54)
    canvas = layer.apply(canvas)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if args.show:
        for tile in tiles:
            cv2.imshow("VLIA offline HOI monitor", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
        cv2.destroyAllWindows()


def render_manifest(args: argparse.Namespace) -> int:
    if args.samples_from is None:
        raise ValueError("--samples-from is required for offline manifest rendering.")
    if args.k not in (2, 4, 6, 8):
        raise ValueError("--k must be one of 2, 4, 6, 8.")
    rows = load_manifest(args.samples_from)
    selected = choose_samples(rows, args.videos, args.n_per_video)
    _, by_sample = index_detections(args.detections)
    score_rows = load_scores(args.scores)
    args.out.mkdir(parents=True, exist_ok=True)
    expected_images = [
        args.out / f"{row['sample_id'].replace('/', '__')}_k{args.k}.png"
        for row in selected
    ]
    if any(path.exists() for path in expected_images) or (args.out / "index.html").exists():
        raise FileExistsError(f"Refusing to overwrite existing viewer files under {args.out}")
    index = ["<!doctype html><meta charset='utf-8'><title>VLIA HOI monitor</title><h1>Offline viewer</h1><ul>"]
    for record in selected:
        safe_id = record["sample_id"].replace("/", "__")
        image_name = f"{safe_id}_k{args.k}.png"
        draw_sample(
            record,
            by_sample.get(record["sample_id"], {}),
            args.out / image_name,
            args.k,
            score_rows.get(record["sample_id"]),
            args,
        )
        index.append(
            f"<li><p>{html.escape(record['sample_id'])} — {html.escape(record['scene'])}</p>"
            f"<img src='{html.escape(image_name)}' style='max-width:100%'></li>"
        )
    index.append("</ul>")
    (args.out / "index.html").write_text("\n".join(index), encoding="utf-8")
    print(f"Rendered {len(selected)} contact sheets to {args.out}")
    print("Detection rates for rendered videos:")
    rendered_videos = sorted({row["video_uid"] for row in selected})
    detection_rows = read_jsonl(args.detections)
    for video in rendered_videos:
        subset = [row for row in detection_rows if row["video_uid"] == video]
        rate = sum(row["detected"] for row in subset) / len(subset)
        print(f"  {video}: {rate:.4f} ({sum(row['detected'] for row in subset)}/{len(subset)})")
    return 0


def run_own_dataset(args: argparse.Namespace) -> int:
    if args.csv is None or args.video_dir is None:
        raise ValueError("--own-dataset requires --csv and --video-dir.")
    validation = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_own_dataset.py"),
            "--csv",
            str(args.csv),
            "--video-dir",
            str(args.video_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(validation.stdout, end="")
    if validation.returncode:
        print(validation.stderr, file=sys.stderr, end="")
        raise ValueError("Own-dataset validation failed; refusing to render.")
    if args.k not in (2, 4, 6, 8):
        raise ValueError("--k must be one of 2, 4, 6, 8.")
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("Own-dataset annotation CSV contains no records.")
    scores = load_scores(args.scores)
    args.out.mkdir(parents=True, exist_ok=True)
    existing_outputs = [args.out / f"{record['video_id']}_own.png" for record in records]
    if any(path.exists() for path in existing_outputs) or (args.out / "index.html").exists():
        raise FileExistsError(f"Refusing to overwrite existing viewer files under {args.out}")
    detector = HandDetector(args.hand_model)
    index = [
        "<!doctype html><meta charset='utf-8'><title>Own dataset HOI monitor</title>"
        "<h1>Offline own-dataset viewer</h1><ul>"
    ]
    tile_w, tile_h, panel_w, bar_h = 320, 240, 560, 120
    red, blue, orange, grey_dot = (220, 30, 30), (30, 90, 220), (255, 150, 0), (120, 120, 120)
    try:
        for record in records:
            video_id = record["video_id"]
            frames = sample_exact_video_frames(args.video_dir / f"{video_id}.mp4")
            duration = float(record["duration"])
            obs = float(record["observation_end"])
            outcome = float(record["outcome_start"])
            if duration <= 0:
                raise ValueError(f"{video_id}: duration must be positive")
            n = len(frames)
            # The sampler spreads frames over the whole clip, so frame i is at about duration*i/(n-1).
            times = [duration * i / (n - 1) for i in range(n)]
            canvas = np.full((tile_h * 2 + bar_h, tile_w * 4 + panel_w, 3), 248, dtype=np.uint8)
            layer = TextLayer()
            tiles = []
            for i, rgb in enumerate(frames):
                _, box = detector.detect(rgb)
                after = times[i] > obs
                tile = render_tile(rgb, box, tile_w, tile_h, grey=after)
                x0, y0 = (i % 4) * tile_w, (i // 4) * tile_h
                paste(canvas, tile, x0, y0)
                note = "after observation_end" if after else "visible"
                layer.add(f"frame {i + 1} (~{times[i]:.1f}s): {note}", (x0 + 6, y0 + tile_h - 8), 15, (25, 25, 25))
                tiles.append(tile)
            x, y = tile_w * 4 + 18, 28
            y = add_wrapped(layer, f"{video_id} | {record.get('task_id', '')} | rep {record.get('rep_id', '')}", x, y, 56)
            for label in ("object", "target", "what", "why", "next"):
                y = add_wrapped(layer, f"{label}: {record.get(label, '')}", x, y, 56)
            y += 6
            y = add_wrapped(layer, "Green: hand. Orange square: HOI proxy (display only).", x, y, 56, 15, (60, 60, 60), 20)
            y = add_wrapped(layer, "Greyed frames are after observation_end: the model must not see them.", x, y, 56, 15, (60, 60, 60), 20)
            picked = pick_scores(scores.get(video_id), args.k)
            if picked:
                y += 6
                for item in picked.get("candidates", [])[:3]:
                    score = float(item.get("cosine", item.get("score", 0)))
                    y = add_wrapped(layer, f"WHY: {item.get('text', '')} ({score:.3f})", x, y, 56)
                for key in ("confidence", "margin", "normalized_entropy"):
                    if key in picked:
                        y = add_wrapped(layer, f"{key}: {float(picked[key]):.3f}", x, y, 56)
            bar_x1, bar_x2 = 24, canvas.shape[1] - 24
            bar_y = tile_h * 2 + 36

            def bx(t: float) -> int:
                return int(bar_x1 + (bar_x2 - bar_x1) * min(max(t / duration, 0.0), 1.0))

            cv2.line(canvas, (bar_x1, bar_y), (bar_x2, bar_y), (90, 90, 90), 8)
            for t in times:
                cv2.circle(canvas, (bx(t), bar_y), 5, grey_dot, -1)
            cv2.line(canvas, (bx(obs), bar_y - 16), (bx(obs), bar_y + 16), red, 3)
            cv2.line(canvas, (bx(outcome), bar_y - 16), (bx(outcome), bar_y + 16), blue, 3)
            cv2.line(canvas, (bx(outcome - 0.5), bar_y - 11), (bx(outcome - 0.5), bar_y + 11), orange, 2)
            layer.add(
                f"observation_end={obs:.2f}s (red) | outcome_start={outcome:.2f}s (blue) | "
                f"gap={outcome - obs:.2f}s (rule: >= 0.50s) | orange tick = outcome_start - 0.5s",
                (bar_x1, bar_y + 38), 16, (20, 20, 20),
            )
            canvas = layer.apply(canvas)
            image_name = f"{video_id}_own.png"
            cv2.imwrite(str(args.out / image_name), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            index.append(
                f"<li><p>{html.escape(video_id)}</p><img src='{html.escape(image_name)}' "
                "style='max-width:100%'></li>"
            )
            if args.show:
                for tile in tiles:
                    cv2.imshow("VLIA own-dataset viewer", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                    if (cv2.waitKey(0) & 0xFF) == ord("q"):
                        break
                cv2.destroyAllWindows()
    finally:
        detector.close()
    index.append("</ul>")
    (args.out / "index.html").write_text("\n".join(index), encoding="utf-8")
    print(f"Rendered {len(records)} validated own-dataset videos to {args.out}")
    return 0


def main() -> int:
    args = parse_args()
    if args.detect_only:
        return run_detection(args)
    if args.own_dataset:
        return run_own_dataset(args)
    return render_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())