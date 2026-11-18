#!/usr/bin/env python3
"""Stage 17: hand-region (HOI) features on the 90 own clips (prereg: results/prereg_stage17_hoi.md).

  g0      (CPU, venv /media/dhqg/d3/venv_mp: mediapipe 0.10.21) MediaPipe Hands on the Stage 16 4 fps grid: per-frame
          detections cached per clip, detection rates per window, gate G0, eye-check sheets for 10 random clips
  (Step 2 subcommands are added only after G0 PASS and Trang's go.)

Frames, windows and metadata come from scripts/run_stage16_scene_motion.py (Stage 16), imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage16_scene_motion as S16  # noqa: E402

WORK = Path("/media/dhqg/d3/vlia_outputs/stage17_hoi")
DET = WORK / "det"
RES = ROOT / "results/stage17_hoi"
MP_SETTINGS = dict(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5)
TABLES = ("A", "B", "C", "D", "E")
WINDOWS = ("primary [start, start+2]", "pre-touch [t_on, start)", "action [start, end]", "idle [0, t_on-0.25]")


def windows(m: dict) -> dict:
    return {WINDOWS[0]: (m["start"], m["start"] + 2.0, True), WINDOWS[1]: (m["t_on"], m["start"], False),
            WINDOWS[2]: (m["start"], m["end"], True), WINDOWS[3]: (0.0, m["t_on"] - 0.25, True)}


def in_window(t: np.ndarray, a: float, b: float, incl_b: bool) -> np.ndarray:
    return (t >= a - 1e-9) & ((t <= b + 1e-9) if incl_b else (t < b - 1e-9))


def read_grid(path: Path, times: np.ndarray):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = {int(round(t * fps)): i for i, t in enumerate(times)}
    out, i, last = [None] * len(times), 0, max(want)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[want[i]] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def detect_clip(hands, m: dict) -> dict:
    times = S16.grid_times(m["dur"])
    frames = read_grid(S16.DATA / m["file"], times)
    det = []
    for fr in frames:
        if fr is None:
            det.append(None)
            continue
        h, w = fr.shape[:2]
        r = hands.process(fr)
        boxes = []
        for lm, hd in zip(r.multi_hand_landmarks or [], r.multi_handedness or []):
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            boxes.append({"box": [min(xs), min(ys), max(xs), max(ys)], "score": hd.classification[0].score,
                          "label": hd.classification[0].label})
        det.append(boxes)
    return {"times": times.tolist(), "size": [int(frames[0].shape[1]), int(frames[0].shape[0])], "det": det}


def cmd_g0(args) -> int:
    import cv2
    import mediapipe as mp

    DET.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    meta = S16.load_meta()
    hands = mp.solutions.hands.Hands(**MP_SETTINGS)
    for v, m in meta.items():
        out = DET / f"{v}.json"
        if out.exists():
            continue
        d = detect_clip(hands, m)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, out)
        print(v, "frames", len(d["times"]), "with hand", sum(1 for x in d["det"] if x), flush=True)
    rows = []
    for v, m in meta.items():
        d = json.loads((DET / f"{v}.json").read_text())
        t = np.array(d["times"])
        hit = np.array([bool(x) for x in d["det"]])
        for wname, (a, b, ib) in windows(m).items():
            sel = in_window(t, a, b, ib)
            n, k = int(sel.sum()), int(hit[sel].sum())
            rows.append({"video_id": v, "table": m["table"], "task": m["task"], "person": m["person"], "rep": m["rep"],
                         "window": wname, "n_frames": n, "n_detected": k, "rate": k / n if n else ""})
    S16.write_csv(RES / "g0_detection.csv", rows)
    # summaries (frame-pooled rates)
    summ = []
    for wname in WINDOWS:
        rw = [r for r in rows if r["window"] == wname]
        for key, groups in (("all", ["all"]), ("table", TABLES), ("person", S16.PERSONS)):
            for g in groups:
                rr = rw if key == "all" else [r for r in rw if r[key] == g]
                n, k = sum(r["n_frames"] for r in rr), sum(r["n_detected"] for r in rr)
                summ.append({"window": wname, "by": key, "group": g, "n_frames": n, "n_detected": k, "rate": k / n if n else None})
    S16.write_csv(RES / "g0_summary.csv", summ)
    prim = {s["group"]: s["rate"] for s in summ if s["window"] == WINDOWS[0] and s["by"] in ("all", "table")}
    gate_rates = prim["all"] >= 0.60 and all(prim[t] >= 0.30 for t in TABLES)
    # eye-check sheets for 10 random clips (seed 0)
    sh = WORK / "g0_sheets"
    sh.mkdir(parents=True, exist_ok=True)
    pick = sorted(np.random.default_rng(0).choice(sorted(meta), 10, replace=False).tolist())
    for v in pick:
        m = meta[v]
        d = json.loads((DET / f"{v}.json").read_text())
        t = np.array(d["times"])
        a, b, ib = windows(m)[WINDOWS[0]]
        idx = np.where(in_window(t, a, b, ib))[0]
        frames = read_grid(S16.DATA / m["file"], t[idx])
        tiles = []
        for j, fr in zip(idx, frames):
            img = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            for bx in d["det"][j] or []:
                x0, y0, x1, y1 = map(int, bx["box"])
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 6)
            img = cv2.resize(img, (480, 270))
            cv2.putText(img, f"{v} {t[j]:.2f}s n={len(d['det'][j] or [])}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            tiles.append(img)
        while len(tiles) % 3:
            tiles.append(np.full_like(tiles[0], 255))
        rows_img = [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)]
        cv2.imwrite(str(sh / f"{v}.jpg"), np.concatenate(rows_img, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 80])
    res = {"primary_rates": prim, "gate_rates_pass": bool(gate_rates), "eye_check_clips": pick, "settings": MP_SETTINGS}
    (RES / "g0_gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("g0").set_defaults(fn=cmd_g0)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
