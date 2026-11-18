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


# ----------------------------------------------------------------------
# G0b: exploration outside the prereg (plan in results/stage17_hoi/DEVIATIONS.md, written before running)
# ----------------------------------------------------------------------
TASK_MODEL = WORK / "models/hand_landmarker.task"


def make_detector(cfg: str):
    """Returns f(rgb) -> list of {"box": [x0, y0, x1, y1], "score": s} in frame pixels."""
    import mediapipe as mp

    def lm_boxes(landmark_lists, w, h, x0=0, y0=0, scores=None):
        out = []
        for i, lm in enumerate(landmark_lists):
            xs = [p.x * w + x0 for p in lm]
            ys = [p.y * h + y0 for p in lm]
            out.append({"box": [min(xs), min(ys), max(xs), max(ys)], "score": None if scores is None else scores[i]})
        return out

    if cfg in ("i", "ii"):
        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.3)

        def run(img, x0=0, y0=0):
            r = hands.process(img)
            h, w = img.shape[:2]
            return lm_boxes([l.landmark for l in (r.multi_hand_landmarks or [])], w, h, x0, y0,
                            [c.classification[0].score for c in (r.multi_handedness or [])])

        if cfg == "i":
            return run

        def tiles(img):
            h, w = img.shape[:2]
            tw, th = int(round(0.6 * w)), int(round(0.6 * h))
            out = []
            for x0, y0 in ((0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)):
                out += run(np.ascontiguousarray(img[y0:y0 + th, x0:x0 + tw]), x0, y0)
            return out

        return tiles
    if cfg == "iii":
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(TASK_MODEL)),
                                            running_mode=vision.RunningMode.IMAGE, num_hands=2,
                                            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
                                            min_tracking_confidence=0.3)
        lm = vision.HandLandmarker.create_from_options(opts)

        def run3(img):
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
            h, w = img.shape[:2]
            return lm_boxes(r.hand_landmarks, w, h, scores=[x[0].score for x in r.handedness])

        return run3
    raise SystemExit(f"config {cfg} not implemented (iv is named in DEVIATIONS before it is run)")


def cmd_g0b(args) -> int:
    cfg = args.config
    det_dir = WORK / f"det_g0b_{cfg}"
    det_dir.mkdir(parents=True, exist_ok=True)
    meta = S16.load_meta()
    detect = make_detector(cfg)
    for v, m in meta.items():
        out = det_dir / f"{v}.json"
        if out.exists():
            continue
        times = S16.grid_times(m["dur"])
        frames = read_grid(S16.DATA / m["file"], times)
        d = {"times": times.tolist(), "size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
             "det": [None if fr is None else detect(fr) for fr in frames]}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, out)
        print(cfg, v, "with hand", sum(1 for x in d["det"] if x), "/", len(d["det"]), flush=True)
    print(f"G0B {cfg} DONE", flush=True)
    return 0


def rates_for(det_dir: Path, meta: dict):
    rows = []
    for v, m in meta.items():
        d = json.loads((det_dir / f"{v}.json").read_text())
        t = np.array(d["times"])
        hit = np.array([bool(x) for x in d["det"]])
        for wname, (a, b, ib) in windows(m).items():
            sel = in_window(t, a, b, ib)
            rows.append({"video_id": v, "table": m["table"], "person": m["person"], "window": wname,
                         "n_frames": int(sel.sum()), "n_detected": int(hit[sel].sum())})
    return rows


def cmd_g0b_summary(args) -> int:
    meta = S16.load_meta()
    out, sel = [], {}
    for cfg, det_dir in (("G0 (prereg, conf 0.5)", DET),) + tuple((c, WORK / f"det_g0b_{c}") for c in ("i", "ii", "iii", "iv")):
        if not det_dir.exists() or len(list(det_dir.glob("t*.json"))) < len(meta):
            continue
        rows = rates_for(det_dir, meta)
        for wname in (WINDOWS[0], WINDOWS[3]):
            rw = [r for r in rows if r["window"] == wname]
            groups = [("all", None, None), ("A-C", "table", ("A", "B", "C")), ("D-E", "table", ("D", "E"))] + \
                     [(t, "table", (t,)) for t in TABLES] + [(p, "person", (p,)) for p in S16.PERSONS] + \
                     [(f"{t}-{p}", "tp", (t, p)) for t in TABLES for p in S16.PERSONS]
            for g, key, vals in groups:
                if key is None:
                    rr = rw
                elif key == "tp":
                    rr = [r for r in rw if r["table"] == vals[0] and r["person"] == vals[1]]
                else:
                    rr = [r for r in rw if r[key] in vals]
                n, k = sum(r["n_frames"] for r in rr), sum(r["n_detected"] for r in rr)
                out.append({"config": cfg, "window": wname, "group": g, "n_frames": n, "n_detected": k, "rate": k / n if n else None})
        get = lambda w, g: next(x["rate"] for x in out if x["config"] == cfg and x["window"] == w and x["group"] == g)  # noqa: E731
        sel[cfg] = {"AC_primary": get(WINDOWS[0], "A-C"), "AC_idle_fp": get(WINDOWS[3], "A-C"),
                    "all_primary": get(WINDOWS[0], "all"), "DE_primary": get(WINDOWS[0], "D-E"),
                    "tables": {t: get(WINDOWS[0], t) for t in TABLES}, "idle_fp_all": get(WINDOWS[3], "all")}
        sel[cfg]["gate_rates_pass"] = bool(sel[cfg]["all_primary"] >= 0.60 and all(r >= 0.30 for r in sel[cfg]["tables"].values()))
    S16.write_csv(RES / "g0b_summary.csv", out)
    cands = [c for c in ("i", "ii", "iii", "iv") if c in sel and sel[c]["AC_idle_fp"] <= 0.05]
    best = max(cands, key=lambda c: (sel[c]["AC_primary"], -("i", "ii", "iii", "iv").index(c))) if cands else None
    res = {"per_config": sel, "chosen_on_A-C": best}
    (RES / "g0b_gate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def cmd_g0b_sheets(args) -> int:
    import cv2

    meta = S16.load_meta()
    det_dir = WORK / f"det_g0b_{args.config}"
    sh = WORK / f"g0b_sheets_{args.config}"
    sh.mkdir(parents=True, exist_ok=True)
    pick = sorted(np.random.default_rng(0).choice(sorted(meta), 10, replace=False).tolist())
    for v in pick:
        m = meta[v]
        d = json.loads((det_dir / f"{v}.json").read_text())
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
        cv2.imwrite(str(sh / f"{v}.jpg"), np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)], axis=0),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("sheets", sh)
    return 0


def crop_box(boxes, W, H):
    """Union of hand boxes -> square of 2.0x its larger side around its centre, clipped to the frame, side >= 128."""
    x0 = min(b["box"][0] for b in boxes); y0 = min(b["box"][1] for b in boxes)
    x1 = max(b["box"][2] for b in boxes); y1 = max(b["box"][3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(128.0, 2.0 * max(x1 - x0, y1 - y0))
    a, b = max(0, int(round(cx - s / 2))), max(0, int(round(cy - s / 2)))
    c, d = min(W, int(round(cx + s / 2))), min(H, int(round(cy + s / 2)))
    return a, b, c, d


def cmd_crops(args) -> int:
    """Step 2a (CPU): per 4 fps grid frame a hand crop (384x384) from the config's detections; the last box is carried
    forward for at most 1.0 s. Saved per clip as uint8 npz under WORK/crops_<config>/ (not in git)."""
    import cv2

    meta = S16.load_meta()
    det_dir = WORK / f"det_g0b_{args.config}"
    out_dir = WORK / f"crops_{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for v, m in meta.items():
        out = out_dir / f"{v}.npz"
        d = json.loads((det_dir / f"{v}.json").read_text())
        t = np.array(d["times"])
        W, H = d["size"]
        boxes, src, last, last_t = [], [], None, -1e9
        for i, ti in enumerate(t):
            if d["det"][i]:
                last, last_t = crop_box(d["det"][i], W, H), ti
                boxes.append(last); src.append(1)
            elif last is not None and ti - last_t <= 1.0 + 1e-9:
                boxes.append(last); src.append(2)
            else:
                boxes.append(None); src.append(0)
        has = np.array([b is not None for b in boxes])
        log.append({"video_id": v, "table": m["table"], "person": m["person"], "n_frames": len(t),
                    "share_with_crop": float(has.mean()), "share_detected": float(np.mean(np.array(src) == 1)),
                    "share_carried": float(np.mean(np.array(src) == 2))})
        if out.exists():
            continue
        frames = read_grid(S16.DATA / m["file"], t[has])
        crops = np.stack([cv2.resize(fr[b[1]:b[3], b[0]:b[2]], (384, 384), interpolation=cv2.INTER_AREA)
                          for fr, b in zip(frames, [b for b in boxes if b is not None])]) if has.any() else np.zeros((0, 384, 384, 3), np.uint8)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, times=t, has_crop=has, source=np.array(src), boxes=np.array([b if b else (-1, -1, -1, -1) for b in boxes]), crops=crops)
        os.replace(tmp, out)
        print(v, "crops", int(has.sum()), "/", len(t), flush=True)
    S16.write_csv(RES / f"crop_coverage_{args.config}.csv", log)
    print("CROPS DONE")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    p = None
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("g0").set_defaults(fn=cmd_g0)
    p = sub.add_parser("g0b")
    p.add_argument("--config", required=True, choices=("i", "ii", "iii", "iv"))
    p.set_defaults(fn=cmd_g0b)
    sub.add_parser("g0b-summary").set_defaults(fn=cmd_g0b_summary)
    p = sub.add_parser("crops")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=cmd_crops)
    p = sub.add_parser("g0b-sheets")
    p.add_argument("--config", required=True)
    p.set_defaults(fn=cmd_g0b_sheets)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
