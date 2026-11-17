#!/usr/bin/env python3
"""Stage 16 Step 0: data inventory, viewpoint contact sheets, action onset t_on, start_s - t_on, gate D1.

SigLIP so400m on the GPU if free (the first clips were embedded on CPU while VLIA trained; cached per clip). No accuracy is computed here.

t_on (Step 0 item 6):
  - frames at 4 fps from 0 s to start_s + 0.5 s;
  - z_ref = mean of the L2-normalised SigLIP embeddings of the frames in [0, 1] s;
  - d_t = mean |z_t - z_ref|;
  - t_on = the first frame after 1 s with d_t > mu_idle + 3 sd_idle, where mu/sd are taken over the [0, 1] s frames.
  "Exceeds 3x its idle standard deviation" is read as 3 sd above the idle mean, because the idle d_t themselves are
  > 0. If no frame crosses the threshold before start_s, then t_on = start_s and the clip is flagged.
Outputs:
  - results/stage16_scene_motion/inventory.csv
  - results/stage16_scene_motion/step0_summary.json
  - /media/dhqg/d3/vlia_outputs/stage16/ (contact sheets, per-clip d_t; not committed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/dhqg/d3/datasets/own_libero10")
WORK = Path("/media/dhqg/d3/vlia_outputs/stage16")
RES = ROOT / "results/stage16_scene_motion"
ENCODER = "google/siglip-so400m-patch14-384"
TABLES = {"A": (0, 1, 7), "B": (4, 6), "C": (2, 8), "D": (3, 9), "E": (5,)}
SHARED = ("A", "B", "C", "D")
FPS_S = 4.0


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(s.get("tags", {}).get("rotate", rot))
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(num) / float(den),
            "duration_s": float(j["format"]["duration"]), "rotation": rot}


def decode_times(path: Path, times: list[float]):
    """RGB frames (rotation applied by cv2) at the given times, nearest frame; sequential read that stops early."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want = {int(round(t * fps)): t for t in times}
    last = max(want)
    out, i = {}, 0
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[want[i]] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return [out[t] for t in times if t in out], [t for t in times if t in out]


def load_encoder():
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(int(os.environ.get("S16_THREADS", "10")))
    proc = AutoProcessor.from_pretrained(ENCODER, token=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(ENCODER, token=False).to(dev).eval()
    return model, proc


def embed(model, proc, frames, bs=8):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    outs = []
    for i in range(0, len(frames), bs):
        inp = proc(images=[Image.fromarray(f) for f in frames[i:i + bs]], return_tensors="pt")
        with torch.inference_mode():
            f = model.get_image_features(pixel_values=inp["pixel_values"].to(model.device))
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            outs.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(outs)


def onset(times: np.ndarray, z: np.ndarray, start_s: float):
    ref = times <= 1.0 + 1e-9
    zr = z[ref].mean(0)
    d = np.abs(z - zr).mean(1)
    mu, sd = float(d[ref].mean()), float(d[ref].std(ddof=1))
    thr = mu + 3 * sd
    after = np.where((times > 1.0 + 1e-9) & (d > thr))[0]
    if len(after) and times[after[0]] <= start_s + 1e-9:
        return float(times[after[0]]), d, thr, mu, sd, False
    return float(start_s), d, thr, mu, sd, True


def sheet(images, labels, ncol, out: Path, w=320):
    from PIL import Image, ImageDraw

    ims = []
    for im in images:
        p = Image.fromarray(im)
        p = p.resize((w, int(p.height * w / p.width)))
        ims.append(p)
    h = max(i.height for i in ims)
    nrow = (len(ims) + ncol - 1) // ncol
    canvas = Image.new("RGB", (ncol * w, nrow * (h + 18)), "white")
    dr = ImageDraw.Draw(canvas)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (k % ncol) * w, (k // ncol) * (h + 18)
        canvas.paste(im, (x, y + 18))
        dr.text((x + 3, y + 3), lab, fill="black")
    canvas.save(out, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    (WORK / "step0").mkdir(parents=True, exist_ok=True)
    rows = sorted(csv.DictReader((DATA / "labels_all.csv").open()), key=lambda r: (r["scene"], int(r["task"]), r["person"], int(r["rep"])))
    if a.limit:
        rows = rows[:a.limit]
    model, proc = load_encoder()
    inv, first, onset_frames = [], {}, {}
    for r in rows:
        v, path = r["video_id"], DATA / r["file"]
        start_s, end_s = float(r["start_s"]), float(r["end_s"])
        meta = ffprobe(path)
        cache = WORK / "step0" / f"{v}.npz"
        if cache.exists():
            c = np.load(cache)
            times, z = c["times"], c["z"]
            fr, _ = decode_times(path, [0.0])
        else:
            t_grid = list(np.round(np.arange(0, min(meta["duration_s"] - 0.05, start_s + 0.5) + 1e-9, 1 / FPS_S), 3))
            fr, tt = decode_times(path, t_grid)
            times, z = np.array(tt), embed(model, proc, fr)
            np.savez(cache.with_suffix(".tmp.npz"), times=times, z=z.astype(np.float16))
            os.replace(cache.with_suffix(".tmp.npz"), cache)
        first[v] = fr[0]
        t_on, d, thr, mu, sd, flag = onset(times, z.astype(np.float32), start_s)
        (WORK / "step0" / f"{v}_d.json").write_text(json.dumps({"times": times.tolist(), "d": d.tolist(), "thr": thr, "mu": mu, "sd": sd}))
        table = r["scene"]
        inv.append({"video_id": v, "table": table, "task": int(r["task"]), "rep": int(r["rep"]), "person": r["person"],
                    "duration_s": round(meta["duration_s"], 3), "fps": round(meta["fps"], 3), "width": meta["width"],
                    "height": meta["height"], "rotation": meta["rotation"], "has_idle_start": int(t_on >= 1.25),
                    "start_s": start_s, "end_s": end_s, "t_on": t_on, "t_on_flag_no_detect": int(flag),
                    "start_minus_ton": round(start_s - t_on, 3), "view": ""})
        print(v, table, f"t_on {t_on:.2f} start {start_s:.2f} diff {start_s - t_on:.2f}{' (no detection)' if flag else ''}", flush=True)
    # contact sheets of first frames, per table
    for tab in TABLES:
        ids = [x["video_id"] for x in inv if x["table"] == tab]
        if ids:
            sheet([first[i] for i in ids], ids, 9 if len(ids) >= 9 else len(ids), WORK / f"contact_first_frames_{tab}.jpg")
    keys = ["video_id", "table", "task", "rep", "person", "duration_s", "fps", "width", "height", "rotation", "has_idle_start",
            "start_s", "end_s", "t_on", "t_on_flag_no_detect", "start_minus_ton", "view"]
    with (RES / "inventory.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(inv)
    # counts and gate D1
    counts = {}
    for x in inv:
        counts.setdefault(x["table"], {}).setdefault(x["task"], {}).setdefault(x["person"], 0)
        counts[x["table"]][x["task"]][x["person"]] += 1
    d1 = {}
    for tab in SHARED:
        ok = all(sum(1 for p, n in counts.get(tab, {}).get(t, {}).items() if n >= 2) >= 2 for t in TABLES[tab])
        d1[tab] = "PASS" if ok else "FAIL"
    diff = np.array([x["start_minus_ton"] for x in inv])
    by_tab = {tab: [x["start_minus_ton"] for x in inv if x["table"] == tab] for tab in TABLES}
    summ = {"n_clips": len(inv), "counts": counts, "D1": d1,
            "start_minus_ton": {"min": float(diff.min()), "median": float(np.median(diff)), "max": float(diff.max()),
                                "by_table": {k: [float(min(v)), float(np.median(v)), float(max(v))] for k, v in by_tab.items() if v}},
            "n_no_detection": int(sum(x["t_on_flag_no_detect"] for x in inv)),
            "n_t_on_lt_1_25": int(sum(1 - x["has_idle_start"] for x in inv)),
            "rotation_nonzero": [x["video_id"] for x in inv if x["rotation"]],
            "t_on_method": "SigLIP so400m (GPU; first clips CPU), 4 fps, ref [0,1] s, d_t = mean|z_t - z_ref|, threshold mu_idle + 3 sd_idle"}
    (RES / "step0_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
