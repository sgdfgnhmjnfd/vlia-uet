#!/usr/bin/env python3
"""Stage 22 Step 2 (Trang, STOP A 27/11): label pack for two independent annotators (Trang, Hung).

S7414 (first pointing of each episode). For every episode:
- clip  = front_view | human_front_view side by side, from frame 0 to the end of the first human pointing subtask
          (human_role_subtask_index 44 -> next index), i.e. before the robot reaches; 10 fps, h264;
- still = the last frame of that clip (both views).
Media go to /media/dhqg/d3/datasets/habit/label_media/<sid>/ (never committed).
One CSV per annotator, each with its own shuffled order, no automatic label: k_from_robot_left (1-4), unclear (1/blank), note.
Also writes anchors_<sid>.csv: end of pointing (pedal), robot reach onset (EE motion after the first rest), for the
check that the clip ends before the robot reaches.

  python scripts/habit_make_label_pack.py S7414     # resumable: finished media are skipped
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/media/dhqg/d3/datasets/habit")
RES = Path(__file__).resolve().parents[1] / "results/stage22_habit_supervisor"
POINT_H = {"S7414": 44, "S8789": 43}
ANNOT = {"trang": 1, "hung": 2}  # shuffle seeds


def reach_onset(d: pd.DataFrame, start: int) -> int:
    """First frame >= start where either arm's end effector moves > 5 mm/frame after >= 5 frames of < 2 mm/frame."""
    best = 10 ** 6
    for r in ("robot0", "robot1"):
        p = np.stack(d[f"{r}.observation.state.cartesian_position"].values)[:, :3]
        v = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
        for t in range(max(start, 5), len(v)):
            if v[t] > 0.005 and (v[t - 5:t] < 0.002).all():
                best = min(best, t)
                break
    return best


def main() -> int:
    sid = sys.argv[1]
    eps = [e for e in map(json.loads, open(ROOT / "full/meta/episodes.jsonl")) if e["sid"] == sid]
    out = ROOT / "label_media" / sid
    out.mkdir(parents=True, exist_ok=True)
    anchors = []
    for e in eps:
        i = e["episode_index"]
        c = f"chunk-{i // 1000:03d}"
        pq = ROOT / f"full/data/{c}/episode_{i:06d}.parquet"
        fv = ROOT / f"full/videos/{c}/observation.images.front_view/episode_{i:06d}.mp4"
        hv = ROOT / f"full/videos/{c}/observation.images.human_front_view/episode_{i:06d}.mp4"
        if not (pq.exists() and fv.exists() and hv.exists()):
            print("missing", i, flush=True)
            continue
        d = pd.read_parquet(pq)
        h = d["human_role_subtask_index"].to_numpy()
        on = np.where(h == POINT_H[sid])[0]
        if len(on) == 0:
            anchors.append({"episode_index": i, "point_start": "", "point_end": "", "reach_onset": "", "status": "no pointing subtask"})
            continue
        s = int(on[0])
        after = np.where((np.arange(len(h)) > s) & (h != POINT_H[sid]))[0]
        end = int(after[0]) if len(after) else len(h)
        ro = reach_onset(d, 20)
        anchors.append({"episode_index": i, "point_start": s, "point_end": end, "reach_onset": ro,
                        "reach_minus_point_end": ro - end, "length": len(h), "status": "ok"})
        clip, still = out / f"{i:06d}.mp4", out / f"{i:06d}.jpg"
        if not clip.exists():
            fl = (f"[0:v]trim=start_frame=0:end_frame={end},setpts=PTS-STARTPTS[a];"
                  f"[1:v]trim=start_frame=0:end_frame={end},setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2[v]")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(fv), "-i", str(hv), "-filter_complex", fl, "-map", "[v]",
                            "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(clip) + ".tmp.mp4"], check=True)
            Path(str(clip) + ".tmp.mp4").rename(clip)
        if not still.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.15", "-i", str(clip), "-frames:v", "1", str(still)], check=True)
    RES.mkdir(parents=True, exist_ok=True)
    with (RES / f"anchors_{sid}.csv").open("w", newline="") as fh:
        keys = []
        for a in anchors:
            keys += [k for k in a if k not in keys]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(anchors)
    ok = [a["episode_index"] for a in anchors if a["status"] == "ok"]
    for name, seed in ANNOT.items():
        f = RES / f"label_{sid}_{name}.csv"
        if f.exists():
            print("keep existing", f)  # never overwrite an annotator file
            continue
        order = np.random.default_rng(seed).permutation(ok)
        with f.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["order", "episode_index", "clip", "still", "k_from_robot_left", "unclear", "note"])
            for n, i in enumerate(order, 1):
                w.writerow([n, int(i), str(out / f"{int(i):06d}.mp4"), str(out / f"{int(i):06d}.jpg"), "", "", ""])
    r = np.array([a["reach_minus_point_end"] for a in anchors if a["status"] == "ok"])
    print(f"{sid}: {len(ok)} episodes with pointing; reach onset - pointing end (frames): min {r.min()} median {np.median(r)} "
          f"max {r.max()}; reach before pointing end: {(r < 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
