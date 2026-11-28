#!/usr/bin/env python
"""Stage 23 Breakfast: same action, different intent.

Subcommands: inventory, cache, selftest, partA, partB, table.
CPU only. Run with:
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
  export HF_HOME=/data/hf_cache
  nice -n 10 .venv/bin/python -I breakfast/scripts/stage23_breakfast.py <cmd>
"""
import argparse
import collections
import csv
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

ZIP = "/data/breakfast_raw/data.zip"
META = "/data/breakfast_raw/meta/data/breakfast"
CACHE = "/data/breakfast_cache"
RES = os.path.expanduser("~/vlia-nha/breakfast/results/stage23_breakfast")
FPS = 15
SIL = "SIL"


# ---------------------------------------------------------------- metadata
def parse_vid(vid):
    """'P03_cam01_P03_cereals' -> (person, cam, dish)."""
    p = vid.split("_")
    assert len(p) == 4 and p[0] == p[2], vid
    return p[0], p[1], p[3]


def list_videos():
    return sorted(f[:-4] for f in os.listdir(f"{META}/groundTruth") if f.endswith(".txt"))


def read_gt(vid):
    with open(f"{META}/groundTruth/{vid}.txt") as f:
        return [l.strip() for l in f if l.strip()]


def segments(labels):
    """Run-length segments: list of (label, start, end_exclusive)."""
    out = []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((labels[s], s, i))
            s = i
    return out


def read_splits():
    sp = {}
    for k in range(1, 5):
        d = {}
        for part in ("train", "test"):
            with open(f"{META}/splits/{part}.split{k}.bundle") as f:
                d[part] = [l.strip()[:-4] for l in f if l.strip()]
        sp[k] = d
    return sp


def feat_header(z, vid):
    f = z.open(f"data/breakfast/features/{vid}.npy")
    v = np.lib.format.read_magic(f)
    shape, fortran, dtype = np.lib.format._read_array_header(f, v)
    return shape, fortran, dtype


def load_feat(z, vid):
    """Return (T, 2048) float32."""
    a = np.load(io.BytesIO(z.read(f"data/breakfast/features/{vid}.npy")))
    assert a.shape[0] == 2048, a.shape
    return np.ascontiguousarray(a.T)


# ---------------------------------------------------------------- inventory
def select_shared(rows, min_dishes=3, min_segs=20, min_people=8, max_actions=8):
    """rows: list of dict(action, dish, n_segs, n_people, ...). Returns A* list."""
    by_a = collections.defaultdict(list)
    for r in rows:
        if r["n_segs"] >= min_segs and r["n_people"] >= min_people:
            by_a[r["action"]].append(r)
    cand = []
    for a, rs in by_a.items():
        if len(rs) >= min_dishes:
            cand.append(dict(action=a, K=len(rs), dishes=sorted(r["dish"] for r in rs),
                             n_total=sum(r["n_segs"] for r in rs)))
    cand.sort(key=lambda c: (-c["n_total"], c["action"]))
    return cand[:max_actions], cand


def cmd_inventory(args):
    t0 = time.time()
    vids = list_videos()
    z = zipfile.ZipFile(ZIP)
    znames = set(z.namelist())
    out = []
    P = lambda *s: (out.append(" ".join(str(x) for x in s)), print(*s))

    P("# Stage 23 inventory (Breakfast, MS-TCN I3D features)\n")
    P(f"- zip: {ZIP}  size={os.path.getsize(ZIP)}")
    P(f"- videos (groundTruth files): {len(vids)}")
    missing = [v for v in vids if f"data/breakfast/features/{v}.npy" not in znames]
    P(f"- videos missing feature file: {len(missing)}")
    P("- name rule: `P<pp>_<cam>_P<pp>_<dish>` split on '_'; person=field0 (==field2), cam=field1, dish=field3")
    for v in vids[:5]:
        P(f"  - {v} -> {parse_vid(v)}")
    meta = {v: parse_vid(v) for v in vids}
    people = sorted({m[0] for m in meta.values()})
    cams = collections.Counter(m[1] for m in meta.values())
    dishes = collections.Counter(m[2] for m in meta.values())
    recs = collections.Counter((m[0], m[2]) for m in meta.values())
    P(f"- people: {len(people)} ({people[0]}..{people[-1]})")
    P(f"- cams: {dict(cams)}")
    P(f"- dishes ({len(dishes)}): {dict(sorted(dishes.items()))}")
    P(f"- recordings (person+dish): {len(recs)}; videos per recording: {dict(sorted(collections.Counter(recs.values()).items()))}")

    # splits
    sp = read_splits()
    P("\n## Official splits")
    leak = False
    for k, d in sp.items():
        tr = {meta[v][0] for v in d["train"]}
        te = {meta[v][0] for v in d["test"]}
        inter = tr & te
        leak |= bool(inter)
        P(f"- split{k}: train {len(d['train'])} vids / {len(tr)} people; test {len(d['test'])} vids / {len(te)} people "
          f"({', '.join(sorted(te))}); overlap people = {sorted(inter)}; train+test = {len(d['train'])+len(d['test'])}")
    allte = collections.Counter(v for d in sp.values() for v in d["test"])
    P(f"- each video in test exactly once across 4 splits: {set(allte.values()) == {1} and len(allte) == len(vids)}")

    # features vs gt
    P("\n## Features vs groundTruth (headers only)")
    diffs = []
    shapes = collections.Counter()
    gt_len = {}
    for v in vids:
        shape, fortran, dtype = feat_header(z, v)
        shapes[(shape[0], str(dtype))] += 1
        n = len(read_gt(v))
        gt_len[v] = n
        diffs.append(shape[1] - n)
    diffs = np.array(diffs)
    P(f"- feature (dim,dtype) counts: {dict(shapes)}; stored as (2048, T)")
    P(f"- T_feat - T_gt: n_mismatch={int((diffs != 0).sum())}, min={diffs.min()}, max={diffs.max()} (mismatch -> crop to shorter)")
    L = np.array(list(gt_len.values())) / FPS
    P(f"- video length (s, at {FPS} fps per MS-TCN): median {np.median(L):.1f}, min {L.min():.1f}, max {L.max():.1f}")

    # fine labels
    P("\n## Fine action labels")
    with open(f"{META}/mapping.txt") as f:
        mapping = [l.split()[1] for l in f if l.strip()]
    P(f"- mapping.txt: {len(mapping)} labels, SIL present: {SIL in mapping}")
    stats = collections.defaultdict(lambda: dict(n=0, people=set(), recs=set(), durs=[]))
    nseg_all = 0
    for v in vids:
        pp, cam, dish = meta[v]
        for lab, s, e in segments(read_gt(v)):
            nseg_all += 1
            if lab == SIL:
                continue
            d = (e - s) / FPS
            if d < 1.0:
                continue
            st = stats[(lab, dish)]
            st["n"] += 1
            st["people"].add(pp)
            st["recs"].add((pp, dish))
            st["durs"].append(d)
    P(f"- total segments (incl. SIL): {nseg_all}")
    rows = []
    for (a, dish), st in sorted(stats.items()):
        rows.append(dict(action=a, dish=dish, n_segs=st["n"], n_people=len(st["people"]),
                         n_recs=len(st["recs"]), median_dur_s=round(float(np.median(st["durs"])), 2)))
    acts_dishes = collections.Counter(r["action"] for r in rows)
    P(f"- non-SIL actions appearing in >=2 dishes (any count, seg>=1s): "
      f"{sorted(a for a, c in acts_dishes.items() if c >= 2)}")
    with open(f"{RES}/shared_actions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    Astar, cand = select_shared(rows)
    P("\n## A* (rule: >=3 dishes, each kept dish >=20 segs from >=8 people, seg>=1s, top-8 by n segs)")
    P(f"- candidates passing: {len(cand)}; kept: {len(Astar)}")
    rmap = {(r["action"], r["dish"]): r for r in rows}
    for c in Astar:
        det = ", ".join(f"{d} {rmap[(c['action'], d)]['n_segs']}seg/{rmap[(c['action'], d)]['n_people']}p" for d in c["dishes"])
        P(f"- {c['action']}: K={c['K']} | {det}")
    for c in cand[len(Astar):]:
        P(f"- (dropped by top-8) {c['action']}: K={c['K']} n={c['n_total']}")
    json.dump(Astar, open(f"{RES}/astar.json", "w"), indent=1)

    gate = (not missing) and (not leak) and len(Astar) >= 3
    P(f"\n## Gate D0: data readable={not missing}, no split leak={not leak}, |A*|={len(Astar)}>=3 -> {'PASS' if gate else 'FAIL'}")
    P(f"\n(inventory took {time.time()-t0:.0f}s)")
    with open(f"{RES}/inventory.md", "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stage23_core import cmd_cache, cmd_selftest, cmd_partA, cmd_partB, cmd_table
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--splits", default="1,2,3,4")
    args, rest = ap.parse_known_args()
    os.makedirs(RES, exist_ok=True)
    {"inventory": cmd_inventory, "cache": cmd_cache, "selftest": cmd_selftest,
     "partA": cmd_partA, "partB": cmd_partB, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
