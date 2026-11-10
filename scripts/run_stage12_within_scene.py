#!/usr/bin/env python3
"""Stage 12 post-hoc, descriptive: within-scene re-scoring of the saved F scores (C2, T1, demos 0..29).

Does watching more of the demo help F choose between the tasks that share a scene? No new model run.
Plan: results/prereg_stage12_within_scene.md.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage12_e2e as S  # noqa: E402

SCORES = S.WORK / "scores.npz"
BDDL_GLOB = "/home/dhqg/lerobot/.venv/lib/python3*/site-packages/libero/libero/bddl_files/libero_10/*.bddl"
OUT = ROOT / "results/stage12_within_scene"
PAIRS = ((0, 1), (7, 5), (5, 9))
DIFFS = (25, 50)


def scenes_from_bddl(paths, original: dict) -> dict:
    """task id -> scene, from `<SCENE>_<task text>.bddl` file names."""
    inv = {S.norm_task(v): int(k) for k, v in original.items()}
    out = {}
    for p in paths:
        m = re.match(r"^(.*?_SCENE\d+)_(.*)$", Path(p).stem)
        t = inv.get(S.norm_task(m.group(2).replace("_", " ")))
        if t is not None:
            out[t] = m.group(1)
    if sorted(out) != sorted(inv.values()):
        raise SystemExit(f"STOP: scene mapping incomplete: {out}")
    return out


def family(scene: str) -> str:
    return re.sub(r"_SCENE\d+$", "", scene)


def groups(label_of: dict) -> dict:
    """task -> list of tasks sharing its label (only labels with >= 2 tasks)."""
    by = {}
    for t, g in label_of.items():
        by.setdefault(g, []).append(t)
    return {t: sorted(by[g]) for t, g in label_of.items() if len(by[g]) >= 2}


def masked_correct(scores: np.ndarray, true: int, cands) -> bool:
    """Top-1 among `cands` only; a tie counts as wrong."""
    cands = list(cands)
    return S.strict_correct(scores[cands], cands.index(true))


def masked_margin(scores: np.ndarray, cands) -> float:
    return S.top2(scores[list(cands)])[1]


def write_csv(path, rows):
    import csv

    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def analyze(keys, scores, scene_of, out: Path, meta: dict) -> dict:
    """keys (N, 3) = (task, demo, p); scores (N, 10). Query demos only (demo < 30)."""
    out.mkdir(parents=True, exist_ok=True)
    q = keys[:, 1] < S.N_QUERY
    idx = {(int(t), int(d), int(p)): i for i, (t, d, p) in enumerate(keys) if q[i]}
    demos = sorted({(t, d) for t, d, _ in idx})
    ps = sorted({p for _, _, p in idx})
    fam_of = {t: family(s) for t, s in scene_of.items()}
    rows, pair_rows, margin_rows = [], [], []

    def block(kind, members_of, sel):
        hits = {p: np.array([masked_correct(scores[idx[(t, d, p)]], t, members_of[t]) for t, d in sel], float) for p in ps}
        clusters = np.arange(len(sel))  # one cluster per demo
        chance = float(np.mean([1 / len(members_of[t]) for t, _ in sel]))
        for p in ps:
            a, lo, hi = S.cluster_bootstrap(clusters, hits[p])
            rows.append({"grouping": kind, "p": p, "n_demos": len(sel), "accuracy": a, "ci95_lo": lo, "ci95_hi": hi, "chance": chance})
        for p in DIFFS:
            a, lo, hi = S.cluster_bootstrap(clusters, hits[p] - hits[0])
            rows.append({"grouping": kind, "p": f"{p}-0", "n_demos": len(sel), "accuracy": a, "ci95_lo": lo, "ci95_hi": hi, "chance": ""})
        return hits

    scene_groups, fam_groups = groups(scene_of), groups(fam_of)
    sel_scene = [(t, d) for t, d in demos if t in scene_groups]
    sel_fam = [(t, d) for t, d in demos if t in fam_groups]
    block("within-scene", scene_groups, sel_scene)
    block("within-family", fam_groups, sel_fam)
    for a, b in PAIRS:
        sel = [(t, d) for t, d in demos if t in (a, b)]
        if not {a, b} <= {t for t, _ in sel}:
            continue
        hits = {p: np.array([masked_correct(scores[idx[(t, d, p)]], t, (a, b)) for t, d in sel], float) for p in ps}
        clusters = np.arange(len(sel))
        for p in ps:
            v, lo, hi = S.cluster_bootstrap(clusters, hits[p])
            pair_rows.append({"pair": f"{a}-{b}", "p": p, "n_demos": len(sel), "accuracy": v, "ci95_lo": lo, "ci95_hi": hi,
                              f"acc_task{a}": float(hits[p][[t == a for t, _ in sel]].mean()),
                              f"acc_task{b}": float(hits[p][[t == b for t, _ in sel]].mean())})
        for p in DIFFS:
            v, lo, hi = S.cluster_bootstrap(clusters, hits[p] - hits[0])
            pair_rows.append({"pair": f"{a}-{b}", "p": f"{p}-0", "n_demos": len(sel), "accuracy": v, "ci95_lo": lo, "ci95_hi": hi})
    for p in ps:
        m_all = [S.top2(scores[idx[(t, d, p)]])[1] for t, d in demos]
        m_sc10 = [S.top2(scores[idx[(t, d, p)]])[1] for t, d in sel_scene]
        m_sc = [masked_margin(scores[idx[(t, d, p)]], scene_groups[t]) for t, d in sel_scene]
        margin_rows.append({"p": p, "mean_margin_10way_all": float(np.mean(m_all)), "n_all": len(m_all),
                            "mean_margin_10way_within_scene_demos": float(np.mean(m_sc10)),
                            "mean_margin_among_scene_tasks": float(np.mean(m_sc)), "n_within_scene": len(sel_scene)})
    write_csv(out / "within_scene.csv", rows)
    write_csv(out / "pairs.csv", pair_rows)
    write_csv(out / "margins.csv", margin_rows)
    (out / "config.json").write_text(json.dumps({**meta, "scene_of_task": {str(t): s for t, s in sorted(scene_of.items())},
                                                 "family_of_task": {str(t): f for t, f in sorted(fam_of.items())},
                                                 "within_scene_tasks": sorted(scene_groups), "within_family_tasks": sorted(fam_groups),
                                                 "setting": "C2, T1, demos 0..29", "bootstrap": {"B": 10000, "seed": 0, "cluster": "demo"},
                                                 "rule": "none (post-hoc, descriptive)"}, indent=2))
    return {"rows": rows, "pairs": pair_rows, "margins": margin_rows}


def main() -> int:
    original = json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"]
    bddl = sorted(glob.glob(BDDL_GLOB))
    scene_of = scenes_from_bddl(bddl, original)
    _snap, _eps, demos, _others, _orig = S.load_meta()
    d = np.load(SCORES)
    keys = d["keys"]
    for t, v in demos.items():  # the saved keys must cover the demo table
        assert sum((keys[:, 0] == t) & (keys[:, 1] < S.N_QUERY) & (keys[:, 2] == 0)) == min(len(v), S.N_QUERY), t
    meta = {"git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "scores_file": str(SCORES), "scores_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),
            "bddl_files": [Path(p).name for p in bddl]}
    res = analyze(keys, d["C2_T1"], scene_of, OUT, meta)
    pct = lambda r: f"{100 * r['accuracy']:.1f} [{100 * r['ci95_lo']:.1f}; {100 * r['ci95_hi']:.1f}]"  # noqa: E731
    lines = ["# Stage 12 within-scene re-scoring (post-hoc, descriptive)", "",
             "Saved Stage 12 scores, C2, T1, demos 0..29. No new model run. No pass/fail rule.", "",
             "| grouping / pair | n | p=0 | p=10 | p=25 | p=50 | p=100 | p25−p0 | p50−p0 | chance |", "|---|---:|---|---|---|---|---|---|---|---:|"]
    for kind in ("within-scene", "within-family"):
        r = {x["p"]: x for x in res["rows"] if x["grouping"] == kind}
        lines.append(f"| {kind} | {r[0]['n_demos']} | " + " | ".join(pct(r[p]) for p in (0, 10, 25, 50, 100, "25-0", "50-0")) +
                     f" | {100 * r[0]['chance']:.1f} |")
    for a, b in PAIRS:
        r = {x["p"]: x for x in res["pairs"] if x["pair"] == f"{a}-{b}"}
        lines.append(f"| pair {a} vs {b} | {r[0]['n_demos']} | " + " | ".join(pct(r[p]) for p in (0, 10, 25, 50, 100, "25-0", "50-0")) + " | 50.0 |")
    lines += ["", "| p | mean 10-way margin (all) | 10-way margin (within-scene demos) | margin among scene tasks |", "|---:|---:|---:|---:|"]
    lines += [f"| {m['p']} | {m['mean_margin_10way_all']:.4f} | {m['mean_margin_10way_within_scene_demos']:.4f} | {m['mean_margin_among_scene_tasks']:.4f} |"
              for m in res["margins"]]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
