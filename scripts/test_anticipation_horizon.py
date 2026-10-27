#!/usr/bin/env python3
"""Unit tests for scripts/run_anticipation_horizon.py on SYNTHETIC data only.

Checks: the timeline follows step order; the same anchors are used at every
horizon; the target is the label of the step containing t+H; the 5-way
candidates hold the target once plus 4 distinct labels of the same video; C2
only uses frames strictly before the context window; a query built from the
target step's frames (oracle) is perfect when each step has its own direction;
the video bootstrap brackets the mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as A  # noqa: E402

D, PER, STEPS = 32, 8, 12


def synth():
    rng = np.random.default_rng(0)
    rows, frames, texts = [], [], {}
    for v in range(3):
        order = list(rng.permutation(STEPS) + 1)  # shuffled step ids in the manifest
        for s in order:
            label = f"act {v}-{s}"
            direction = rng.normal(size=D)
            texts[label] = direction
            rows.append({"sample_id": f"v{v}/step{s}", "video_uid": f"u{v}", "step_id": str(s),
                         "what": label, "why": label})
            frames.append(np.tile(direction, (PER, 1)) + 0.01 * rng.normal(size=(PER, D)))
    return rows, A.normalize(np.asarray(frames, dtype=np.float32)), texts


def main() -> int:
    rows, frames, texts = synth()
    tl = A.build_timelines(rows, frames)
    for uid, t in tl.items():
        steps = [int(rows[i]["step_id"]) for i in t["rows"]]
        assert steps == sorted(steps), "timeline not in step order"
        assert len(t["frames"]) == STEPS * PER
    horizons = (2, 8, 16)
    queries, dropped = A.make_queries(tl, 8, horizons, rows, "what")
    assert not dropped
    anchors = {h: sorted((q["video_uid"], q["t"]) for q in queries if q["H"] == h) for h in horizons}
    assert anchors[2] == anchors[8] == anchors[16], "anchors differ across H"
    for q in queries:
        t = tl[q["video_uid"]]
        assert q["target_step"] == t["step_of"][q["t"] + q["H"]]
        assert q["target"] == rows[t["rows"][q["target_step"]]]["what"]
        c = q["cands5"]
        assert c[0] == q["target"] and len(set(c)) == 5 and c.count(q["target"]) == 1
        vid_labels = {rows[i]["what"] for i in t["rows"]}
        assert set(c) <= vid_labels
        assert q["same_step"] == (q["anchor_step"] == q["target_step"])
    # C2 uses only frames before the window: changing later frames must not change it
    names = sorted(texts)
    emb = A.normalize(np.asarray([texts[n] for n in names], dtype=np.float32))
    index = {n: i for i, n in enumerate(names)}
    mu = frames.reshape(-1, D).mean(axis=0)
    q1, o1 = A.query_vectors(tl, queries, 8, "C2", mu)
    tl2 = {k: {**v, "frames": v["frames"].copy()} for k, v in tl.items()}
    for v in tl2.values():
        v["frames"][-1] += 5.0
    probe = [i for i, q in enumerate(queries) if q["t"] < len(tl[q["video_uid"]]["frames"]) - 1]
    q2, _ = A.query_vectors(tl2, queries, 8, "C2", mu)
    assert np.allclose(q1[probe], q2[probe]), "C2 used a future frame"
    m = A.score_queries(A.query_vectors(tl, queries, 8, "C0", mu)[1], queries, index, emb)
    assert np.nanmean(m["hit5"]) == 1.0, "oracle should be perfect on separable synthetic data"
    vids = np.array([q["video_uid"] for q in queries])
    mean, lo, hi = A.video_bootstrap(m["rr"], vids, 200)
    assert lo <= mean <= hi
    assert A.selective_accuracy(np.array([1, 0, 1, 0.]), np.array([4, 3, 2, 1.]), 0.5) == 0.5
    print("all anticipation-horizon tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
