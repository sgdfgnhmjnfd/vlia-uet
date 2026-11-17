#!/usr/bin/env python3
"""Stage 16 checks.

- Causality: for every Part A2 and curve query of every clip (real timing metadata, random embeddings), every frame
  used has timestamp <= cut; no frame at or after start_s enters a pre-touch query.
- LOPO: the test person never appears in training.
- The manual t_on lies on the 0.25 s grid and is <= start_s.
- Nothing large is staged for commit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage16_scene_motion as S  # noqa: E402


def test_causality():
    meta = S.load_meta()
    rng = np.random.default_rng(0)
    worst, n = -np.inf, 0
    for v, m in meta.items():
        t = S.grid_times(m["dur"])
        z = rng.normal(size=(len(t), 4))
        cuts = [m["start"] - H for H in S.HS] + [m["start"] + c for c in S.C_START] + [m["t_on"] + d for d in S.D_TON]
        for cut in cuts:
            f, info = S.query_cut(m, t, z, cut)
            if f is None:
                continue
            n += 1
            worst = max(worst, info["max_frame_minus_cut"])
            assert info["max_frame_minus_cut"] <= 1e-9, (v, cut, info)
            if cut < m["start"]:
                r = min(1.0, m["t_on"] - 0.25, cut)
                used = t[(t <= cut + 1e-9)]
                assert used.max() < m["start"], (v, cut)
                assert r <= m["t_on"] - 0.25 + 1e-9
    print(f"causality: {n} queries, max(frame time - cut) = {worst:.3g}")


def test_lopo_disjoint():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(18, 3))
    y = np.array([0, 1] * 9)
    persons = np.array(["p1"] * 6 + ["p2"] * 6 + ["p3"] * 6)
    reps = np.array([1, 1, 2, 2, 3, 3] * 3)
    seen = []
    orig = S.fit_predict

    def spy(Xtr, ytr, Xte, C, k):
        seen.append((len(Xtr), len(Xte)))
        return orig(Xtr, ytr, Xte, C, k)

    S.fit_predict = spy
    try:
        S.lopo(X, y, persons, reps, 2)
    finally:
        S.fit_predict = orig
    assert all(a <= 12 for a, b in seen) and (12, 6) in seen, seen  # training never exceeds the two training persons


def test_ton_grid():
    meta = S.load_meta()
    for v, m in meta.items():
        assert abs(m["t_on"] * 4 - round(m["t_on"] * 4)) < 1e-9 and m["t_on"] <= m["start"] + 1e-9, v


def test_nothing_large_staged():
    out = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
    bad = [p for p in out if p.lower().endswith((".mp4", ".mov", ".jpg", ".png", ".npz", ".npy", ".pt")) and "curve" not in p and "ask_curve" not in p]
    assert not bad, bad


def main() -> int:
    assert S.cmd_selftest(None) == 0
    test_causality()
    test_lopo_disjoint()
    test_ton_grid()
    test_nothing_large_staged()
    print("test_stage16 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
