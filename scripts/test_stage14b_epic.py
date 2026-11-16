#!/usr/bin/env python3
"""Checks for Stage 14b Part D: participant-disjoint folds, anticipation windows end before the segment, verb and noun
scoring on hand examples, and no video, frame or feature file staged for commit."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage14b_epic as E  # noqa: E402


def test_scoring():
    verbs, nouns = E.load_classes()
    hand = [("open door", 3, 3, True, True), ("close the cupboard", 4, 3, True, True), ("take plate", 0, 2, True, True),
            ("grab spoons", 0, 1, True, True), ("put knife", 1, 4, True, True), ("wash hands", 2, 3, True, False),
            ("turn on tap", 0, 0, False, True), ("pick-up spoon", 0, 1, True, True), ("", 0, 0, False, False)]
    for text, v, n, ev, en in hand:
        assert E.verb_ok(text, verbs[v]) == ev, (text, v)
        assert E.noun_ok(text, nouns[n]) == en, (text, n)
    assert E.noun_ok("rinse water tap", nouns[0])  # instance tap:water as the pair "water tap"


def test_windows():
    rng = np.random.default_rng(0)
    for _ in range(200):
        st = rng.uniform(0, 30)
        sp = st + rng.uniform(0.2, 10)
        fps = rng.choice([50.0, 59.94])
        for c in E.CONDS:
            t = E.cond_times(st, sp, fps, c)
            if c.startswith("ahead"):
                g = float(c.split("_g")[1])
                if st - g - 2 < 0:
                    assert t is None
                else:
                    assert t[-1] < st and t[-1] <= st - g and t[0] >= 0 and len(t) == 8
            else:
                assert np.isclose(t[0], st) and t[-1] <= sp + 1e-9 and len(t) == 8


def test_folds():
    if not (E.WORK / "units.json").exists():
        return
    U = E.load_units()
    for f in E.folds(U):
        te = {u["participant"] for u in U["units"] if u["fold"] == f}
        tr = {u["participant"] for u in U["units"] if u["fold"] != f}
        assert te and not te & tr


def test_nothing_staged():
    staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    bad = [p for p in staged + tracked if p.lower().endswith((".mp4", ".jpg", ".png", ".npz", ".npy", ".pt", ".part")) and "epic" in p.lower()]
    assert not bad, bad


def main() -> int:
    assert E.cmd_selftest(None) == 0
    test_scoring()
    test_windows()
    test_folds()
    test_nothing_staged()
    print("test_stage14b_epic passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
