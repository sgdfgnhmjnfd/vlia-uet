#!/usr/bin/env python3
"""Synthetic checks for scripts/run_stage12_within_scene.py: candidate masking, ties, pairing by demo, BDDL scenes."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage12_within_scene as W  # noqa: E402


def main() -> int:
    s = np.array([0.9, 0.5, 0.6, 0.1])
    assert not W.masked_correct(s, 1, range(4)) and W.masked_correct(s, 1, (1, 3)), "masking to the scene"
    assert not W.masked_correct(np.array([0.5, 0.5, 0.9]), 0, (0, 1)), "a tie counts as wrong"
    assert np.isclose(W.masked_margin(s, (1, 2, 3)), 0.1)
    original = {"0": "put a in b", "1": "put c in b", "2": "open d"}
    paths = ["X/LIVING_ROOM_SCENE2_put_a_in_b.bddl", "X/LIVING_ROOM_SCENE2_put_c_in_b.bddl", "X/KITCHEN_SCENE10_open_d.bddl"]
    sc = W.scenes_from_bddl(paths, original)
    assert sc == {0: "LIVING_ROOM_SCENE2", 1: "LIVING_ROOM_SCENE2", 2: "KITCHEN_SCENE10"} and W.family(sc[2]) == "KITCHEN"
    assert W.groups(sc) == {0: [0, 1], 1: [0, 1]}
    # fake scores: 3 tasks x 4 demos x p in {0, 10, 25, 50, 100}. Task 0 vs 1 share a scene.
    # At p=0 the scene task is wrong for every demo of tasks 0/1 (score of the other scene task higher); at p>=25 right.
    keys, rows = [], []
    for t in range(3):
        for d in range(4):
            for p in (0, 10, 25, 50, 100):
                v = np.zeros(10)
                v[2] = 5.0  # the out-of-scene task always wins the 10-way vote
                other = 1 - t if t < 2 else 0
                if t < 2:
                    v[t], v[other] = (1.0, 2.0) if p < 25 else (2.0, 1.0)
                    if (t, d) == (0, 3) and p == 50:
                        v[t] = v[other]  # one tie at p=50
                keys.append((t, d, p))
                rows.append(v)
    keys, scores = np.array(keys), np.array(rows)
    perm = np.random.default_rng(0).permutation(len(keys))  # order of rows must not matter (pairing by demo)
    with tempfile.TemporaryDirectory() as tmp:
        res = W.analyze(keys[perm], scores[perm], {0: "LIVING_ROOM_SCENE2", 1: "LIVING_ROOM_SCENE2", 2: "STUDY_SCENE1"}, Path(tmp), {})
        r = {x["p"]: x for x in csv.DictReader((Path(tmp) / "within_scene.csv").open()) if x["grouping"] == "within-scene"}
        assert r["0"]["n_demos"] == "8" and float(r["0"]["accuracy"]) == 0.0 and float(r["25"]["accuracy"]) == 1.0
        assert np.isclose(float(r["50"]["accuracy"]), 7 / 8), "tie counted as wrong"
        assert float(r["25-0"]["accuracy"]) == 1.0 and np.isclose(float(r["50-0"]["accuracy"]), 7 / 8)
        assert np.isclose(float(r["0"]["chance"]), 0.5)
        pr = {(x["pair"], x["p"]): x for x in csv.DictReader((Path(tmp) / "pairs.csv").open())}
        assert float(pr[("0-1", "0")]["accuracy"]) == 0.0 and float(pr[("0-1", "25")]["accuracy"]) == 1.0
        m = {int(x["p"]): x for x in csv.DictReader((Path(tmp) / "margins.csv").open())}
        assert np.isclose(float(m[25]["mean_margin_among_scene_tasks"]), 1.0) and np.isclose(float(m[0]["mean_margin_10way_all"]), (8 * 3 + 4 * 5) / 12)
    print("test_stage12_within_scene passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
