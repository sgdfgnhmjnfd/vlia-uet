#!/usr/bin/env python3
"""Checks for Stage 14b: the split is disjoint, the scene object lists are complete and identical within a scene,
the selection rule, and the snap file format (synthetic)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage14b_variants as V  # noqa: E402

WORDS = {v: k for k, v in V.NUM.items()}


def count_listed(names: str) -> int:
    n = 0
    for part in names.split(", "):
        first = part.split()[0]
        n += WORDS.get(first, 1)
    return n


def test_split():
    cal, test = V.cal_units(), V.test_units()
    assert len(cal) == 80 and len(test) == 299
    assert not set(cal) & set(test)
    assert all(d >= 30 for _, d in cal) and all(d < 30 for _, d in test)


def test_objects():
    orig = V.S.originals()
    scenes = {t: V.scene_of(t, orig) for t in range(V.N_TASKS)}
    for t in range(V.N_TASKS):
        inst = V.bddl_instances(V.bddl_file(orig[t]).read_text())
        names = scenes[t][1]
        assert count_listed(names) == len(inst), (t, names, inst)  # every declared object and fixture is listed
        for i in inst:
            base = re.sub(r"_\d+$", "", i).replace("_", " ")
            assert base.split()[-1] in names or V.plural(base.split()[-1]) in names, (t, i, names)
        assert not re.search(r"_|\d+$", names.replace("chefmate 8", "")), names
    by_scene = {}
    for t, (sc, names) in scenes.items():
        by_scene.setdefault(sc, set()).add(names)
    assert all(len(v) == 1 for v in by_scene.values()), by_scene
    assert scenes[0] == scenes[1] and scenes[0][0] == "LIVING_ROOM_SCENE2"
    assert "two moka pots" in scenes[8][1] and "two plates" in scenes[4][1]
    assert len(by_scene) == 9


def test_selection():
    assert V.select({"G0": {25: .30, 50: .40}, "G1": {25: .40, 50: .30}, "G2": {25: .1, 50: .1}})[0] == "G0"
    assert V.select({"G0": {25: .30, 50: .30}, "G1": {25: .50, 50: .30}, "G2": {25: .45, 50: .35}})[0] == "G1"
    assert V.select({"G0": {25: .10, 50: .10}, "G1": {25: .20, 50: .30}, "G2": {25: .30, 50: .30}})[0] == "G2"


def test_snap():
    orig = {t: f"desc {t}" for t in range(10)}
    units = [(0, 0), (1, 3), (2, 5)]
    near, cos = {}, {}
    for t, d in units:
        for p in V.LIB_P:
            near[(t, d, p)] = t if d != 3 else -1
            c = np.zeros(10)
            c[[1, 4]] = 0.5
            cos[(t, d, p)] = c
    s = V.snap_from(near, cos, units, "G1", orig)
    assert set(s) == {"tokens", "pred", "reuse", "new"}
    c = "E2E-snap-G1-p25"
    assert s["tokens"][c]["0,0"] == "desc 0" and s["pred"][c]["1,3"] == 1 and s["tokens"][c]["1,3"] == "desc 1"
    assert s["reuse"][c] == ["0,0", "1,3", "2,5"] and s["new"][c] == []
    assert len(s["tokens"]) == len(V.LIB_P)
    a = V.agreement({(0, 0, 25): 0, (0, 0, 50): 0, (1, 3, 25): 2, (1, 3, 50): 1}, [(0, 0), (1, 3)], None)
    assert a == {"n_agree": 1, "n_disagree": 1, "acc_p50_agree": 1.0, "acc_p50_disagree": 1.0}


def main() -> int:
    assert V.cmd_selftest(None) == 0
    test_split()
    test_objects()
    test_selection()
    test_snap()
    print("test_stage14b passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
