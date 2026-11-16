#!/usr/bin/env python3
"""Checks for Stage 14b Part B (W): folds hold no scene / demo on both sides, unseen frames are left out, the locked
test split is never opened, W never receives a candidate list, and the text scores on hand examples."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage14b_writer as WR  # noqa: E402

# the first 20 development WHY sentences by sample id, hand-labelled (verb, object)
HAND = [("align reference for painting", "align", "reference"), ("refining paint texture", "refining", "texture"),
        ("remove excess paint", "remove", "paint"), ("verify paint state before proceeding", "verify", "state"),
        ("blend paint", "blend", "paint"), ("assessing painting progress", "assessing", "progress"),
        ("checking reference details", "checking", "details"), ("checking tape adhesion", "checking", "adhesion"),
        ("check paint dryness or texture", "check", "dryness"), ("stabilize artwork", "stabilize", "artwork"),
        ("check paint dryness", "check", "dryness"), ("positioning reference materials", "positioning", "materials"),
        ("blend paint", "blend", "paint"), ("evaluate painting status", "evaluate", "status"),
        ("prepare to mask painting edges", "prepare", "edges"), ("preparing tape strip", "preparing", "strip"),
        ("preparing tape for masking the artwork", "preparing", "tape"), ("prepare tape strip for masking", "prepare", "strip"),
        ("finish masking tape application", "finish", "application"), ("prepare to secure tape strip", "prepare", "strip")]


def test_text_scores():
    for s, v, o in HAND:
        assert WR.verb_object(s) == (v, o), (s, WR.verb_object(s))
    assert WR.token_f1("open the door", "open door") == 0.8
    assert WR.token_f1("x y", "a b") == 0.0 and WR.token_f1("Open, door!", "open door") == 1.0
    # ROUGE-L: LCS("the cat sat on the mat", "the cat on the mat") = 5
    p, r = 5 / 6, 5 / 5
    assert np.isclose(WR.rouge_l("the cat sat on the mat", "the cat on the mat"), 2 * p * r / (p + r))
    assert WR.rouge_l("a b", "c d") == 0.0 and WR.lcs(list("abcbdab"), list("bdcaba")) == 4


def test_ego_folds_and_locked_test():
    data = WR.ego_data()
    assert len(data["sids"]) == 2503 and len(set(data["vids"])) == 27
    man = {l.split('"sample_id": "')[1].split('"')[0] for l in WR.MANIFEST.open() if l.strip()}
    assert set(data["sids"]) <= man
    for f in range(WR.N_FOLDS_EGO):
        te, tr = data["fold"] == f, data["fold"] != f
        assert not set(data["scenes"][te]) & set(data["scenes"][tr])
    src = Path(WR.__file__).read_text()
    assert "test.jsonl" not in src and "egointent_full_split_v1" not in src
    assert "metadata.jsonl" not in src


def test_prefix_and_no_candidates():
    data = {"feats": np.arange(2 * 8 * 3, dtype=float).reshape(2, 8, 3), "texts": {t: ["a", "b"] for t in WR.EGO_TAGS}}
    ex = WR.ego_examples(data, [0, 1], (2,))
    assert all(v.shape[0] == 2 for v, _, _ in ex) and np.array_equal(ex[0][0], data["feats"][0, :2])
    rng = np.random.default_rng(0)
    ex = WR.ego_examples(data, [0, 1], WR.EGO_K, rng)
    for v, _, _ in ex:
        k = v.shape[0]
        assert k in WR.EGO_K and np.array_equal(v, data["feats"][0 if v[0, 0] == 0 else 1, :k])
    for fn in (WR.W.generate, WR.cmd_write_ego, WR.lib_write, WR.cmd_write_libero, WR.cmd_unseen):
        code = inspect.getsource(fn)
        assert "cand" not in code and "build_kway" not in code, fn.__name__
    assert list(inspect.signature(WR.W.generate).parameters) == ["self", "vis_list", "tag", "bs"]


def test_libero_folds():
    L = WR.lib_data()
    assert len(L["label"]) == 379
    for f in range(5):
        te = {k for k, g in L["outer"].items() if g == f}
        tr = {k for k, g in L["outer"].items() if g != f}
        assert te and not te & tr
    assert all(WR.obj_place_ok(L["orig"][t], t) for t in range(10))
    assert all(WR.obj_place_ok(s, t) for t in range(10) for s in WR.TRAIN_SENTENCES[t] if t != 4)


def main() -> int:
    assert WR.cmd_selftest(None) == 0
    test_text_scores()
    test_prefix_and_no_candidates()
    test_ego_folds_and_locked_test()
    test_libero_folds()
    print("test_stage14b_writer passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
