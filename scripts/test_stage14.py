#!/usr/bin/env python3
"""Synthetic checks for Stage 14 part A beyond `run_stage14_generate.py selftest`: the E2E-gen conditions of
libero_goal_ablation.py (free-sentence token path on a fake table), the LIBERO text-match analysis with GA4-GA6,
the robot tables with p = 50 reuse, and the human-check sampling, on fake files with a fake text encoder."""
from __future__ import annotations

import json
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import run_stage14_generate as S  # noqa: E402


class FakeEncoder:
    """Bag-of-words hashing into 64 dims; the template adds a fixed offset (only to check it is applied)."""

    def __call__(self, texts, template: bool):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in (S.TEMPLATE.format(why=t) if template else t).lower().replace(".", "").split():
                v[zlib.crc32(w.encode()) % 64] += 1
            out.append(v / max(np.linalg.norm(v), 1e-12))
        return np.array(out)


def test_conditions():
    original = {t: f"orig {t}" for t in range(10)}
    tok = {"E2E-gen": {"3,7": "I want the soup in the basket."}, "E2E-gen-p50": {"3,7": "I want x."}}
    assert G.intention_text("E2E-gen", 3, original, None, 7, tok) == "I want the soup in the basket."
    assert G.intention_text("E2E-gen-p50", 3, original, None, 7, tok) == "I want x."
    assert G.intention_text("I-only", 3, original, None) == "orig 3" and G.intention_text("E2E", 3, original, None, 7, {"3,7": "orig 1"}) == "orig 1"
    assert set(G.GEN_CONDITIONS) <= set(G.INTENTION_CONDITIONS)
    assert G.reused("E2E-gen-p50", 3, 7, {"E2E-gen-p50": ["3,7"]}) and not G.reused("E2E-gen", 3, 7, {"E2E-gen-p50": ["3,7"]})
    src = (ROOT / "scripts/libero_goal_ablation.py").read_text()
    assert 'ep_dir / f"{c}_t{t}_e{e}.json"' in src and "not own.probe and reused(c, t, e, e2e_reuse)" in src
    # the free sentence is looked up by its own text in the table built by `plan`, never as another task's row
    table = {s: i for i, s in enumerate(sorted(set(original.values()) | {"I want the soup in the basket."}))}
    assert table[G.intention_text("E2E-gen", 3, original, None, 7, tok)] != table["orig 3"]


def test_libero_and_robot():
    enc = FakeEncoder()
    orig = S.originals()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        S.WORK, S.RES, S.EVAL = tmp / "work", tmp / "res", tmp / "eval"
        (S.WORK / "gen_libero").mkdir(parents=True)
        (S.EVAL / "episodes").mkdir(parents=True)
        units = S.libero_units()
        assert len(units) == 299 and (8, 29) not in units

        def sentence(t, d, p):
            if p == 0:
                return "I want things tidy."
            if p in (10, 25) and d % 3 == 0:
                return orig[(t + 1) % 10] + "."  # wrong early on a third of demos
            return orig[t] + "."

        for t, d in units:
            for p in S.LIB_P:
                S.write_json(S.lib_path(t, d, p), {"task": t, "demo": d, "p": p, "sentence": sentence(t, d, p)})
        S.write_json(S.blank_path("libero"), {"k": 8, "sentence": "I want the table clean."})
        assert S.cmd_human_check(None) == 0
        import csv

        hc = list(csv.DictReader((S.RES / "human_check_libero.csv").open()))
        assert len(hc) == 60 and len({(r["task"], r["demo"], r["p"]) for r in hc}) == 60 and all(r["rating"] == "" for r in hc)
        assert "match" not in hc[0] and "nearest" not in " ".join(hc[0])
        rows, lib, embs = S.analyze_libero(enc, orig)
        acc = {r["p"]: r for r in lib["acc"]}
        n_wrong = sum(1 for t, d in units if d % 3 == 0)
        assert acc[100]["open_ended_10way"] == 1.0 and np.isclose(acc[25]["open_ended_10way"], 1 - n_wrong / 299)
        assert acc[0]["n_distinct_sentences"] == 1
        v = lib["verdicts"]
        assert v["GA4"]["pass"] and v["GA5"]["pass"] and v["GA6"]["pass"] and np.isclose(v["GA6"]["delta"], n_wrong / 299)
        assert acc[100]["within_scene_0v1"] == 1.0
        # plan + robot tables with p = 50 reuse
        sent = {(p, t, d): lib["sents"][(t, d, p)] for p in (25, 50) for t, d in units}
        plan = S.plan_from(sent, "I want the table clean.", units)
        assert len(plan["new"]["E2E-gen-p50"]) == n_wrong
        S.RES.mkdir(exist_ok=True)
        (S.RES / "e2e_gen_tokens.json").write_text(json.dumps(plan))

        def ep(c, t, e, z, ok):
            (S.EVAL / "episodes" / f"{c}_t{t}_e{e}.json").write_text(json.dumps(
                {"condition": c, "task": t, "episode": e, "seed": 1000 + e, "init_state_id": e, "z_int_task": z, "success": ok, "seconds": 9.0}))

        for t, e in units:
            ep("I-only", t, e, orig[t], e < 20)
            ep("I-wrong", t, e, orig[(t + 5) % 10], False)
            ep("E2E", t, e, orig[t], e < 10)
            ep("E2E-gen", t, e, plan["tokens"]["E2E-gen"][f"{t},{e}"], e < 15 and e % 3 != 0)
            ep("E2E-gen-blank", t, e, "I want the table clean.", False)
        lib["cos_true"] = {(r["task"], r["demo"], r["p"]): r["cos_true"] for r in rows}
        lib["E_desc_cos"] = {(t, d): embs["desc"] @ embs["E"][lib["sents"][(t, d, 25)]] for t, d in units}
        r = S.analyze_robot(lib, orig)
        assert "E2E-gen-p50" not in r["episodes"], "p50 must wait for its new rollouts"
        for k in plan["new"]["E2E-gen-p50"]:
            t, e = map(int, k.split(","))
            ep("E2E-gen-p50", t, e, plan["tokens"]["E2E-gen-p50"][k], True)
        r = S.analyze_robot(lib, orig)
        row = {x["condition"]: x for x in r["summary"]}
        expect_gen = sum(1 for t, e in units if e < 15 and e % 3 != 0)
        assert row["E2E-gen"]["successes"] == expect_gen and row["E2E-gen-blank"]["successes"] == 0
        assert row["E2E-gen-p50"]["successes"] == expect_gen + n_wrong and r["episodes"]["E2E-gen-p50"] == {"reused": 299 - n_wrong, "new": n_wrong}
        ga3 = next(x for x in r["paired"] if x["rule"] == "GA3")
        assert np.isclose(ga3["delta"], expect_gen / 299) and ga3["ci95_lo"] > 0
        assert np.isclose(r["text_match_p25"], 1 - n_wrong / 299) and r["by_text_match"][0.0]["E2E_gen_successes"] == 0
        assert sum(x["n"] for x in r["by_cos_tertile"]) == 299
        assert r["snap"].startswith("pending") or r["snap"] == "done"
        # a wrong z_int_task in an episode file is caught
        ep("E2E-gen", 0, 1, "something else", True)
        try:
            S.analyze_robot(lib, orig)
            raise AssertionError("token mismatch not caught")
        except AssertionError as exc:
            assert "E2E-gen" in str(exc)


def main() -> int:
    assert S.cmd_selftest(None) == 0
    test_conditions()
    test_libero_and_robot()
    print("test_stage14 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
