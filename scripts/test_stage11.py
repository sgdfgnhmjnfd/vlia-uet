#!/usr/bin/env python3
"""Synthetic checks for Stage 11 (scripts/libero_goal_ablation.py): phrasing lookup and --analyze-stage11
on fake episode JSONs with known successes (no robot, no model)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402


def main() -> int:
    original = {t: f"orig {t}" for t in range(10)}
    ph = {"goal": {str(t): f"goal {t}" for t in range(10)}, "para": {str(t): f"para {t}" for t in range(10)}}
    real = json.loads((ROOT / "results/stage11_libero_goal/phrasings.json").read_text())
    assert sorted(real) == ["goal", "para"] and all(sorted(real[k], key=int) == [str(t) for t in range(10)] for k in real)
    assert G.intention_text("I-only", 3, original, ph) == "orig 3"
    assert G.intention_text("I-wrong", 3, original, ph) == "orig 8"
    assert G.intention_text("I-goal", 3, original, ph) == "goal 3"
    assert G.intention_text("I-para", 9, original, ph) == "para 9"
    assert G.intention_text("I-goal", 0, original, real) == "I want the alphabet soup and the tomato sauce to end up in the basket"
    assert set(G.STAGE11_VLIA) <= set(G.INTENTION_CONDITIONS)

    # fake episodes, e = 0..29; e < 10 reproduce the R2 counts
    def success(c, t, e):
        if e < 10:
            return {"I-only": t < 2, "I-wrong": False, "correct": t == 0 or (t == 1 and e < 9), "empty": False,
                    "I-goal": t == 0, "I-para": False}[c]
        return {"I-only": t < 3, "I-wrong": False, "I-goal": t < 3, "I-para": t < 1, "correct": t < 3,
                "empty": False}[c]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        vd, bd, res = tmp / "vlia", tmp / "base", tmp / "res"
        (vd / "episodes").mkdir(parents=True)
        bd.mkdir()
        (vd / "instructions.json").write_text(json.dumps({"original": {str(t): v for t, v in original.items()}}))
        (tmp / "ph.json").write_text(json.dumps(ph))
        texts = list(original.values()) + list(ph["goal"].values()) + list(ph["para"].values())
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(len(texts), 16))
        emb[10:20] = emb[:10] + 0.1 * rng.normal(size=(10, 16))  # goal close to its own task
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.savez(tmp / "emb.npz", tasks=np.asarray(texts), emb=emb.astype(np.float32))
        for t in range(10):
            for e in range(30):
                for c in G.STAGE11_VLIA + G.STAGE11_BASE + ("wrong",):
                    if c == "wrong" and e >= 10:
                        continue
                    rec = {"condition": c, "task": t, "episode": e, "seed": 1000 + e, "init_state_id": e,
                           "success": bool(success(c, t, e)) if c != "wrong" else False, "seconds": 9.0,
                           "z_int_task": G.intention_text(c, t, original, ph) if c.startswith("I-") else None}
                    (vd / "episodes" if c.startswith("I-") else bd).joinpath(f"{c}_t{t}_e{e}.json").write_text(json.dumps(rec))
        # an extra unpaired episode must be ignored
        (vd / "episodes" / "I-goal_t0_e30.json").write_text(json.dumps(
            {"condition": "I-goal", "task": 0, "episode": 30, "seed": 1030, "init_state_id": 30, "success": True,
             "seconds": 9.0, "z_int_task": "goal 0"}))
        own = argparse.Namespace(out=str(vd), v1_episodes=str(bd), results=str(res), phrasings=str(tmp / "ph.json"),
                                 task_embeddings=str(tmp / "emb.npz"), vlia_checkpoint="v", baseline_checkpoint="b",
                                 git_commit="x")
        assert G.analyze_stage11(own) == 0
        summ = {(r["episodes"], r["condition"]): r for r in csv.DictReader((res / "summary.csv").open())}
        assert summ[("e0-9", "I-only")]["successes"] == "20" and summ[("e0-9", "correct")]["successes"] == "19"
        # e0-29: I-only 20 + 60 = 80; I-goal 10 + 60 = 70; I-para 0 + 20 = 20; correct 19 + 60 = 79
        assert [summ[("e0-29", c)]["successes"] for c in ("I-only", "I-goal", "I-para", "correct", "I-wrong")] == \
            ["80", "70", "20", "79", "0"] and summ[("e0-29", "I-only")]["n"] == "300"
        pr = {(r["episodes"], r["comparison"]): float(r["delta"]) for r in csv.DictReader((res / "paired.csv").open())}
        assert np.isclose(pr[("e0-29", "I-goal - I-wrong")], 70 / 300)
        assert np.isclose(pr[("e0-29", "I-goal - I-only")], -10 / 300)
        assert np.isclose(pr[("e0-29", "I-only - correct")], 1 / 300)
        cfg = json.loads((res / "config.json").read_text())
        assert cfg["verdicts"]["G1"]["pass"] and cfg["verdicts"]["G2"]["pass"] and cfg["gate_R2"]["pass"]
        diag = list(csv.DictReader((res / "text_diagnostic.csv").open()))
        assert all(r["rank_of_own_task"] == "1" for r in diag if r["phrasing"] == "goal")
        assert all(int(r["rank_of_own_task"]) >= 1 and -1 <= float(r["cos_to_own_original"]) <= 1 for r in diag)
        # R2 must stop the analysis when e = 0..9 does not reproduce Stage 4
        (vd / "episodes" / "I-only_t5_e0.json").write_text(json.dumps(
            {"condition": "I-only", "task": 5, "episode": 0, "seed": 1000, "init_state_id": 0, "success": True,
             "seconds": 9.0, "z_int_task": "orig 5"}))
        try:
            G.analyze_stage11(own)
            raise AssertionError("R2 did not stop")
        except SystemExit as exc:
            assert "R2" in str(exc)
    print("test_stage11 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
