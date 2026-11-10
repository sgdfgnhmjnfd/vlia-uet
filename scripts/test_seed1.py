#!/usr/bin/env python3
"""Synthetic checks for scripts/analyze_seed1.py: pairing by (task, episode) across seeds, pooled bootstrap."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_seed1 as A  # noqa: E402


def main() -> int:
    # pooled: identical seeds -> plain rate; both seeds of a unit stay together (CI from unit means)
    s0 = np.r_[np.ones(10), np.zeros(10)]
    p, lo, hi = A.pooled(s0, s0.copy())
    assert p == 0.5 and lo < 0.5 < hi
    p, lo, hi = A.pooled(np.ones(20), np.zeros(20))
    assert p == 0.5 and lo == hi == 0.5, "unit means are all 0.5 -> zero-width CI"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d0, d1 = tmp / "s0", tmp / "s1"
        d0.mkdir(), d1.mkdir()
        tokens = {f"{t},{e}": f"orig {(t + (e % 2)) % 10}" for t in range(10) for e in range(30) if (t, e) != (8, 29)}
        (tmp / "tok.json").write_text(json.dumps({"tokens": tokens}))
        rng = np.random.default_rng(0)
        order = [(c, t, e) for c in A.CONDS for t in range(10) for e in range(30)]
        for d, seed_ok in ((d0, lambda c, t, e: c == "I-only" and t < 2), (d1, lambda c, t, e: c != "I-wrong" and t < 3 and e % 2 == 0)):
            for i in rng.permutation(len(order)):  # write in random order: pairing must not depend on file order
                c, t, e = order[i]
                if c == "E2E" and (t, e) == (8, 29):
                    continue
                rec = {"condition": c, "task": t, "episode": e, "seed": 1000 + e, "init_state_id": e, "success": bool(seed_ok(c, t, e)),
                       "seconds": 9.0, "z_int_task": tokens.get(f"{t},{e}") if c == "E2E" else f"orig {t}"}
                (d / f"{c}_t{t}_e{e}.json").write_text(json.dumps(rec))
        # an unrelated condition is ignored
        (d1 / "I-goal_t0_e0.json").write_text(json.dumps({"condition": "I-goal", "task": 0, "episode": 0, "seed": 1000,
                                                          "init_state_id": 0, "success": True, "seconds": 9.0}))
        v = A.analyze(d0, d1, tmp / "tok.json", tmp / "res", {})
        summ = {(r["condition"], r["seed"]): r for r in csv.DictReader((tmp / "res/summary.csv").open())}
        assert summ[("I-only", "seed 0")]["successes"] == "60" and summ[("I-only", "seed 1")]["successes"] == "45"
        assert summ[("E2E", "seed 1")]["n"] == "299" and summ[("E2E", "seed 1")]["successes"] == "45"
        assert summ[("I-only", "pooled")]["successes"] == "105" and np.isclose(float(summ[("I-only", "pooled")]["rate"]), 105 / 600)
        pr = {r["comparison"]: float(r["delta"]) for r in csv.DictReader((tmp / "res/paired.csv").open())}
        assert np.isclose(pr["I-only: seed 1 - seed 0"], (45 - 60) / 300)
        assert np.isclose(pr["seed-1 E2E - seed-1 I-wrong"], 45 / 299)
        assert v["S1"]["pass"] and v["S2"]["pass"]
        # a seed mismatch inside a unit must stop the analysis
        r = json.loads((d1 / "I-only_t0_e0.json").read_text())
        r["seed"] = 1001
        (d1 / "I-only_t0_e0.json").write_text(json.dumps(r))
        try:
            A.analyze(d0, d1, tmp / "tok.json", tmp / "res", {})
            raise AssertionError("seed mismatch not caught")
        except AssertionError as exc:
            assert "I-only" in str(exc)
    print("test_seed1 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
