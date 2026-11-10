#!/usr/bin/env python3
"""Synthetic checks for the token matrix: I-tok naming and token lookup, matrix assembly from I-only / I-wrong /
I-tok files, missing-entry detection, and `lookup` against a hand-made answer."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import token_matrix as TM  # noqa: E402


def main() -> int:
    original = {t: f"orig {t}" for t in range(10)}
    # I-tok naming and token lookup
    conds = G.expand_conditions(["I-tok"], 2, G.parse_tasks("0-9"))
    assert conds == ["I-tok0", "I-tok1", "I-tok3", "I-tok4", "I-tok5", "I-tok6", "I-tok8", "I-tok9"], conds
    assert G.expand_conditions(["I-only", "I-tok"], 0, [1, 5]) == ["I-only", "I-tok1"]
    assert G.intention_text("I-tok8", 2, original, None) == "orig 8" and G.intention_text("I-only", 2, original, None) == "orig 2"
    assert "I-tok" in G.INTENTION_CONDITIONS and G.parse_tasks("1,3-4") == [1, 3, 4]
    assert TM.entry_file(2, 2, 5) == "I-only_t2_e5.json" and TM.entry_file(2, 7, 5) == "I-wrong_t2_e5.json"
    assert TM.entry_file(2, 8, 5) == "I-tok8_t2_e5.json"
    # fake episodes: success iff k == t (from I-only) or (t, k) == (0, 1) and e even, or k == wrong(t) and e == 0
    truth = np.zeros((10, 10, 30))
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ep"
        d.mkdir()
        for t in range(10):
            for k in range(10):
                for e in range(30):
                    ok = k == t or ((t, k) == (0, 1) and e % 2 == 0) or (k == G.wrong_task(t) and e == 0)
                    truth[t, k, e] = ok
                    c = "I-only" if k == t else "I-wrong" if k == G.wrong_task(t) else f"I-tok{k}"
                    rec = {"condition": c, "task": t, "episode": e, "seed": 1000 + e, "init_state_id": e, "instruction": "",
                           "z_int_task": original[k], "success": bool(ok), "seconds": 9.0}
                    if c.startswith("I-tok"):
                        rec["token_task"] = k
                    (d / f"{c}_t{t}_e{e}.json").write_text(json.dumps(rec))
        m, secs, missing = TM.build(d, original)
        assert not missing and np.array_equal(m, truth) and len(secs) == 10 * 8 * 30
        assert m[3, 3].sum() == 30 and m[3, 8, 0] == 1 and m[3, 8, 1:].sum() == 0, "k = t from I-only, k = t+5 from I-wrong"
        # lookup against a hand-made answer
        tokens = {"0,0": "orig 1", "0,1": "orig 1", "4,7": "orig 4", "4,0": "orig 9", "6,3": "orig 2"}
        assert TM.lookup(m, tokens, original) == {"0,0": 1.0, "0,1": 0.0, "4,7": 1.0, "4,0": 1.0, "6,3": 0.0}
        # analysis numbers on the fake matrix (Stage 12 tokens replaced by a fake plan)
        (Path(tmp) / "tok.json").write_text(json.dumps({"tokens": {"0,0": "orig 1", "0,1": "orig 1", "2,4": "orig 7"}}))
        (d / "E2E_t0_e0.json").write_text(json.dumps({"success": True, "z_int_task": "orig 1"}))
        (d / "E2E_t0_e1.json").write_text(json.dumps({"success": True, "z_int_task": "orig 1"}))
        TM.STAGE12_TOKENS = Path(tmp) / "tok.json"
        _, out = TM.seed_analysis(m, 0, d, original)
        assert out["diagonal"]["successes"] == 300 and out["off_diagonal"]["successes"] == 15 + 10
        assert out["gate_L1"] == {"agreement": 0.5, "n_units": 2, "below_85pct": True}, out["gate_L1"]
        r0 = out["per_task"][0]
        assert np.isclose(r0["delta"], 1 - (15 + 1) / 30 / 9) and r0["wrong_tokens_with_successes"] == {1: 15, 5: 1}
        # missing-entry detection and z_int_task check
        (d / "I-tok4_t2_e7.json").unlink()
        m2, _, missing = TM.build(d, original)
        assert missing == [(2, 4, 7)] and np.isnan(m2[2, 4, 7])
        r = json.loads((d / "I-tok6_t2_e0.json").read_text())
        r["z_int_task"] = "orig 3"
        (d / "I-tok6_t2_e0.json").write_text(json.dumps(r))
        try:
            TM.build(d, original)
            raise SystemExit("wrong z_int_task not caught")
        except AssertionError:
            pass
    print("test_token_matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
