#!/usr/bin/env python3
"""Synthetic checks for the Stage 13 code paths outside `run_stage13_human.py selftest`:
E2E-human file naming and token lookup on a fake table, plus the clip-level helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import run_stage13_human as H  # noqa: E402


def main() -> int:
    original = {t: f"orig {t}" for t in range(10)}
    tokens = {f"{t},{e}": f"orig {(t + 1) % 10}" for t in range(10) for e in range(30)}
    table = {f"orig {t}": np.full(4, float(t)) for t in range(10)}
    assert "E2E-human" in G.INTENTION_CONDITIONS and "E2E" in G.INTENTION_CONDITIONS
    assert G.intention_text("E2E-human", 3, original, None, 7, tokens) == "orig 4"
    assert table[G.intention_text("E2E-human", 3, original, None, 7, tokens)][0] == 4.0
    assert G.intention_text("E2E", 3, original, None, 7, tokens) == "orig 4", "E2E unchanged"
    assert G.intention_text("I-only", 3, original, None, 7, tokens) == "orig 3" and G.intention_text("I-wrong", 3, original, None) == "orig 8"
    # E2E-human episodes get their own file names, so Stage 12 E2E files are never skipped or overwritten
    name = lambda c, t, e: f"{c}_t{t}_e{e}.json"  # noqa: E731  (the naming used by libero_goal_ablation.run)
    assert name("E2E-human", 0, 0) == "E2E-human_t0_e0.json" != name("E2E", 0, 0)
    src = (ROOT / "scripts/libero_goal_ablation.py").read_text()
    assert 'ep_dir / f"{c}_t{t}_e{e}.json"' in src
    # every unit maps to an existing primary clip
    assert {f"t{t}_r{H.rep_of(e)}" for t, e in H.units()} == {f"t{t}_r{r}" for t in range(10) for r in H.REPS}
    assert H.cmd_selftest(None) == 0
    print("test_stage13 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
