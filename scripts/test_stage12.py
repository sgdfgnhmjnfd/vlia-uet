#!/usr/bin/env python3
"""Synthetic checks for the Stage 12 code paths outside `run_stage12_e2e.py selftest`: E2E token lookup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import run_stage12_e2e as S  # noqa: E402


def main() -> int:
    original = {t: f"orig {t}" for t in range(10)}
    tokens = {"0,0": "orig 0", "0,1": "orig 7", "8,28": "orig 8"}
    table = {f"orig {t}": np.full(4, float(t)) for t in range(10)}
    assert G.intention_text("E2E", 0, original, None, 1, tokens) == "orig 7"
    assert table[G.intention_text("E2E", 0, original, None, 1, tokens)][0] == 7.0
    assert G.intention_text("I-only", 0, original, None, 1, tokens) == "orig 0", "existing conditions unchanged"
    assert G.intention_text("I-wrong", 0, original, None) == "orig 5"
    assert "8,29" not in tokens and "E2E" in G.INTENTION_CONDITIONS
    assert S.cmd_selftest(None) == 0
    print("test_stage12 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
