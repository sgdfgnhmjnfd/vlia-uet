#!/usr/bin/env python3
"""Synthetic checks for scripts/run_stage10_cascade.py (no data, no model)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage10_cascade as S  # noqa: E402


def main() -> int:
    valid = np.ones(4, bool)
    # margin = top-1 - top-2, label-free
    sc = np.array([[0.1, 0.5, 0.2, 0.0, 0.4], [0.9, 0.1, 0.0, 0.0, 0.0], [0.3, 0.3, 0.1, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 9.0]])
    assert np.allclose(S.f_margin(sc), [0.1, 0.8, 0.0, 9.0])
    # 4 queries, 2 folds; F wrong where unsure, Qwen right everywhere
    zf = np.array([[0.0, 0.1, 0, 0, 0], [5.0, 0, 0, 0, 0], [0.0, 0.1, 0, 0, 0], [5.0, 0, 0, 0, 0]])
    zm = np.array([[5.0, 0, 0, 0, 0]] * 4)
    margin = S.f_margin(zf)
    splits = [(0, np.array([0, 1]), np.array([2, 3])), (1, np.array([2, 3]), np.array([0, 1]))]
    gamma, rows = S.nested_gamma({8: zf}, {8: zm}, valid, splits, (8,))
    assert gamma == {0: 0.25, 1: 0.25} and rows[0]["train_hits_gamma_0.0"] == 1, (gamma, rows)
    # b=50%: threshold = median of training margins (0.1, 5.0) -> 2.55; query 0 (margin 0.1) calls Qwen
    h, c = S.cascade(zf, zm, margin, valid, splits, gamma, 0.5)
    assert c.tolist() == [True, False, True, False] and h.tolist() == [1, 1, 1, 1]
    # b=10%: threshold 0.59 -> same low-margin queries; strict < : a margin equal to t does not call Qwen
    h, c = S.cascade(zf, zm, margin, valid, splits, {0: 0.0, 1: 0.0}, 0.1)
    assert c.tolist() == [True, False, True, False] and h.tolist() == [0, 1, 0, 1], "gamma 0 keeps F"
    tie = np.array([0.1, 0.1, 0.1, 0.1])
    _, c = S.cascade(zf, zm, tie, valid, splits, gamma, 0.25)
    assert not c.any(), "margin == threshold must not call Qwen"
    # b=100%: every query fused, even margins above every training margin
    big = np.array([1e9, 1e9, 1e9, 1e9])
    _, c = S.cascade(zf, zm, big, valid, splits, gamma, 1.0)
    assert c.all()
    print("test_stage10_cascade passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
