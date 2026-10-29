#!/usr/bin/env python3
"""Synthetic checks for scripts/run_stage9.py and the 9a `frames` query (no data, no model)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage9 as S  # noqa: E402
import run_time_horizon as TH  # noqa: E402


def main() -> int:
    # text normalisation
    assert S.norm_text("  Cut  the Bread ") == "cut the bread"
    # distinct-text ranking: max over rows, wording of the best row, ties keep first appearance
    texts = ["Cut bread", "cut  bread", "wash cup", "pour tea", "open box", "close box", "stir"]
    ranked = S.rank_texts(np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.3, 0.0]), list(range(7)), texts)
    assert [r[0] for r in ranked] == ["cut bread", "wash cup", "pour tea", "close box", "open box", "stir"], ranked
    assert ranked[0][1] == "cut  bread" and ranked[0][2] == 0.9, "best row's wording and score"
    assert S.recall_at(ranked, "pour tea", 3) == 1.0 and S.recall_at(ranked, "open box", 4) == 0.0
    # row-wise z-score ignores missing options; pick ignores NaN and breaks ties to the first option
    z = S.zscore_rows(np.array([[1.0, 2.0, 3.0, np.nan, np.nan]]))
    assert np.allclose(z[0, :3], [-1.2247449, 0.0, 1.2247449]) and np.isnan(z[0, 3:]).all()
    assert S.pick(np.array([[np.nan, 1.0, 1.0], [0.0, -1.0, np.nan]])).tolist() == [1, 0]
    # nested gamma for recognition: useless Qwen keeps gamma 0 (= F); helpful Qwen is picked on the other fold
    zf = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
    ok = np.array([[False, True, False]] * 4)
    p, ch = S.nested_gamma_rec(zf, np.zeros_like(zf), ok, np.array([0, 0, 1, 1]), [0, 1])
    assert set(ch.values()) == {0.0} and p.tolist() == [0, 0, 0, 0], "ties must keep F"
    zq = np.array([[0, 5.0, 0], [0, 5.0, 0], [0, 5.0, 0], [0, 5.0, 0]])
    p, ch = S.nested_gamma_rec(zf, zq, ok, np.array([0, 0, 1, 1]), [0, 1])
    assert ch[0] > 0 and ch[1] > 0 and p.tolist() == [1, 1, 1, 1]
    # history: up to HISTORY previous steps with the k=8 label, current step with the k label
    members = {i: list(range(8)) for i in range(8)}
    lab8 = [f"s{j}@8" for j in range(8)]
    lab2 = [f"s{j}@2" for j in range(8)]
    assert S.history_from_labels(members, 7, lab8, lab2) == ["s2@8", "s3@8", "s4@8", "s5@8", "s6@8", "s7@2"]
    assert S.history_from_labels(members, 0, lab8, lab2) == ["s0@2"]
    # question and options
    q = S.QUESTION.format(options=S.options_text(["cut bread", "wash cup"]))
    assert q.endswith("Options: A. cut bread B. wash cup Answer with one letter.")
    # 9a strided query
    rng = np.random.default_rng(0)
    emb = TH.normalize(rng.normal(size=(40, 8)))
    mu = emb.mean(axis=0)
    assert np.allclose(TH.strided_query(emb, 20, 8, 8, mu), TH.query_vec(emb, 20, 8, mu)), "N=W equals contiguous"
    ref = emb[:13].mean(axis=0)
    assert np.allclose(TH.strided_query(emb, 20, 8, 2, mu), TH.normalize(TH.normalize(emb[[16, 20]] - ref).mean(axis=0)))
    print("test_stage9 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
