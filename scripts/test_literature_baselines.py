#!/usr/bin/env python3
"""End-to-end test of scripts/run_literature_baselines.py on SYNTHETIC data only
(reuses the fake cache of test_v3_supplementary; the LLM is replaced by a stub)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402
import run_ladder_v1 as L  # noqa: E402
import run_literature_baselines as LB  # noqa: E402
import run_v3_supplementary as V  # noqa: E402
import test_v3_supplementary as T  # noqa: E402


class FakeLLM:
    def __init__(self, device):
        self.rng = np.random.default_rng(0)

    def score(self, history, options):
        assert 1 <= len(history) <= LB.HISTORY + 1 and len(options) == 5
        return list(self.rng.normal(size=5))


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        manifest, ladder, tv = T.build(tmp)
        L.MANIFEST, L.LADDER_ROOT = manifest, ladder
        V.ROOT = LB.ROOT = tmp
        LB.B = 200
        rng = np.random.default_rng(1)
        vec = lambda t: tv.setdefault(t, rng.normal(size=T.D))  # noqa: E731
        AH.encode_texts = lambda enc, texts, device=None: L.normalize(
            np.asarray([vec(t) for t in texts], dtype=np.float32))
        data = V.load_data()
        f = V.f_out_of_fold(data)
        V.F_EXPECTED_HITS = {k: int(f[k]["kway5"].sum()) for k in V.PREFIXES}
        LB.LLM = FakeLLM

        class A:
            overwrite = True
            device = "cpu"
            probe = 0
            quick = True
            no_gate = True
        assert LB.cmd_selftest(A) == 0
        assert LB.cmd_horizon(A) == 0
        assert LB.cmd_next(A) == 0
        # copy_true is always right when the target is the current step
        import csv
        rows = list(csv.DictReader((tmp / "results/lit_baselines_horizon/summary.csv").open()))
        assert {r["method"] for r in rows} == set(LB.METHODS)
        # NEXT gate against a matching reference must pass without no_gate
        A.no_gate = False
        ref = tmp / "results/v3_next_step"
        ref.mkdir(parents=True)
        nxt = list(csv.DictReader((tmp / "results/lit_baselines_next/summary.csv").open()))
        with (ref / "summary.csv").open("w") as h:
            h.write("label,centering,k,5way_acc\n")
            for r in nxt:
                if r["method"] == "ours":
                    h.write(f"NEXT,C2,{r['k']},{r['acc5']}\n")
        assert LB.cmd_next(A) == 0
        assert LB.cmd_antgpt(A) == 0
        assert LB.cmd_antgpt(A) == 0  # resume: nothing left to score
        n_lines = len((tmp / "results/lit_baselines_next/antgpt_scores/true.jsonl").read_text().splitlines())
        assert n_lines == int(data.valid5.sum()) or n_lines > 0
        assert LB.cmd_antgpt_analyze(A) == 0
        assert LB.cmd_fusion(A) == 0
        assert (tmp / "results/lit_baselines_next/fusion_summary.csv").exists()
        assert (tmp / "results/lit_baselines_next/antgpt_summary.csv").exists()
    print("all literature-baseline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
