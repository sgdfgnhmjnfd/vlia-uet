#!/usr/bin/env python3
"""prereg v3 Stage 1: aggregation x context-fusion grid, nested selection vs F.

Aggregators over frames 1..k: A0 mean, A3 frame k, A4 exponential recency
(w_j ∝ exp(-(k-j)/h), h in {1,2,4}); score s = cos_C0 + lam * cos_C2 with
lam in {0, .25, .5, 1, 2, inf} (inf = cos_C2 only; (A0, inf) = F).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ladder_v1 as L  # noqa: E402
import v3_common as V  # noqa: E402

AGGS = (("A0", None), ("A3", None), ("A4", 1.0), ("A4", 2.0), ("A4", 4.0))
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0, float("inf"))
GRID = [(a, h, lam) for a, h in AGGS for lam in LAMS]
OUT = ROOT / "results/v3_stage1_aggfusion"


def name(cfg):
    a, h, lam = cfg
    return f"{a}{'' if h is None else f'h{int(h)}'}|lam={'inf' if np.isinf(lam) else lam}"


def agg_vec(frames, k, a, h):
    fk = frames[:, :k]
    if a == "A0":
        v = fk.mean(axis=1)
    elif a == "A3":
        v = fk[:, k - 1]
    else:
        j = np.arange(1, k + 1)
        w = np.exp(-(k - j) / h)
        v = np.einsum("j,njd->nd", w / w.sum(), fk)
    return L.normalize(v)


def scores(q0, q2, t0, ta, tb, ub_rows, cands=None, lam=0.0):
    """Candidate (n,K) or gallery (n,m) scores. ub_rows: (n,) use table b."""
    if cands is None:  # gallery: t0/ta/tb are (m, D)
        c0 = q0 @ t0.T
        c2 = np.where(ub_rows[:, None], q2 @ tb.T, q2 @ ta.T)
    else:
        safe = np.where(cands >= 0, cands, 0)
        c0 = np.einsum("nd,ncd->nc", q0, t0[safe])
        c2 = np.einsum("nd,ncd->nc", q2, np.where(ub_rows[:, None, None], tb[safe], ta[safe]))
    return c2 if np.isinf(lam) else c0 + lam * c2


def main() -> int:
    data = V.load_data()
    mf = V.f_gate(data)
    fr0, t0, _, _ = V.prep_c0(data)
    OUT.mkdir(parents=True, exist_ok=True)
    sel_rows, parts, desc_parts = [], [], {g: [] for g in range(len(GRID))}
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        fr2, ta, tb, ub = V.prep_f(data, tr)
        # training-fold selection
        totals = np.zeros(len(GRID), dtype=np.int64)
        cache = {}
        for gi, (a, h, lam) in enumerate(GRID):
            for k in V.PREFIXES:
                key = (a, h, k)
                if key not in cache:
                    cache[key] = (agg_vec(fr0, k, a, h), agg_vec(fr2, k, a, h))
                q0, q2 = cache[key]
                s = scores(q0[tr], q2[tr], t0, ta, tb, ub[tr], data.cands5[tr], lam)
                totals[gi] += int(L.strict_hits(s, data.valid5[tr]).sum())
        gi = int(np.argmax(totals))
        sel_rows.append({"fold": f, "grid_index": gi, "config": name(GRID[gi]),
                         "train_mean_5way": totals[gi] / (4 * len(tr)),
                         "F_train_mean_5way": totals[GRID.index(("A0", None, float("inf")))] / (4 * len(tr)),
                         "n_tied": int((totals == totals[gi]).sum())})
        # held-out metrics for the selected config and (descriptively) every config
        for g, (a, h, lam) in enumerate(GRID):
            res = {}
            for k in V.PREFIXES:
                q0, q2 = cache[(a, h, k)]
                res[k] = V.scorer_metrics(
                    lambda i, q0=q0, q2=q2, lam=lam: scores(q0[i], q2[i], t0[i], ta[i], tb[i], ub[i], None, lam),
                    lambda i, c, q0=q0, q2=q2, lam=lam: scores(q0[i], q2[i], t0, ta, tb, ub[i], c, lam),
                    data, idx)
            desc_parts[g].append((idx, res))
            if g == gi:
                parts.append((idx, res))
        print(f"fold {f}: selected {name(GRID[gi])}", flush=True)
    nested = L.pool(data, parts)
    pair = V.paired_vs_f(data, "nested", nested, mf)
    v1 = V.improves_on_f(pair)
    summary = []
    for label, m in (("F", mf), ("nested_selected", nested)):
        for k in V.PREFIXES:
            summary.append({"pipeline": label, "k": k, "MRR": float(m[k]["rr"].mean()),
                            "5way_acc": float(m[k]["kway5"].mean()), "10way_acc": float(m[k]["kway10"].mean()),
                            "5way_hits": int(m[k]["kway5"].sum())})
    desc = []
    for g in range(len(GRID)):
        pq = L.pool(data, desc_parts[g])
        for k in V.PREFIXES:
            desc.append({"grid_index": g, "config": name(GRID[g]), "k": k, "MRR": float(pq[k]["rr"].mean()),
                         "5way_acc": float(pq[k]["kway5"].mean()), "10way_acc": float(pq[k]["kway10"].mean()),
                         "note": "not selection-corrected; no verdict"})
    assert [r["5way_hits"] for r in summary if r["pipeline"] == "F"] == [V.F_HITS[k] for k in V.PREFIXES]
    # F config in the descriptive grid must equal F exactly
    fg = GRID.index(("A0", None, float("inf")))
    assert [round(r["5way_acc"] * data.n) for r in desc if r["grid_index"] == fg] == [V.F_HITS[k] for k in V.PREFIXES]
    V.write_csv(OUT / "summary.csv", summary)
    V.write_csv(OUT / "selected_by_fold.csv", sel_rows)
    V.write_csv(OUT / "paired_vs_F.csv", pair)
    V.write_csv(OUT / "grid_descriptive.csv", desc)
    verdict = {"V1": {**v1, "rule": "improves on F: 5-way delta CI lo > 0 at >= 2/4 prefixes AND MRR delta CI hi > 0 at 4/4"}}
    lines = ["# v3 Stage 1: aggregation × context fusion (nested selection vs F)", "",
             "| pipeline | k | MRR | 5-way | 10-way |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['pipeline']} | {r['k']} | {r['MRR']:.4f} | {r['5way_acc']:.4f} | {r['10way_acc']:.4f} |" for r in summary]
    lines += ["", "| k | metric | nested − F [95% CI] |", "|---:|---|---|"]
    lines += [f"| {r['k']} | {r['metric']} | {r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |" for r in pair]
    lines += ["", "| fold | selected | train mean 5-way | F train mean 5-way | ties |", "|---:|---|---:|---:|---:|"]
    lines += [f"| {r['fold']} | {r['config']} | {r['train_mean_5way']:.4f} | {r['F_train_mean_5way']:.4f} | {r['n_tied']} |"
              for r in sel_rows]
    lines += ["", "```", json.dumps(verdict, indent=2), "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    (OUT / "config.json").write_text(json.dumps({"grid": [name(g) for g in GRID], "verdicts": verdict,
                                                 "selected_by_fold": sel_rows, **V.meta()}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
