#!/usr/bin/env python3
"""Shared Track A helpers for prereg v3 (results/prereg_v3.md).

- `load_data()`: ladder Data restricted to so400m.
- `f_gate(data)`: reproduce F's out-of-fold 5-way hits exactly (853/874/907/969).
- `pipeline_metrics(...)`: per-query held-out rr / kway5 / kway10 for a scorer.
- `paired_vs_f(...)` and `improves_on_f(...)`: the common verdict rule.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ladder_v1 as L  # noqa: E402
import run_prefix_length_ablation as P  # noqa: E402
import run_scene_cv as CV  # noqa: E402

ENC = "siglip-so400m-patch14-384"
PREFIXES = (2, 4, 6, 8)
F_HITS = {2: 853, 4: 874, 6: 907, 8: 969}
PREREG = "results/prereg_v3.md"


def load_data():
    return L.Data(encoders=(ENC,), minilm=False)


def f_selected():
    rows = list(csv.DictReader((ROOT / "results/ladder_v1/selected_configs.csv").open()))
    cfgs = {(r["encoder"], r["aggregator"], r["centering"], r["template"]) for r in rows}
    assert cfgs == {(ENC, "A0", "C2", "T1")}, cfgs
    return rows


def prep_f(data, train_idx):
    """F's processing for one outer fold: (frames_C2, text_a, text_b, use_b)."""
    return data.prep(ENC, "C2", "T1", train_idx)


def prep_c0(data):
    t = data.text[ENC]["T1"]
    return data.frames[ENC], t, t, np.zeros(data.n, bool)


def scorer_metrics(score_gallery, score_cands, data, idx):
    """score_gallery(idx) -> (n, n) scores vs held-out gallery texts;
    score_cands(idx, cands) -> (n, K). Returns per-query metric dict."""
    ids = data.text_id[idx]
    gs = score_gallery(idx)
    ranks = P.first_positive_rank(gs, ids[:, None] == ids[None, :])
    s5 = score_cands(idx, data.cands5[idx])
    s10 = score_cands(idx, data.cands10[idx])
    return {"rr": 1.0 / ranks, "kway5": L.strict_hits(s5, data.valid5[idx]).astype(float),
            "kway10": L.strict_hits(s10, data.valid10[idx]).astype(float)}


def f_metrics(data):
    """F out-of-fold per-query metrics {k: {metric: (N,)}} and the gate check."""
    parts = []
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        fr, ta, tb, ub = prep_f(data, tr)
        res = {}
        for k in PREFIXES:
            res[k] = scorer_metrics(
                lambda i, k=k: L.gallery_scores(fr, ta, tb, ub, i, "A0", k),
                lambda i, c, k=k: L.candidate_scores(fr, ta, tb, ub, i, c, "A0", k),
                data, idx)
        parts.append((idx, res))
    return L.pool(data, parts)


def f_gate(data):
    f_selected()
    m = f_metrics(data)
    got = {k: int(m[k]["kway5"].sum()) for k in PREFIXES}
    if got != F_HITS:
        raise SystemExit(f"STOP Track A: F reproduction failed {got} != {F_HITS}")
    return m


def paired_vs_f(data, label, m, mf, prefixes=PREFIXES, idx=None):
    rows = []
    sel = np.arange(data.n) if idx is None else idx
    for k in prefixes:
        for metric in ("rr", "kway5", "kway10"):
            if metric not in m[k]:
                continue
            d = m[k][metric][sel] - mf[k][metric][sel]
            pt, lo, hi = CV.video_bootstrap(data.vids[sel], d, np.ones(len(sel)))
            rows.append({"comparison": f"{label} - F", "k": k, "metric": "MRR" if metric == "rr" else metric,
                         "delta": pt, "ci95_lo": lo, "ci95_hi": hi, "n_queries": int(len(sel)),
                         "value": float(m[k][metric][sel].mean()), "F_value": float(mf[k][metric][sel].mean())})
    return rows


def improves_on_f(pair_rows, prefixes=PREFIXES, mrr=True):
    k5 = [r["k"] for r in pair_rows if r["metric"] == "kway5" and r["ci95_lo"] > 0]
    out = {"5way_ci_lo_gt0_prefixes": k5, "5way_part": len(k5) >= 2}
    if mrr:
        ok = [r["k"] for r in pair_rows if r["metric"] == "MRR" and r["ci95_hi"] > 0]
        out["MRR_ci_hi_gt0_prefixes"] = ok
        out["MRR_part"] = len(ok) == len(prefixes)
        out["pass"] = out["5way_part"] and out["MRR_part"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def meta():
    return {"preregistration_sha256": {PREREG: L.sha256(ROOT / PREREG)}, "git_commit": L.git_commit(),
            "test_split_read": False}


if __name__ == "__main__":
    d = load_data()
    m = f_gate(d)
    print(json.dumps({k: int(m[k]["kway5"].sum()) for k in PREFIXES}))
