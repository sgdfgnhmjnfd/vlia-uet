#!/usr/bin/env python3
"""Stage 10: call Qwen only when F is unsure (cascade on saved scores, CPU).

Reuses the Stage 8c analyze path of scripts/v3_stage3_mllm.py (F out-of-fold candidate scores, evaluated
queries, zscore, nested gamma, strict hits). A held-out query whose F margin (top-1 - top-2 of F's 5 raw
candidate scores) is below the b-quantile of the training folds' margins uses z(F) + gamma*z(Qwen);
all others use F. b = 100% is full fusion. Pre-registration: results/prereg_stage10_cascade.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV  # noqa: E402
import v3_stage3_mllm as M  # noqa: E402

BUDGETS = (0.10, 0.25, 0.50, 1.00)
DIAG_B = 0.25
PREREG = "results/prereg_stage10_cascade.md"
REF = ROOT / "results/v3_stage3_mllm"
OUT = ROOT / "results/stage10_cascade"


def f_margin(scores: np.ndarray) -> np.ndarray:
    """top-1 - top-2 of each row's 5 candidate scores (no labels)."""
    top = -np.sort(-scores, axis=1)[:, :2]
    return top[:, 0] - top[:, 1]


def nested_gamma(zf, zm, valid, splits, prefixes, gammas=M.GAMMAS):
    """Stage 8c selection: per outer fold, argmax over gammas of strict hits summed over k on the training
    queries (np.argmax -> ties to the first, smaller gamma). splits = [(fold, test_idx, train_idx)]."""
    import run_ladder_v1 as L

    chosen, rows = {}, []
    for f, _, tr in splits:
        totals = [sum(int(L.strict_hits(zf[k][tr] + g * zm[k][tr], valid[tr]).sum()) for k in prefixes)
                  for g in gammas]
        gi = int(np.argmax(totals))
        chosen[f] = gammas[gi]
        rows.append({"fold": f, "gamma": gammas[gi], **{f"train_hits_gamma_{g}": t for g, t in zip(gammas, totals)}})
    return chosen, rows


def cascade(zf, zm, margin, valid, splits, gamma, b):
    """Strict hits and Qwen-call mask for one k and budget b."""
    import run_ladder_v1 as L

    n = len(margin)
    hits, called = np.zeros(n), np.zeros(n, bool)
    for f, idx, tr in splits:
        t = np.inf if b >= 1.0 else float(np.quantile(margin[tr], b))
        use = margin[idx] < t
        s = np.where(use[:, None], zf[idx] + gamma[f] * zm[idx], zf[idx])
        hits[idx] = L.strict_hits(s, valid[idx])
        called[idx] = use
    return hits, called


def write_csv(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    import run_ladder_v1 as L
    import v3_common as V

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-dir", default="/media/dhqg/d3/vlia_outputs/v3_mllm")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ks = M.PREFIXES
    data = V.load_data()
    mf = V.f_gate(data)  # F hits {2: 853, 4: 874, 6: 907, 8: 969} or SystemExit
    recs = [json.loads(p.read_text()) for p in sorted((Path(args.score_dir) / "scores").glob("*.json"))]
    real = {(r["sample_id"], r["k"]): r for r in recs if not r["control"]}
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    f_scores = {k: np.zeros((data.n, 5)) for k in ks}
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        fr, ta, tb, ub = V.prep_f(data, tr)
        for k in ks:
            f_scores[k][idx] = L.candidate_scores(fr, ta, tb, ub, idx, data.cands5[idx], "A0", k)
    ev = np.asarray(sorted({pos[s] for (s, k) in real if all((s, kk) in real for kk in ks)}))
    for i in ev:
        assert real[(data.sample_ids[i], 2)]["candidate_sample_ids"] == [data.sample_ids[c] for c in data.cands5[i]]
    in_ev = np.zeros(data.n, bool)
    in_ev[ev] = True
    zf = {k: M.zscore(f_scores[k]) for k in ks}
    zm = {k: np.zeros((data.n, 5)) for k in ks}
    for k in ks:
        zm[k][ev] = M.zscore(np.asarray([real[(data.sample_ids[i], k)]["score"] for i in ev]))
    splits = []
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        splits.append((f, idx[in_ev[idx]], tr[in_ev[tr]]))
    gamma, gamma_rows = nested_gamma(zf, zm, data.valid5, splits, ks)
    margin = {k: f_margin(f_scores[k]) for k in ks}
    ref = {(r["method"], int(r["k"])): r for r in csv.DictReader((REF / "summary.csv").open())}
    sec = json.loads((REF / "config.json").read_text())["run_info"]["seconds_per_query"]
    vids = data.vids[ev]

    # G1
    g1 = {}
    for k in ks:
        got, exp = float(mf[k]["kway5"][ev].mean()), float(ref[("MLLM", k)]["F_5way_acc_same_queries"])
        g1[k] = {"got": got, "expected": exp, "pass": abs(got - exp) < 1e-12}
    print(f"GATE G1 F acc on evaluated queries {g1}: {'PASS' if all(v['pass'] for v in g1.values()) else 'FAIL'}")
    if not all(v["pass"] for v in g1.values()):
        raise SystemExit("STOP: G1 failed")
    # G2
    res = {(k, b): cascade(zf[k], zm[k], margin[k], data.valid5, splits, gamma, b) for k in ks for b in BUDGETS}
    ref_g = list(csv.DictReader((REF / "fusion_gamma_by_fold.csv").open()))
    gamma_ok = [{c: str(v) for c, v in r.items()} for r in gamma_rows] == ref_g
    g2 = {"gamma_rows_match": gamma_ok}
    for k in ks:
        got, exp = float(res[(k, 1.0)][0][ev].mean()), float(ref[("fusion(F + gamma*MLLM)", k)]["5way_acc"])
        g2[k] = {"got": got, "expected": exp, "pass": abs(got - exp) < 1e-12}
    g2_pass = gamma_ok and all(g2[k]["pass"] for k in ks)
    print(f"GATE G2 full fusion {g2}: {'PASS' if g2_pass else 'FAIL'}")
    if not g2_pass:
        raise SystemExit("STOP: G2 failed")

    summary, paired, groups = [], [], []
    for k in ks:
        fh = mf[k]["kway5"][ev]
        full = res[(k, 1.0)][0][ev]
        summary.append({"k": k, "budget": "F only", "acc": float(fh.mean()), "frac_qwen": 0.0, "qwen_s_per_query": 0.0,
                        "n_queries": len(ev)})
        for b in BUDGETS:
            h, c = res[(k, b)][0][ev], res[(k, b)][1][ev]
            summary.append({"k": k, "budget": b, "acc": float(h.mean()), "frac_qwen": float(c.mean()),
                            "qwen_s_per_query": float(c.mean() * sec), "n_queries": len(ev)})
            for name, base in (("F", fh), ("full fusion", full)):
                if b >= 1.0 and name == "full fusion":
                    continue
                p, lo, hi = CV.video_bootstrap(vids, h - base, np.ones(len(ev)))
                paired.append({"k": k, "budget": b, "comparison": f"cascade - {name}", "delta": p, "ci95_lo": lo,
                               "ci95_hi": hi, "n_queries": len(ev)})
        c = res[(k, DIAG_B)][1][ev]
        for gname, sel in (("low margin (called Qwen)", c), ("high margin (F only)", ~c)):
            groups.append({"k": k, "budget": DIAG_B, "group": gname, "n": int(sel.sum()),
                           "F_acc": float(fh[sel].mean()), "fusion_acc": float(full[sel].mean()),
                           "fusion_minus_F": float((full[sel] - fh[sel]).mean())})

    def pair(k, b, name):
        return next(r for r in paired if r["k"] == k and r["budget"] == b and r["comparison"] == f"cascade - {name}")

    c1 = pair(8, DIAG_B, "F")
    full_gain = pair(8, 1.0, "F")["delta"]
    verdicts = {
        "C1": {"pass": bool(c1["delta"] > 0 and c1["ci95_lo"] > 0), "delta": c1["delta"],
               "ci95": [c1["ci95_lo"], c1["ci95_hi"]], "rule": "k=8, b=25%: cascade - F > 0, CI lower bound > 0"},
        "C2": {"pass": bool(c1["delta"] >= 0.5 * full_gain), "cascade_minus_F": c1["delta"],
               "full_fusion_minus_F": full_gain, "kept_fraction": c1["delta"] / full_gain if full_gain else None,
               "rule": "k=8, b=25%: (cascade - F) >= 0.5 * (full fusion - F), point estimates"},
    }
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "paired.csv", paired)
    write_csv(OUT / "groups.csv", groups)
    (OUT / "config.json").write_text(json.dumps({
        "budgets": BUDGETS, "diag_budget": DIAG_B, "gammas": M.GAMMAS, "gamma_by_fold": gamma,
        "qwen_seconds_per_query": sec, "quantile": "np.quantile linear on training-fold margins, use if margin < t",
        "b100": "every query fused (t = +inf)", "bootstrap": {"B": 10000, "seed": 0, "cluster": "video_uid"},
        "gates": {"G1": g1, "G2": g2}, "verdicts": verdicts,
        "preregistration_sha256": {PREREG: L.sha256(ROOT / PREREG)}, "git_commit": L.git_commit(),
        "test_split_read": False}, indent=2, default=str))
    for r in summary:
        print(f"k={r['k']} b={r['budget']}: acc {100 * r['acc']:.1f}  qwen {100 * r['frac_qwen']:.1f}%  "
              f"{r['qwen_s_per_query']:.2f} s/q")
    for r in paired:
        print(f"k={r['k']} b={r['budget']} {r['comparison']}: {100 * r['delta']:+.1f} "
              f"[{100 * r['ci95_lo']:+.1f}; {100 * r['ci95_hi']:+.1f}]")
    for r in groups:
        print(f"k={r['k']} {r['group']} n={r['n']}: F {100 * r['F_acc']:.1f}  fusion {100 * r['fusion_acc']:.1f}")
    print(json.dumps(verdicts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
