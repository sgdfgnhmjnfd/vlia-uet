#!/usr/bin/env python3
"""Selective prediction on the frozen (nested-selected) pipeline (prereg v2, Stage 2).

Out-of-fold K=5 / K=10 scores of the nested-selected ladder pipeline are
recomputed from the Stage 2 cache with each fold's selected config
(`results/ladder_v1/selected_configs.csv`; per-fold C1 means from the training
folds). They must reproduce `results/ladder_v1/summary.csv` nested_selected
5-way hits exactly at every k, else the stage stops. Analysis code is the S1
code (`run_selective_e1.py`): margin primary, B=2000 video bootstrap, seed 0.

S1b PASS iff the CI lower bound of acc@50% - acc@100% (margin, K=5) is > 0 at
>= 2 of 4 prefixes.
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
import run_scene_cv as CV  # noqa: E402
import run_selective_e1 as S  # noqa: E402

OUT = ROOT / "results/selective_frozen_v1"
PREFIXES = (2, 4, 6, 8)


def nested_scores(data: L.Data, K: int):
    """{k: (cos (N, K), valid (N,))} out of fold, each fold with its selected config."""
    sel = list(csv.DictReader((ROOT / "results/ladder_v1/selected_configs.csv").open()))
    cands = data.cands5 if K == 5 else data.cands10
    valid = (cands >= 0).all(axis=1)
    out = {k: np.zeros((data.n, K)) for k in PREFIXES}
    for row in sel:
        f = int(row["fold"])
        enc, agg, cent, tmpl = row["encoder"], row["aggregator"], row["centering"], row["template"]
        idx, train_idx = data.fold_idx(f)
        fr, ta, tb, ub = data.prep(enc, cent, tmpl, train_idx)
        for k in PREFIXES:
            out[k][idx] = L.candidate_scores(fr, ta, tb, ub, idx, cands[idx], agg, k)
    return out, valid, sel


def main() -> int:
    enc_needed = sorted({r["encoder"] for r in csv.DictReader((ROOT / "results/ladder_v1/selected_configs.csv").open())})
    data = L.Data(encoders=tuple(enc_needed), minilm=False)
    reported = {int(r["k"]): r for r in csv.DictReader((ROOT / "results/ladder_v1/summary.csv").open())
                if r["pipeline"] == "nested_selected"}
    scores = {}
    check_rows = []
    for K in (5, 10):
        cos, valid, sel = nested_scores(data, K)
        for k in PREFIXES:
            hits = L.strict_hits(cos[k], valid)
            col = "5way_acc" if K == 5 else "10way_acc"
            rep = float(reported[k][col])
            check_rows.append({"K": K, "k": k, "hits": int(hits.sum()), "acc": float(hits.mean()), "reported_acc": rep})
            if K == 5 and int(hits.sum()) != int(reported[k]["5way_hits"]):
                raise SystemExit(f"STOP: K=5 k={k} hits {int(hits.sum())} != reported {reported[k]['5way_hits']}")
            if abs(float(hits.mean()) - rep) > 1e-12:
                raise SystemExit(f"STOP: K={K} k={k} acc {hits.mean()} != reported {rep}")
            scores[(K, k)] = (cos[k][valid], valid)
    print("reproduction check:", check_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    curve_rows, summary_rows, op_rows, plot_data = [], [], [], {}
    for (K, k), (cos, valid) in scores.items():
        q = np.flatnonzero(valid)
        correct = (cos[:, :1] > cos[:, 1:]).all(axis=1).astype(float)
        vids = data.vids[q]
        for name, s in S.decision_scores(cos).items():
            rc = S.risk_curve(s, correct)
            n = len(s)
            step = max(1, n // 200)
            for m in list(range(step, n, step)) + [n]:
                curve_rows.append({"K": K, "k": k, "score": name, "n_accepted": m, "coverage": m / n,
                                   "selective_risk": float(rc[m - 1])})
            if K == 5:
                plot_data[(k, name)] = rc
            row = {"K": K, "k": k, "score": name, "role": "primary" if name == "margin" and K == 5 else "secondary",
                   "n_queries": n, "overall_acc": float(correct.mean())}
            for cov in S.COVERAGES:
                row[f"acc@{int(cov * 100)}"] = float(1 - rc[S.n_at(cov, n) - 1])
            row["AURC"] = float(rc.mean())
            row["random_rejection_AURC"] = float(1 - correct.mean())
            row["AURC_excess"] = row["random_rejection_AURC"] - row["AURC"]
            row["gain50"] = row["acc@50"] - row["acc@100"]
            row.update({"gain50_ci_lo": "", "gain50_ci_hi": "", "AURC_excess_ci_lo": "", "AURC_excess_ci_hi": ""})
            if name in ("margin", "confidence", "entropy_neg"):
                g, e = S.video_bootstrap_stats(vids, s, correct)
                row["gain50_ci_lo"], row["gain50_ci_hi"] = (float(x) for x in np.percentile(g, [2.5, 97.5]))
                row["AURC_excess_ci_lo"], row["AURC_excess_ci_hi"] = (float(x) for x in np.percentile(e, [2.5, 97.5]))
                for tc in S.TARGET_COVERAGES:
                    accept, thr = S.fold_thresholds(s, data.fold_of[q], tc)
                    a_pt, a_lo, a_hi = CV.video_bootstrap(vids, correct * accept, accept.astype(float), B=S.B_SELECTIVE)
                    c_pt, c_lo, c_hi = CV.video_bootstrap(vids, accept.astype(float), np.ones(n), B=S.B_SELECTIVE)
                    op_rows.append({"K": K, "k": k, "score": name, "target_coverage": tc,
                                    "realized_coverage": c_pt, "coverage_ci_lo": c_lo, "coverage_ci_hi": c_hi,
                                    "selective_acc": a_pt, "selective_acc_ci_lo": a_lo, "selective_acc_ci_hi": a_hi,
                                    "overall_acc": float(correct.mean()),
                                    "thresholds_by_fold": json.dumps({f: round(v, 6) for f, v in thr.items()})})
            summary_rows.append(row)
        print(f"K={K} k={k} done", flush=True)
    S.write_csv(OUT / "risk_coverage_curves.csv", curve_rows)
    S.write_csv(OUT / "summary.csv", summary_rows)
    S.write_csv(OUT / "operational_thresholds.csv", op_rows)
    S.write_csv(OUT / "reproduction_check.csv", check_rows)
    primary = [r for r in summary_rows if r["K"] == 5 and r["score"] == "margin"]
    passing = [r["k"] for r in primary if r["gain50_ci_lo"] > 0]
    verdict = {"S1b": {"pass": len(passing) >= 2, "prefixes_ci_lo_gt0": passing,
                       "rule": "margin, K=5, frozen pipeline: 95% CI lower bound of acc@50% - acc@100% > 0 at >= 2 of 4 prefixes"}}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, k in zip(axes, PREFIXES):
        for name in ("margin", "confidence", "entropy_neg"):
            rc = plot_data[(k, name)]
            ax.plot(np.arange(1, len(rc) + 1) / len(rc), rc, label=name)
        ax.axhline([r for r in primary if r["k"] == k][0]["random_rejection_AURC"], color="gray", ls="--",
                   label="random rejection")
        ax.set_title(f"frozen pipeline, K=5, k={k}")
        ax.set_xlabel("coverage")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("selective risk (1 - accuracy)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=150)
    plt.close(fig)

    lines = ["# Selective prediction on the frozen (nested-selected) pipeline", "",
             "Out-of-fold scores of `siglip-so400m-patch14-384|A0|C2|T1` (selected in every fold), recomputed and "
             "checked against results/ladder_v1/summary.csv. Video bootstrap B=2000, seed 0.", "",
             "| K | k | score | overall acc | acc@80 | acc@50 | acc@20 | acc@50 − acc@100 [95% CI] | AURC | random AURC | AURC excess [95% CI] |",
             "|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in summary_rows:
        if r["score"] not in ("margin", "confidence", "entropy_neg"):
            continue
        lines.append(f"| {r['K']} | {r['k']} | {r['score']} | {r['overall_acc']:.4f} | {r['acc@80']:.4f} | "
                     f"{r['acc@50']:.4f} | {r['acc@20']:.4f} | {r['gain50']:+.4f} [{r['gain50_ci_lo']:+.4f}, "
                     f"{r['gain50_ci_hi']:+.4f}] | {r['AURC']:.4f} | {r['random_rejection_AURC']:.4f} | "
                     f"{r['AURC_excess']:+.4f} [{r['AURC_excess_ci_lo']:+.4f}, {r['AURC_excess_ci_hi']:+.4f}] |")
    lines += ["", "## Operational thresholds (fold quantiles, never tuned on the evaluated fold)", "",
              "| K | k | score | target | realized cov [95% CI] | selective acc [95% CI] | overall acc |",
              "|---:|---:|---|---:|---|---|---:|"]
    for r in op_rows:
        lines.append(f"| {r['K']} | {r['k']} | {r['score']} | {r['target_coverage']} | {r['realized_coverage']:.3f} "
                     f"[{r['coverage_ci_lo']:.3f}, {r['coverage_ci_hi']:.3f}] | {r['selective_acc']:.4f} "
                     f"[{r['selective_acc_ci_lo']:.4f}, {r['selective_acc_ci_hi']:.4f}] | {r['overall_acc']:.4f} |")
    lines += ["", f"S1b: {'PASS' if verdict['S1b']['pass'] else 'FAIL'} (prefixes with CI lower bound > 0: {passing})", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    (OUT / "config.json").write_text(json.dumps({
        "pipeline": "nested-selected (ladder_v1), per-fold selected configs", "selected_configs": sel,
        "reproduction_check": check_rows, "scores": ["margin (primary)", "confidence", "entropy_neg"],
        "temperatures": {"primary": 0.07, "secondary": [0.02, 0.2]},
        "bootstrap": {"B": S.B_SELECTIVE, "seed": 0, "cluster": "video_uid", "resort_per_resample": True},
        "verdicts": verdict, "preregistration_sha256": {"results/prereg_v2.md": L.sha256(ROOT / "results/prereg_v2.md")},
        "git_commit": L.git_commit(), "test_split_read": False,
    }, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
