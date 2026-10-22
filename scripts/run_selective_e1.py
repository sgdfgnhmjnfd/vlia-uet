#!/usr/bin/env python3
"""Selective prediction on SigLIP baseline scores (`results/prereg_daily_v1.md`, Stage 2).

Per amendment 1 (`results/prereg_ladder_v1_amendment_1.md`) the baseline
embeddings are the float32 re-extraction in
`/media/dhqg/d3/vlia_outputs/ladder/siglip-base-patch16-224/`, and the
reproduction check is the amended gate (|K=5 hits - E1 hits| <= 2 per k),
which must already have passed (`results/ladder_v1/gate_amended.json`).

Scores (higher = more willing to predict): margin (PRIMARY) = top1 - top2
cosine; confidence = max softmax(cos/T); entropy_neg = -normalized entropy of
softmax(cos/T); T = 0.07 primary, 0.02 / 0.2 secondary. K=5 primary, K=10
secondary. `--selftest` is numpy-only.
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

PREFIXES = (2, 4, 6, 8)
COVERAGES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.2)
TARGET_COVERAGES = (0.8, 0.5, 0.2)
TEMPS = (0.07, 0.02, 0.2)
B_SELECTIVE = 2000


# ======================================================================
# numpy-only logic
# ======================================================================

def softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    z = scores / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decision_scores(cos: np.ndarray) -> dict[str, np.ndarray]:
    srt = np.sort(cos, axis=1)
    out = {"margin": srt[:, -1] - srt[:, -2]}
    for t in TEMPS:
        p = softmax(cos, t)
        ent = -(p * np.log(np.clip(p, 1e-15, 1.0))).sum(axis=1) / np.log(cos.shape[1])
        suffix = "" if t == 0.07 else f"_T{t}"
        out[f"confidence{suffix}"] = p.max(axis=1)
        out[f"entropy_neg{suffix}"] = -ent
    return out


def risk_curve(score: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Selective risk after accepting the top-n queries, n = 1..N
    (descending score, ties in stable input order)."""
    order = np.argsort(-score, kind="stable")
    err = 1.0 - correct[order].astype(float)
    return np.cumsum(err) / np.arange(1, len(err) + 1)


def n_at(coverage: float, n: int) -> int:
    return max(1, int(round(coverage * n)))


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_curve(score, correct).mean())


def sel_acc(score: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    return float(1.0 - risk_curve(score, correct)[n_at(coverage, len(score)) - 1])


def video_bootstrap_stats(clusters, score, correct, B=B_SELECTIVE, seed=0):
    """Resample whole videos; recompute the sort per resample. Returns arrays
    of (acc@50% - acc@100%) and (overall error - AURC)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    rng = np.random.default_rng(seed)
    gain, excess = np.zeros(B), np.zeros(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        s, c = score[idx], correct[idx]
        rc = risk_curve(s, c)
        gain[b] = (1 - rc[n_at(0.5, len(s)) - 1]) - c.mean()
        excess[b] = (1 - c.mean()) - rc.mean()
    return gain, excess


def fold_thresholds(score, fold_of, target_coverage):
    """Threshold for each fold = quantile of the score on queries OUTSIDE that
    fold; returns the accept mask pooled over held-out folds and thresholds."""
    accept = np.zeros(len(score), dtype=bool)
    thr = {}
    for f in np.unique(fold_of):
        held = fold_of == f
        q = float(np.quantile(score[~held], 1.0 - target_coverage))
        thr[int(f)] = q
        accept[held] = score[held] >= q
    return accept, thr


def selftest() -> None:
    score = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    assert np.allclose(risk_curve(score, correct), [0, 0, 1 / 3, 0.5])
    assert abs(aurc(score, correct) - (0 + 0 + 1 / 3 + 0.5) / 4) < 1e-12
    assert sel_acc(score, correct, 0.5) == 1.0 and sel_acc(score, correct, 1.0) == 0.5
    # ties broken by stable input order
    assert np.allclose(risk_curve(np.zeros(3), np.array([0, 1, 1])), [1, 0.5, 1 / 3])
    # random rejection: AURC on shuffled scores ~ overall error
    rng = np.random.default_rng(0)
    c = (rng.random(2000) < 0.3).astype(float)
    vals = [aurc(rng.random(2000), c) for _ in range(200)]
    assert abs(np.mean(vals) - (1 - c.mean())) < 0.01, np.mean(vals)
    # fold-quantile thresholding never touches the held-out fold
    s = rng.random(100)
    folds = np.repeat(np.arange(4), 25)
    acc1, thr1 = fold_thresholds(s, folds, 0.5)
    for f in range(4):
        s2 = s.copy()
        s2[folds == f] = rng.random(25) * 100  # perturb only the held-out fold
        _, thr2 = fold_thresholds(s2, folds, 0.5)
        assert thr2[f] == thr1[f]
    # bootstrap helper: a perfect score with all-correct data gives zero gain
    g, e = video_bootstrap_stats(np.repeat(np.arange(5), 4), np.arange(20.0), np.ones(20), B=50)
    assert np.allclose(g, 0) and np.allclose(e, 0)
    point, lo, hi = CV.video_bootstrap(np.repeat(np.arange(3), 2), np.ones(6), np.ones(6), B=100)
    assert point == lo == hi == 1.0
    print("selftest: PASS (risk-coverage/AURC, stable ties, random-rejection AURC, "
          "fold-quantile thresholds held-out-free, bootstrap helpers)")


# ======================================================================
# analysis
# ======================================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out: Path) -> int:
    import run_ladder_v1 as L

    gate = json.loads((ROOT / "results/ladder_v1/gate_amended.json").read_text())
    if not gate["pass"]:
        raise SystemExit("amended baseline gate did not pass; stop")
    data = L.Data(encoders=(L.BASELINE_ENCODER,), minilm=False)
    assert data.meta[L.BASELINE_ENCODER]["frame_dtype"] == "float32"
    expected = L.e1_expected()
    fr = data.frames[L.BASELINE_ENCODER]
    t0 = data.text[L.BASELINE_ENCODER]["T0"]
    gate_rows = []
    for k in PREFIXES:
        hits = L.baseline_hits(data, k)
        exp_hits = int(round(expected[k] * data.n))
        gate_rows.append({"k": k, "e1_hits": exp_hits, "hits": int(hits.sum())})
        if abs(int(hits.sum()) - exp_hits) > 2:
            raise AssertionError(f"amended gate fails at k={k}")

    out.mkdir(parents=True, exist_ok=True)
    curve_rows, summary_rows, op_rows = [], [], []
    plot_data = {}
    for K, cands, valid in ((5, data.cands5, data.valid5), (10, data.cands10, data.valid10)):
        q = np.flatnonzero(valid)  # queries with a full K-way set
        for k in PREFIXES:
            cos = L.candidate_scores(fr, t0, t0, np.zeros(data.n, bool), q, cands[q], "A0", k)
            correct = (cos[:, :1] > cos[:, 1:]).all(axis=1).astype(float)
            scores = decision_scores(cos)
            vids = data.vids[q]
            for name, s in scores.items():
                primary = name == "margin"
                rc = risk_curve(s, correct)
                n = len(s)
                step = max(1, n // 200)
                for m in list(range(step, n, step)) + [n]:
                    curve_rows.append({"K": K, "k": k, "score": name, "n_accepted": m,
                                       "coverage": m / n, "selective_risk": float(rc[m - 1])})
                if K == 5:
                    plot_data[(k, name)] = rc
                row = {"K": K, "k": k, "score": name, "role": "primary" if primary and K == 5 else "secondary",
                       "n_queries": n, "overall_acc": float(correct.mean())}
                for cov in COVERAGES:
                    row[f"acc@{int(cov * 100)}"] = float(1 - rc[n_at(cov, n) - 1])
                row["AURC"] = float(rc.mean())
                row["random_rejection_AURC"] = float(1 - correct.mean())
                row["AURC_excess"] = row["random_rejection_AURC"] - row["AURC"]
                row["gain50"] = row["acc@50"] - row["acc@100"]
                if name in ("margin", "confidence", "entropy_neg"):
                    g, e = video_bootstrap_stats(vids, s, correct)
                    row["gain50_ci_lo"], row["gain50_ci_hi"] = (float(x) for x in np.percentile(g, [2.5, 97.5]))
                    row["AURC_excess_ci_lo"], row["AURC_excess_ci_hi"] = (float(x) for x in np.percentile(e, [2.5, 97.5]))
                summary_rows.append(row)
                if name in ("margin", "confidence", "entropy_neg"):
                    for tc in TARGET_COVERAGES:
                        accept, thr = fold_thresholds(s, data.fold_of[q], tc)
                        acc_pt, acc_lo, acc_hi = CV.video_bootstrap(vids, correct * accept, accept.astype(float), B=B_SELECTIVE)
                        cov_pt, cov_lo, cov_hi = CV.video_bootstrap(vids, accept.astype(float), np.ones(n), B=B_SELECTIVE)
                        op_rows.append({
                            "K": K, "k": k, "score": name, "target_coverage": tc,
                            "realized_coverage": cov_pt, "coverage_ci_lo": cov_lo, "coverage_ci_hi": cov_hi,
                            "selective_acc": acc_pt, "selective_acc_ci_lo": acc_lo, "selective_acc_ci_hi": acc_hi,
                            "overall_acc": float(correct.mean()),
                            "thresholds_by_fold": json.dumps({f: round(v, 6) for f, v in thr.items()}),
                        })
            print(f"K={K} k={k} done", flush=True)
    for row in summary_rows:  # uniform columns
        for col in ("gain50_ci_lo", "gain50_ci_hi", "AURC_excess_ci_lo", "AURC_excess_ci_hi"):
            row.setdefault(col, "")
    write_csv(out / "risk_coverage_curves.csv", curve_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "operational_thresholds.csv", op_rows)

    primary = [r for r in summary_rows if r["K"] == 5 and r["score"] == "margin"]
    passing = [r["k"] for r in primary if r["gain50_ci_lo"] > 0]
    s1 = len(passing) >= 2
    verdict = {"S1": {"pass": s1, "prefixes_ci_lo_gt0": passing,
                      "rule": "margin, K=5: 95% CI lower bound of acc@50% - acc@100% > 0 at >= 2 of 4 prefixes"}}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, k in zip(axes, PREFIXES):
        for name in ("margin", "confidence", "entropy_neg"):
            rc = plot_data[(k, name)]
            ax.plot(np.arange(1, len(rc) + 1) / len(rc), rc, label=name)
        overall = [r for r in primary if r["k"] == k][0]["random_rejection_AURC"]
        ax.axhline(overall, color="gray", ls="--", label="random rejection")
        ax.set_title(f"K=5, k={k}")
        ax.set_xlabel("coverage")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("selective risk (1 - accuracy)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=150)
    plt.close(fig)

    lines = ["# Selective prediction on SigLIP baseline scores", "",
             "Float32 baseline re-extraction (amendment 1). Video bootstrap B=2000, seed 0.", "",
             "| K | k | score | overall acc | acc@80 | acc@50 | acc@20 | acc@50 - acc@100 [95% CI] | AURC | random AURC | AURC excess [95% CI] |",
             "|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in summary_rows:
        if r["score"] not in ("margin", "confidence", "entropy_neg"):
            continue
        lines.append(
            f"| {r['K']} | {r['k']} | {r['score']} | {r['overall_acc']:.4f} | {r['acc@80']:.4f} | {r['acc@50']:.4f} | "
            f"{r['acc@20']:.4f} | {r['gain50']:+.4f} [{r['gain50_ci_lo']:+.4f}, {r['gain50_ci_hi']:+.4f}] | "
            f"{r['AURC']:.4f} | {r['random_rejection_AURC']:.4f} | {r['AURC_excess']:+.4f} "
            f"[{r['AURC_excess_ci_lo']:+.4f}, {r['AURC_excess_ci_hi']:+.4f}] |")
    lines += ["", "## Temperature sensitivity (secondary, K=5)", "", "| k | score | acc@50 - acc@100 | AURC excess |", "|---:|---|---:|---:|"]
    for r in summary_rows:
        if r["K"] == 5 and "_T" in r["score"]:
            lines.append(f"| {r['k']} | {r['score']} | {r['gain50']:+.4f} | {r['AURC_excess']:+.4f} |")
    lines += ["", "## Operational thresholds (fold quantiles, never tuned on the evaluated fold)", "",
              "| K | k | score | target cov | realized cov [95% CI] | selective acc [95% CI] | overall acc |",
              "|---:|---:|---|---:|---|---|---:|"]
    for r in op_rows:
        lines.append(f"| {r['K']} | {r['k']} | {r['score']} | {r['target_coverage']} | {r['realized_coverage']:.3f} "
                     f"[{r['coverage_ci_lo']:.3f}, {r['coverage_ci_hi']:.3f}] | {r['selective_acc']:.4f} "
                     f"[{r['selective_acc_ci_lo']:.4f}, {r['selective_acc_ci_hi']:.4f}] | {r['overall_acc']:.4f} |")
    lines += ["", f"S1: {'PASS' if s1 else 'FAIL'} (prefixes with CI lower bound > 0: {passing})", ""]
    (out / "summary.md").write_text("\n".join(lines))
    config = {
        "embeddings": str(L.LADDER_ROOT / L.BASELINE_ENCODER), "frame_dtype": "float32",
        "gate": "amendment 1 (|hits - E1 hits| <= 2)", "gate_rows": gate_rows,
        "prefixes": PREFIXES, "kway": [5, 10], "scores": ["margin (primary)", "confidence", "entropy_neg"],
        "temperatures": {"primary": 0.07, "secondary": [0.02, 0.2]},
        "coverage_n": "round(coverage * N), at least 1", "ties": "stable input (manifest) order",
        "bootstrap": {"B": B_SELECTIVE, "seed": 0, "cluster": "video_uid", "resort_per_resample": True},
        "operational_thresholds": "per fold: quantile(score outside fold, 1 - target); accept score >= threshold",
        "verdicts": verdict,
        "preregistration_sha256": {p: L.sha256(ROOT / p) for p in L.PREREGS},
        "git_commit": L.git_commit(), "test_split_read": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/selective_e1_v1")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    return run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
