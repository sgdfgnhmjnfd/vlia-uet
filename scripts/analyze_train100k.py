#!/usr/bin/env python3
"""Final 100k tables (prereg_train100k.md). numpy only, CPU.

  python scripts/analyze_train100k.py --vlia DIR [--base DIR] [--vlia-25k DIR --base-25k DIR] [--curve DIR ...] --out RESULTS_DIR

DIR is either an eval root of run_final_eval_100k.sh (one sub-folder per row, each with episodes/*.json) or a flat
episodes folder (the 25k runs). Every *.json with condition/task/episode/success below DIR is read; episodes e >= 30
and probe files are ignored. Reused units (results/train_100k/unit_sentences.json) take the result of I-only (VLIA) or
correct (baseline) on the same unit. E2E-ask (Stage 12) = E2E with the asked units replaced by I-only.

Outputs: summary.csv/.md, paired.csv, token_vs_instruction.csv, per_task.csv, learning_curve.csv/.png, config.json.
Rules T1-T5: paired bootstrap over units (B = 10,000, seed 0), CI lower bound > 0.
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
from libero_goal_ablation import paired_bootstrap, wilson  # noqa: E402

US = ROOT / "results/train_100k/unit_sentences.json"
E2E = ROOT / "results/stage12_end_to_end/e2e_tokens.json"
VLIA_ROWS = ("I-only", "I-wrong", "I-para", "I-goal", "E2E", "E2E-ask", "E2E-sup-p25", "E2E-gen-p25")
BASE_ROWS = ("correct", "empty", "L-para", "L-goal", "L-E2E", "L-sup-p25", "L-gen-p25")
RULES = [("T1", "I-only", "empty"), ("T2", "I-only", "I-wrong"), ("T3", "I-goal", "I-wrong"), ("T4", "I-para", "I-wrong"),
         ("T5", "E2E", "I-wrong")]
TOKEN_VS_INSTR = [("I-para", "L-para"), ("I-goal", "L-goal"), ("E2E", "L-E2E"), ("E2E-sup-p25", "L-sup-p25"),
                  ("E2E-gen-p25", "L-gen-p25")]
OTHER = [("I-only", "correct"), ("L-para", "correct"), ("L-goal", "correct"), ("E2E-ask", "E2E")]


def load_eps(dirs) -> dict:
    """{(condition, task, episode): success} from every episode JSON under the folders (e < 30, no probe)."""
    out = {}
    for d in dirs or []:
        for p in Path(d).rglob("*.json"):
            if "probe" in p.parts or p.name in ("instructions.json",):
                continue
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(r, dict) or not {"condition", "task", "episode", "success"} <= set(r):
                continue
            if r["episode"] >= 30:
                continue
            out[(r["condition"], int(r["task"]), int(r["episode"]))] = bool(r["success"])
    return out


def row_success(eps: dict, us: dict, ask: dict) -> dict:
    """{row: {unit: 0/1}} for every row with all its units present (reuse resolved); rows with missing units omitted."""
    rows = us["rows"]
    res = {}
    for r in us["order"] + ["E2E-ask"]:
        units = us["units_299"] if r == "E2E-ask" else list(rows[r])
        vals = {}
        for k in units:
            t, e = map(int, k.split(","))
            if r == "E2E-ask":
                src = ("I-only", t, e) if ask.get(k) else ("E2E", t, e)
            else:
                src = ((rows[r][k]["reuse"], t, e) if rows[r][k]["reuse"] else (r, t, e))
            if src in eps:
                vals[k] = int(eps[src])
        if vals and len(vals) == len(units):
            res[r] = vals
        elif vals:
            res[r + " (incomplete)"] = vals
    return res


def fmt(x, n=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def paired(S: dict, a: str, b: str):
    if a not in S or b not in S:
        return None
    units = [k for k in S[a] if k in S[b]]
    d = np.array([S[a][k] - S[b][k] for k in units], float)
    m, lo, hi = paired_bootstrap(d, B=10000, seed=0)
    return {"a": a, "b": b, "n_units": len(units), "diff": m, "ci_lo": lo, "ci_hi": hi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlia", nargs="*", default=[])
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--vlia-25k", nargs="*", default=[])
    ap.add_argument("--base-25k", nargs="*", default=[])
    ap.add_argument("--curve", nargs="*", default=[], help="curve eval roots, named <model>_<step>_curve")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    us = json.loads(US.read_text())
    ask = json.loads(E2E.read_text())["ask"]
    S = row_success(load_eps(a.vlia + a.base), us, ask)
    S25 = row_success(load_eps(a.vlia_25k + a.base_25k), us, ask) if (a.vlia_25k or a.base_25k) else {}

    summ = []
    for r in list(VLIA_ROWS) + list(BASE_ROWS):
        row = {"row": r, "model": "VLIA" if r in VLIA_ROWS else "baseline"}
        for tag, D in (("", S), ("_25k", S25)):
            v = D.get(r)
            if v:
                k, n = sum(v.values()), len(v)
                lo, hi = wilson(k, n)
                row.update({f"successes{tag}": k, f"units{tag}": n, f"rate{tag}": k / n, f"wilson_lo{tag}": lo, f"wilson_hi{tag}": hi})
        summ.append(row)
    write_csv(out / "summary.csv", summ)
    rules = []
    for name, x, y in RULES:
        p = paired(S, x, y)
        if p:
            p.update({"rule": name, "verdict": "PASS" if p["ci_lo"] > 0 else "FAIL"})
            rules.append(p)
    other = [p for p in (paired(S, x, y) for x, y in OTHER) if p]
    tvi = [p for p in (paired(S, x, y) for x, y in TOKEN_VS_INSTR) if p]
    for p in tvi:
        p["better"] = "token" if p["ci_lo"] > 0 else ("instruction" if p["ci_hi"] < 0 else "no clear difference")
    vs25 = []
    for r in S:
        if r in S25:
            p = paired({"now": S[r], "25k": S25[r]}, "now", "25k")
            if p:
                p.update({"a": f"{r} (this run)", "b": f"{r} (25k)"})
                vs25.append(p)
    write_csv(out / "paired.csv", [dict(kind="rule", **p) for p in rules] + [dict(kind="descriptive", **p) for p in other + vs25])
    write_csv(out / "token_vs_instruction.csv", tvi)
    per = []
    for r, v in S.items():
        for t in range(10):
            u = [x for k, x in v.items() if k.startswith(f"{t},")]
            if u:
                per.append({"row": r, "task": t, "successes": sum(u), "units": len(u)})
    write_csv(out / "per_task.csv", per)
    # learning curve (10 episodes per task at each milestone)
    curve = []
    for d in a.curve:
        name = Path(d).name
        model, step = name.split("_")[0], int(name.split("_")[1])
        eps = load_eps([d])
        for c in ("I-only", "correct"):
            v = [s for (cc, t, e), s in eps.items() if cc == c and e < 10]
            if v:
                lo, hi = wilson(sum(v), len(v))
                curve.append({"model": model, "step": step, "row": c, "successes": sum(v), "units": len(v), "wilson_lo": lo, "wilson_hi": hi})
    write_csv(out / "learning_curve.csv", curve)
    if curve:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for model, col in (("vlia", "#d1495b"), ("baseline", "#2a7fb8")):
            pts = sorted((c["step"], c["successes"] / c["units"]) for c in curve if c["model"] == model)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, label=model)
        ax.set_xlabel("training step"); ax.set_ylabel("success rate (10 episodes/task)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=120)
    md = ["| row | model | successes / units | rate | Wilson 95% | 25k |", "|---|---|---|---|---|---|"]
    for r in summ:
        if "units" in r or "units_25k" in r:
            md.append(f"| {r['row']} | {r['model']} | {r.get('successes', '')}/{r.get('units', '')} | {fmt(r.get('rate'))} | "
                      f"[{fmt(r.get('wilson_lo'))}, {fmt(r.get('wilson_hi'))}] | {r.get('successes_25k', '')}/{r.get('units_25k', '')} |")
    md += ["", "| rule | a − b | units | diff | 95% CI | verdict |", "|---|---|---|---|---|---|"]
    md += [f"| {p['rule']} | {p['a']} − {p['b']} | {p['n_units']} | {p['diff']:+.3f} | [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | {p['verdict']} |" for p in rules]
    md += ["", "| token − instruction | units | diff | 95% CI | better |", "|---|---|---|---|---|"]
    md += [f"| {p['a']} − {p['b']} | {p['n_units']} | {p['diff']:+.3f} | [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | {p['better']} |" for p in tvi]
    md += ["", "| descriptive | units | diff | 95% CI |", "|---|---|---|---|"]
    md += [f"| {p['a']} − {p['b']} | {p['n_units']} | {p['diff']:+.3f} | [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] |" for p in other + vs25]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    (out / "config.json").write_text(json.dumps({"vlia": a.vlia, "base": a.base, "vlia_25k": a.vlia_25k, "base_25k": a.base_25k,
                                                 "curve": a.curve, "unit_sentences": str(US), "bootstrap": {"B": 10000, "seed": 0},
                                                 "rows_found": sorted(S), "rows_found_25k": sorted(S25)}, indent=1))
    print("\n".join(md))
    return 0


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
