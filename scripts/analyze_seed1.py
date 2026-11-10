#!/usr/bin/env python3
"""VLIA seed 1 vs seed 0 on LIBERO-10 (I-only, I-wrong, E2E; e = 0..29). Pre-registration: results/prereg_seed1.md.

S1: seed-1 I-only - seed-1 I-wrong; S2: seed-1 E2E - seed-1 I-wrong (299 units). Paired bootstrap over units.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402

OUT = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1")
CONDS = ("I-only", "I-wrong", "E2E")
N_TASKS, N_EP = 10, 30


def load_eps(ep_dir: Path) -> dict:
    by = {}
    for p in sorted(Path(ep_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if r["condition"] in CONDS:
            by[(r["condition"], r["task"], r["episode"])] = r
    return by


def units_for(cond: str, e2e_keys) -> list:
    if cond == "E2E":
        return sorted(tuple(map(int, k.split(","))) for k in e2e_keys)
    return [(t, e) for t in range(N_TASKS) for e in range(N_EP)]


def paired(by0: dict, by1: dict, cond: str, units: list):
    """Success arrays of both seeds, matched by (task, episode); seeds and init states must agree."""
    s0, s1 = [], []
    for t, e in units:
        a, b = by0[(cond, t, e)], by1[(cond, t, e)]
        assert (a["seed"], a["init_state_id"]) == (b["seed"], b["init_state_id"]) == (1000 + e, e), (cond, t, e)
        s0.append(float(a["success"]))
        s1.append(float(b["success"]))
    return np.array(s0), np.array(s1)


def pooled(s0: np.ndarray, s1: np.ndarray):
    """Two-seed rate; resampling units keeps both seeds of a unit together."""
    return G.paired_bootstrap((s0 + s1) / 2)


def write_csv(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def analyze(dir0, dir1, tokens_path, res, extra) -> dict:
    res.mkdir(parents=True, exist_ok=True)
    tokens = json.loads(Path(tokens_path).read_text())["tokens"]
    by0, by1 = load_eps(dir0), load_eps(dir1)
    for (c, t, e), r in by1.items():
        if c == "E2E":
            assert r["z_int_task"] == tokens[f"{t},{e}"], (t, e)
    succ, units = {}, {}
    for c in CONDS:
        units[c] = units_for(c, tokens)
        succ[c] = paired(by0, by1, c, units[c])
    summary, pairs, per_task = [], [], []
    for c in CONDS:
        for seed, s in (("seed 0", succ[c][0]), ("seed 1", succ[c][1])):
            k, n = int(s.sum()), len(s)
            lo, hi = G.wilson(k, n)
            summary.append({"condition": c, "seed": seed, "successes": k, "n": n, "rate": k / n, "ci95_lo": lo, "ci95_hi": hi,
                            "ci_method": "Wilson"})
        p, lo, hi = pooled(*succ[c])
        summary.append({"condition": c, "seed": "pooled", "successes": int(succ[c][0].sum() + succ[c][1].sum()),
                        "n": 2 * len(units[c]), "rate": p, "ci95_lo": lo, "ci95_hi": hi, "ci_method": "unit bootstrap, seeds kept together"})
    wrong1 = dict(zip(units["I-wrong"], succ["I-wrong"][1]))
    for rule, c in (("S1", "I-only"), ("S2", "E2E")):
        w = np.array([wrong1[u] for u in units[c]])
        d, lo, hi = G.paired_bootstrap(succ[c][1] - w)
        pairs.append({"rule": rule, "comparison": f"seed-1 {c} - seed-1 I-wrong", "delta": d, "ci95_lo": lo, "ci95_hi": hi,
                      "n_units": len(units[c]), "pass": lo > 0})
    for c in CONDS:
        d, lo, hi = G.paired_bootstrap(succ[c][1] - succ[c][0])
        pairs.append({"rule": "descriptive", "comparison": f"{c}: seed 1 - seed 0", "delta": d, "ci95_lo": lo, "ci95_hi": hi,
                      "n_units": len(units[c]), "pass": ""})
    for t in range(N_TASKS):
        row = {"task": t}
        for c in CONDS:
            m = np.array([u[0] == t for u in units[c]])
            row[f"{c}_seed0"], row[f"{c}_seed1"], row[f"{c}_n"] = int(succ[c][0][m].sum()), int(succ[c][1][m].sum()), int(m.sum())
        per_task.append(row)
    verdicts = {r["rule"]: {"pass": bool(r["pass"]), "delta": r["delta"], "ci95": [r["ci95_lo"], r["ci95_hi"]], "rule": r["comparison"] + " CI lower bound > 0"}
                for r in pairs if r["rule"] in ("S1", "S2")}
    write_csv(res / "summary.csv", summary)
    write_csv(res / "paired.csv", pairs)
    write_csv(res / "per_task.csv", per_task)
    lines = ["# VLIA seed 1 vs seed 0 (LIBERO-10, e = 0..29)", "", "| condition | seed | successes / n | rate | 95% CI |", "|---|---|---:|---:|---|"]
    lines += [f"| {r['condition']} | {r['seed']} | {r['successes']} / {r['n']} | {r['rate']:.3f} | [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] |" for r in summary]
    lines += ["", "| rule | comparison | delta [95% CI] |", "|---|---|---|"]
    lines += [f"| {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in pairs]
    lines += ["", "| task | " + " | ".join(f"{c} s0 / s1" for c in CONDS) + " |", "|---:|---|---|---|"]
    lines += [f"| {r['task']} | " + " | ".join(f"{r[c + '_seed0']} / {r[c + '_seed1']} (of {r[c + '_n']})" for c in CONDS) + " |" for r in per_task]
    lines += ["", "```", json.dumps(verdicts, indent=2), "```", ""]
    (res / "summary.md").write_text("\n".join(lines))
    secs = {c: float(np.mean([by1[(c, t, e)]["seconds"] for t, e in units[c]])) for c in CONDS}
    (res / "config.json").write_text(json.dumps({**extra, "seed0_episodes": str(dir0), "seed1_episodes": str(dir1),
                                                 "e2e_tokens": str(tokens_path), "seconds_per_episode_seed1": secs,
                                                 "bootstrap": {"B": 10000, "seed": 0, "unit": "(task, episode)"},
                                                 "verdicts": verdicts}, indent=2))
    print((res / "summary.md").read_text())
    return verdicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed0-episodes", default=str(OUT / "eval/episodes"))
    ap.add_argument("--seed1-episodes", default=str(OUT / "eval_seed1/episodes"))
    ap.add_argument("--e2e-tokens", default=str(ROOT / "results/stage12_end_to_end/e2e_tokens.json"))
    ap.add_argument("--results", default=str(ROOT / "results/seed1_vlia"))
    ap.add_argument("--extra-json", help="training / S0 metadata to merge into config.json")
    args = ap.parse_args()
    extra = json.loads(Path(args.extra_json).read_text()) if args.extra_json else {}
    extra["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    analyze(Path(args.seed0_episodes), Path(args.seed1_episodes), args.e2e_tokens, Path(args.results), extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
