#!/usr/bin/env python3
"""Bosung 20/11: tables for block 1 (closed-loop C, E2E-sup-p25-CL) and blocks 2-3 (evaluation seeds 2000/3000).

Reads episode JSONs only (CPU). Analysis as in results/bosung_2026-11-20/prereg.md: Wilson 95% per row, paired
bootstrap over units (B = 10,000, seed 0), exact McNemar (two-sided binomial on discordant pairs). Rows that are not
complete yet are reported with their current count and marked "incomplete".

  python scripts/analyze_bosung.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/bosung_2026-11-20"
B = Path("/media/dhqg/d3/vlia_outputs/bosung_2026-11-20")
OLD_V = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/episodes")
OLD_B = Path("/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/episodes")
NB, BSEED = 10_000, 0


def load(d: Path, cond: str) -> dict:
    out = {}
    for f in glob.glob(f"{d}/{cond}_t*_e*.json"):
        r = json.load(open(f))
        out[f"{r['task']},{r['episode']}"] = int(bool(r["success"]))
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * p)


def boot_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BSEED)
    idx = rng.integers(0, len(a), size=(NB, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def units_299() -> list[str]:
    return list(json.loads((ROOT / "results/train_100k/unit_sentences.json").read_text())["units_299"])


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    U299 = units_299()
    U300 = [f"{t},{e}" for t in range(10) for e in range(30)]
    cfg = {"bootstrap_B": NB, "bootstrap_seed": BSEED, "ci": "Wilson 95%", "mcnemar": "exact two-sided binomial",
           "repo": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}

    # ---------- block 1: closed-loop C ----------
    cl = load(B / "vlia_25000/E2E-sup-p25-CL/episodes", "E2E-sup-p25-CL")
    old = {c: load(OLD_V, c) for c in ("E2E", "I-only", "I-wrong", "E2E-sup-p25")}
    built = {k: old["E2E-sup-p25"].get(k, old["I-only"].get(k)) for k in U299}  # old estimate: 298 reuse I-only
    rows1, md1 = [], []
    have = [k for k in U299 if k in cl]
    complete1 = len(have) == len(U299)
    a = np.array([cl[k] for k in have])
    lo, hi = wilson(int(a.sum()), len(a))
    md1 += ["# Block 1: closed-loop C (`E2E-sup-p25-CL`), VLIA 25k, seed 1000 + e", "",
            f"`E2E-sup-p25-CL`: **{int(a.sum())}/{len(a)}** = {pct(a.mean())}% (Wilson 95% {pct(lo)}–{pct(hi)})"
            + ("" if complete1 else " **incomplete**"), "",
            "| compared with | old | CL − old (pp) | bootstrap 95% | CL won / old won | McNemar p |", "|---|---:|---:|---|---:|---:|"]
    for name, ref in [("built estimate `E2E-sup-p25` (52/299)", built), ("`E2E` (F)", old["E2E"]),
                      ("`I-only` (same units)", old["I-only"]), ("`I-wrong`", old["I-wrong"])]:
        ks = [k for k in have if ref.get(k) is not None]
        x, y = np.array([cl[k] for k in ks]), np.array([ref[k] for k in ks])
        d, dlo, dhi = boot_diff(x, y)
        n10, n01, p = mcnemar(x, y)
        md1.append(f"| {name} | {int(y.sum())}/{len(y)} | {100 * d:+.1f} | {100 * dlo:+.1f} … {100 * dhi:+.1f} | {n10} / {n01} | {p:.3g} |")
        rows1.append({"compare": name, "n": len(ks), "cl": int(x.sum()), "old": int(y.sum()), "diff_pp": round(100 * d, 2),
                      "boot_lo": round(100 * dlo, 2), "boot_hi": round(100 * dhi, 2), "cl_only": n10, "old_only": n01, "mcnemar_p": p})
    # 2x2 vs built estimate and per task
    x = np.array([cl[k] for k in have]); y = np.array([built[k] for k in have])
    md1 += ["", "2x2 against the built estimate (units):", "",
            "| | built success | built failure |", "|---|---:|---:|",
            f"| CL success | {int(((x == 1) & (y == 1)).sum())} | {int(((x == 1) & (y == 0)).sum())} |",
            f"| CL failure | {int(((x == 0) & (y == 1)).sum())} | {int(((x == 0) & (y == 0)).sum())} |",
            "", "Per task (successes):", "", "| task | CL | built | I-only | E2E |", "|---:|---:|---:|---:|---:|"]
    for t in range(10):
        kt = [k for k in have if int(k.split(",")[0]) == t]
        md1.append(f"| {t} | {sum(cl[k] for k in kt)}/{len(kt)} | {sum(built[k] for k in kt)} | "
                   f"{sum(old['I-only'].get(k, 0) for k in kt)} | {sum(old['E2E'].get(k, 0) for k in kt)} |")
    md1 += ["", "The old built estimate stays in every table; this row is an independent supplementary measurement."]
    with (RES / "c_vong_kin.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows1[0])); w.writeheader(); w.writerows(rows1)
    (RES / "c_vong_kin.md").write_text("\n".join(md1) + "\n")

    # ---------- blocks 2-3: evaluation seeds ----------
    seeds = {"baseline": {1000: load(OLD_B, "correct"), 2000: load(B / "baseline_25000/correct_s2000/episodes", "correct"),
                          3000: load(B / "baseline_25000/correct_s3000/episodes", "correct")},
             "vlia": {1000: old["I-only"], 2000: load(B / "vlia_25000/I-only_s2000/episodes", "I-only"),
                      3000: load(B / "vlia_25000/I-only_s3000/episodes", "I-only")}}
    assert sum(seeds["baseline"][1000].values()) == 55 and sum(seeds["vlia"][1000].values()) == 52, "seed 1000 check failed"
    rows2, md2 = [], ["# Blocks 2–3: evaluation seed (rollout seed, not training seed), 25k checkpoints, 300 units", "",
                      "| model | seed 1000 | seed 2000 | seed 3000 | mean ± SD (successes) |", "|---|---:|---:|---:|---:|"]
    for m, lab in [("baseline", "baseline `correct`"), ("vlia", "VLIA `I-only`")]:
        cnt = []
        cells = []
        for s in (1000, 2000, 3000):
            r = seeds[m][s]; k, n = sum(r.values()), len(r); lo, hi = wilson(k, n); cnt.append(k)
            cells.append(f"{k}/{n} ({pct(lo)}–{pct(hi)})" + ("" if n == 300 else " incomplete"))
            rows2.append({"model": m, "seed_base": s, "success": k, "n": n, "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
        md2.append(f"| {lab} | " + " | ".join(cells) + f" | {np.mean(cnt):.1f} ± {np.std(cnt, ddof=1):.1f} |")
    md2 += ["", "VLIA − baseline per seed (paired over units):", "",
            "| seed | VLIA − baseline (pp) | bootstrap 95% | VLIA won / baseline won | McNemar p |", "|---:|---:|---|---:|---:|"]
    for s in (1000, 2000, 3000):
        ks = [k for k in U300 if k in seeds["vlia"][s] and k in seeds["baseline"][s]]
        x = np.array([seeds["vlia"][s][k] for k in ks]); y = np.array([seeds["baseline"][s][k] for k in ks])
        d, dlo, dhi = boot_diff(x, y); n10, n01, p = mcnemar(x, y)
        md2.append(f"| {s} | {100 * d:+.1f} | {100 * dlo:+.1f} … {100 * dhi:+.1f} | {n10} / {n01} | {p:.3g} |")
        rows2.append({"model": "vlia-baseline", "seed_base": s, "success": "", "n": len(ks), "wilson_lo": round(100 * dlo, 2), "wilson_hi": round(100 * dhi, 2)})
    md2 += ["", "Units whose result changes between seeds (same model, same unit):", ""]
    for m in ("baseline", "vlia"):
        ks = [k for k in U300 if all(k in seeds[m][s] for s in (1000, 2000, 3000))]
        v = np.array([[seeds[m][s][k] for s in (1000, 2000, 3000)] for k in ks])
        always, never = int((v.sum(1) == 3).sum()), int((v.sum(1) == 0).sum())
        md2.append(f"- {m}: {len(ks)} units; success in all 3 seeds {always}, in none {never}, **changes {len(ks) - always - never}**")
    md2 += ["", "Per task (successes, seeds 1000 / 2000 / 3000):", "", "| task | baseline | VLIA |", "|---:|---|---|"]
    for t in range(10):
        f = lambda m: " / ".join(str(sum(v for k, v in seeds[m][s].items() if int(k.split(",")[0]) == t)) for s in (1000, 2000, 3000))
        md2.append(f"| {t} | {f('baseline')} | {f('vlia')} |")
    with (RES / "seed_danh_gia.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows2[0])); w.writeheader(); w.writerows(rows2)
    (RES / "seed_danh_gia.md").write_text("\n".join(md2) + "\n")
    g0 = B / "g0.json"
    cfg["g0"] = json.loads(g0.read_text()) if g0.exists() else None
    (RES / "config.json").write_text(json.dumps(cfg, indent=1))
    print((RES / "c_vong_kin.md").read_text()); print((RES / "seed_danh_gia.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
