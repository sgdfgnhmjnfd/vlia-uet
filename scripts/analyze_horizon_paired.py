#!/usr/bin/env python3
"""Paired differences between consecutive horizons of run_anticipation_horizon.py.

Reads <run>/per_query.csv.gz. For each (label, centering) and each pair of
consecutive horizons (H_a, H_b), pairs the SAME anchors (video, t) and reports
acc5(H_a) - acc5(H_b) with a 95% CI from a bootstrap over videos. A pair whose
CI contains 0 means the two horizons cannot be told apart: the accuracy curve
has flattened there (the "convergence point").

Output: <run>/paired_consecutive.csv, also printed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paired_ci(a: np.ndarray, b: np.ndarray, vids: np.ndarray, n_boot: int, seed: int = 0):
    diff = a - b
    uniq = np.unique(vids)
    sums = np.array([diff[vids == u].sum() for u in uniq])
    counts = np.array([(vids == u).sum() for u in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "results/anticipation_horizon_v1")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    hits = defaultdict(dict)  # (label, centering) -> {(video, t, H): hit5}
    with gzip.open(args.run / "per_query.csv.gz", "rt", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["hit5"] in ("", "nan"):
                continue
            hits[(r["label"], r["centering"])][(r["video_uid"], int(r["t"]), int(r["H"]))] = float(r["hit5"])

    out = []
    for (label, centering), d in sorted(hits.items()):
        horizons = sorted({k[2] for k in d})
        for ha, hb in zip(horizons, horizons[1:]):
            keys = sorted({(v, t) for v, t, h in d if h == ha} & {(v, t) for v, t, h in d if h == hb})
            a = np.array([d[(v, t, ha)] for v, t in keys])
            b = np.array([d[(v, t, hb)] for v, t in keys])
            vids = np.array([v for v, _ in keys])
            mean, lo, hi = paired_ci(a, b, vids, args.bootstrap)
            out.append({"label": label, "centering": centering, "H_a": ha, "H_b": hb, "n": len(keys),
                        "acc_a": a.mean(), "acc_b": b.mean(), "diff": mean, "diff_lo": lo, "diff_hi": hi,
                        "flat": lo <= 0 <= hi})

    with (args.run / "paired_consecutive.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for r in out:
        print(f"{r['label']} {r['centering']} H{r['H_a']:>2}->H{r['H_b']:>2}  "
              f"{100 * r['acc_a']:.1f} -> {100 * r['acc_b']:.1f}  diff {100 * r['diff']:+.1f} "
              f"[{100 * r['diff_lo']:+.1f}; {100 * r['diff_hi']:+.1f}]  {'FLAT' if r['flat'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
