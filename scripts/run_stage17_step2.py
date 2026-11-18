#!/usr/bin/env python3
"""Stage 17 Step 2 (exploratory, outside the prereg: config (ii) passed only in G0b; D-E 55.6% < 60%).

  embed    (GPU, only while VLIA 100k is paused) SigLIP so400m on the hand crops of config (ii)
  analyze  (CPU) R0 reproduction of Stage 16; F+ vs F+H (and H alone) at p25 and p50, Stage 16 LOPO within table

H = mean over the observed frames of the crop embedding. On a frame with no crop, the full-frame embedding of that frame
is used instead (Trang, 18/11), and the share of such frames is reported.
F+H = [S, mean full-frame z_t over the observed frames, H].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage16_scene_motion as S16  # noqa: E402

WORK = Path("/media/dhqg/d3/vlia_outputs/stage17_hoi")
CROPS = WORK / "crops_ii"
EMB = WORK / "emb_ii"
RES = ROOT / "results/stage17_hoi"
R0_TARGET = {"S": 0.420, "M": 0.679, "F+": 0.741}


def cmd_embed(args) -> int:
    import run_stage16_step0 as S0

    EMB.mkdir(parents=True, exist_ok=True)
    meta = S16.load_meta()
    model, proc = S0.load_encoder()
    for v in meta:
        out = EMB / f"{v}.npz"
        if out.exists():
            continue
        c = np.load(CROPS / f"{v}.npz")
        crops = c["crops"]
        z = S0.embed(model, proc, list(crops), bs=16) if len(crops) else np.zeros((0, 1152), np.float32)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, times=c["times"], has_crop=c["has_crop"], z_crop=z.astype(np.float32))
        os.replace(tmp, out)
        print(v, len(crops), flush=True)
    print("EMBED DONE")
    return 0


def obs_idx(m, times, p):
    """Observed frame indices of Stage 16 Part A at level p (same rule as S16.query_p, at most 16)."""
    if p == 0:
        return np.array([], int)
    a, b = m["t_on"], m["t_on"] + p / 100 * (m["t_end"] - m["t_on"])
    return S16.pick16(np.where((times >= a - 1e-9) & (times <= b + 1e-9))[0])


def cmd_analyze(args) -> int:
    meta = S16.load_meta()
    E = S16.load_emb(meta)
    shared = [v for v in meta if meta[v]["table"] in S16.SHARED]
    H = {}
    for v in meta:
        e = np.load(EMB / f"{v}.npz")
        zc = np.zeros((len(e["times"]), 1152), np.float32)
        zc[e["has_crop"]] = e["z_crop"]
        assert np.allclose(e["times"], E[v]["times"])
        H[v] = (e["has_crop"], zc)

    def feats(v, p):
        m, t, z = meta[v], E[v]["times"], E[v]["z"]
        f, _ = S16.query_p(m, t, z, p)
        idx = obs_idx(m, t, p)
        has, zc = H[v]
        if len(idx):
            hz = np.where(has[idx][:, None], zc[idx], z[idx])
            h = hz.mean(0)
            fb = int((~has[idx]).sum())
        else:
            h, fb = f["S"], 0
        f["H"] = h
        f["F+H"] = np.concatenate([f["F+"], h])
        return f, len(idx), fb

    rows, paired, preds, cover, jobs = [], [], {}, [], []
    for p in (25, 50):
        F = {v: feats(v, p) for v in shared}
        for t in S16.SHARED:
            vs = [v for v in shared if meta[v]["table"] == t]
            n_obs = sum(F[v][1] for v in vs)
            n_fb = sum(F[v][2] for v in vs)
            cover.append({"p": p, "table": t, "observed_frames": n_obs, "fallback_full_frame": n_fb, "fallback_share": n_fb / n_obs})
            tasks = sorted(set(meta[v]["task"] for v in vs))
            y = np.array([tasks.index(meta[v]["task"]) for v in vs])
            persons = np.array([meta[v]["person"] for v in vs])
            reps = np.array([meta[v]["rep"] for v in vs])
            for feat in ("S", "M", "F+", "H", "F+H"):
                X = np.stack([F[v][0][feat] for v in vs]).astype(np.float64)
                jobs.append(((p, t, feat), X, y, persons, reps, len(tasks), vs, tasks))
    from multiprocessing import Pool

    with Pool(int(os.environ.get("S17_PROCS", "6"))) as pool:
        outs = pool.map(_lopo_job, jobs)
    for (key, X, y, persons, reps, k, vs, tasks), pr in zip(jobs, outs):
        p, t, feat = key
        for v, c in zip(vs, pr):
            preds[(p, feat, v)] = int(tasks[c] == meta[v]["task"])
    for p in (25, 50):
        n_all = sum(c["observed_frames"] for c in cover if c["p"] == p)
        fb_all = sum(c["fallback_full_frame"] for c in cover if c["p"] == p)
        cover.append({"p": p, "table": "pooled", "observed_frames": n_all, "fallback_full_frame": fb_all, "fallback_share": fb_all / n_all})
        for t in S16.SHARED + ("pooled",):
            vs = shared if t == "pooled" else [v for v in shared if meta[v]["table"] == t]
            chance = float(np.mean([1 / len(S16.TABLES[meta[v]["table"]]) for v in vs]))
            for feat in ("S", "M", "F+", "H", "F+H"):
                k = sum(preds[(p, feat, v)] for v in vs)
                a, lo, hi = S16.wilson(k, len(vs))
                rows.append({"p": p, "table": t, "feature": feat, "n": len(vs), "acc": a, "ci_lo": lo, "ci_hi": hi, "chance": chance})
            for a_, b_ in (("F+H", "F+"), ("H", "F+")):
                d = S16.paired_boot([preds[(p, a_, v)] for v in vs], [preds[(p, b_, v)] for v in vs], [meta[v]["table"] for v in vs])
                n10 = sum(preds[(p, a_, v)] == 1 and preds[(p, b_, v)] == 0 for v in vs)
                n01 = sum(preds[(p, a_, v)] == 0 and preds[(p, b_, v)] == 1 for v in vs)
                paired.append({"p": p, "table": t, "comparison": f"{a_} - {b_}", "diff": d[0], "ci_lo": d[1], "ci_hi": d[2],
                               "clips_only_first_correct": n10, "clips_only_second_correct": n01, "mcnemar_exact_p": mcnemar(n10, n01)})
    # R0: Stage 16 reproduction (pooled, p50 / S)
    r0 = {"S": next(r["acc"] for r in rows if r["p"] == 50 and r["table"] == "pooled" and r["feature"] == "S"),
          "M": next(r["acc"] for r in rows if r["p"] == 50 and r["table"] == "pooled" and r["feature"] == "M"),
          "F+": next(r["acc"] for r in rows if r["p"] == 50 and r["table"] == "pooled" and r["feature"] == "F+")}
    r0_pass = all(abs(r0[k] - R0_TARGET[k]) <= 0.001 + 1e-9 for k in r0)
    S16.write_csv(RES / "accuracy.csv", rows)
    S16.write_csv(RES / "paired.csv", paired)
    S16.write_csv(RES / "fallback_share.csv", cover)
    h4a = next(x for x in paired if x["p"] == 25 and x["table"] == "pooled" and x["comparison"] == "F+H - F+")
    h4b = next(r for r in rows if r["p"] == 50 and r["table"] == "pooled" and r["feature"] == "H")
    lines = ["# Stage 17 Step 2: exploratory (outside the prereg; config (ii) from G0b, D-E 55.6% < 60%)", "",
             f"R0 (Stage 16 reproduction, pooled): S {r0['S']:.3f}, M p50 {r0['M']:.3f}, F+ p50 {r0['F+']:.3f} -> {'PASS' if r0_pass else 'FAIL'}", "",
             "| rule | value | 95% CI | verdict |", "|---|---:|---|---|",
             f"| H4-a: F+H - F+ at p25 (pooled, paired) | {h4a['diff']:+.3f} | [{h4a['ci_lo']:+.3f}, {h4a['ci_hi']:+.3f}] | {'PASS' if h4a['ci_lo'] > 0 else 'FAIL'} |",
             f"| H4-b: H at p50 > chance 0.444 (Wilson lower bound) | {h4b['acc']:.3f} | [{h4b['ci_lo']:.3f}, {h4b['ci_hi']:.3f}] | {'PASS' if h4b['ci_lo'] > 0.444 else 'FAIL'} |",
             "| H4-c (stable-recognition time) | not run (Trang's Step 2 scope: p25 and p50 only) | | |"]
    (RES / "rules.md").write_text("\n".join(lines) + "\n")
    cfg = {"scope": "exploratory, outside prereg", "config": "G0b (ii): MediaPipe Hands conf 0.3 on 2x2 tiles",
           "H_no_crop_rule": "full-frame embedding of the same frame (Trang 18/11), not the prereg training-mean imputation",
           "R0": r0, "R0_pass": r0_pass, "fallback": cover}
    (RES / "config.json").write_text(json.dumps(cfg, indent=1))
    print("\n".join(lines))
    return 0


def _lopo_job(job):
    import torch

    torch.set_num_threads(1)
    key, X, y, persons, reps, k, vs, tasks = job
    probs, _, _ = S16.lopo(X, y, persons, reps, k)
    return probs.argmax(1)


def mcnemar(b, c):
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed").set_defaults(fn=cmd_embed)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
