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


C_GRID_AFTER = [round(0.25 * i, 2) for i in range(9)]  # 0 .. 2.0 s after start_s
TAU_FEATS = ("S", "M", "F+", "H", "F+H")


def cut_feats(m, t, z, has, zc, cut):
    """Stage 16 cut-based query plus H and F+H on the same observed frames (no-crop frame -> full-frame embedding)."""
    f, info = S16.query_cut(m, t, z, cut)
    if f is None:
        return None
    r = min(1.0, m["t_on"] - 0.25, cut)
    obs = S16.pick16(np.where((t > r + 1e-9) & (t <= cut + 1e-9))[0])
    assert len(obs) == 0 or t[obs].max() <= cut + 1e-9
    h = np.where(has[obs][:, None], zc[obs], z[obs]).mean(0) if len(obs) else f["S"]
    f["H"] = h
    f["F+H"] = np.concatenate([f["F+"], h])
    return f


def per_clip_tau(ok_row):
    o = S16.stable_onset(np.asarray(ok_row, bool))
    return (C_GRID_AFTER[o], False) if o is not None else (C_GRID_AFTER[-1], True)


def cmd_tau(args) -> int:
    from multiprocessing import Pool

    meta = S16.load_meta()
    E = S16.load_emb(meta)
    shared = [v for v in meta if meta[v]["table"] in S16.SHARED]
    Hc = {}
    for v in meta:
        e = np.load(EMB / f"{v}.npz")
        zc = np.zeros((len(e["times"]), 1152), np.float32)
        zc[e["has_crop"]] = e["z_crop"]
        Hc[v] = (e["has_crop"], zc)
    jobs = []
    for c in C_GRID_AFTER:
        for t in S16.SHARED:
            vs = [v for v in shared if meta[v]["table"] == t]
            F = [cut_feats(meta[v], E[v]["times"], E[v]["z"], *Hc[v], meta[v]["start"] + c) for v in vs]
            assert all(x is not None for x in F)
            tasks = sorted(set(meta[v]["task"] for v in vs))
            y = np.array([tasks.index(meta[v]["task"]) for v in vs])
            persons = np.array([meta[v]["person"] for v in vs])
            reps = np.array([meta[v]["rep"] for v in vs])
            for feat in TAU_FEATS:
                jobs.append(((c, t, feat), np.stack([x[feat] for x in F]).astype(np.float64), y, persons, reps, len(tasks), vs, tasks))
    with Pool(int(os.environ.get("S17_PROCS", "4"))) as pool:
        outs = pool.map(_lopo_job, jobs)
    corr = {f: np.zeros((len(shared), len(C_GRID_AFTER)), int) for f in TAU_FEATS}
    pos = {v: i for i, v in enumerate(shared)}
    for (key, X, y, persons, reps, k, vs, tasks), pr in zip(jobs, outs):
        c, t, feat = key
        j = C_GRID_AFTER.index(c)
        for v, cc in zip(vs, pr):
            corr[feat][pos[v], j] = int(tasks[cc] == meta[v]["task"])
    strata = np.array([meta[v]["table"] for v in shared])
    tau = {f: np.array([per_clip_tau(corr[f][i])[0] for i in range(len(shared))]) for f in TAU_FEATS}
    nost = {f: np.array([per_clip_tau(corr[f][i])[1] for i in range(len(shared))]) for f in TAU_FEATS}
    rows, onset_rows, curve = [], [], []
    idxB = S16.strat_boot_idx(strata, 10000, seed=0)
    for t in S16.SHARED + ("pooled",):
        m_ = np.ones(len(shared), bool) if t == "pooled" else strata == t
        row = {"table": t, "n": int(m_.sum())}
        for f in TAU_FEATS:
            row[f"tau_median_{f}"] = float(np.median(tau[f][m_]))
            row[f"no_stable_{f}"] = int(nost[f][m_].sum())
        d = tau["F+H"] - tau["F+"]
        if t == "pooled":
            bm = np.median(d[idxB], axis=1)
        else:
            sub = np.where(m_)[0]
            rng = np.random.default_rng(0)
            bm = np.median(d[sub][rng.integers(0, len(sub), size=(10000, len(sub)))], axis=1)
        row.update({"median_diff_FH_minus_F": float(np.median(d[m_])), "diff_ci_lo": float(np.quantile(bm, 0.025)),
                    "diff_ci_hi": float(np.quantile(bm, 0.975)), "mean_diff": float(d[m_].mean()),
                    "clips_FH_earlier": int((d[m_] < 0).sum()), "clips_FH_later": int((d[m_] > 0).sum())})
        rows.append(row)
        # Stage 16 onset (Wilson lower > chance, stable), with bootstrap interval
        chance = float(np.mean([1 / len(S16.TABLES[x]) for x in strata[m_]]))
        sub = np.where(m_)[0]
        idx = S16.strat_boot_idx(strata[sub], 2000, seed=1)
        for f in TAU_FEATS:
            k = corr[f][sub].sum(0)
            lo = np.array([S16.wilson(k[j], len(sub))[1] for j in range(len(C_GRID_AFTER))])
            o = S16.stable_onset(lo > chance)
            for j, c in enumerate(C_GRID_AFTER):
                a, l_, h_ = S16.wilson(k[j], len(sub))
                curve.append({"table": t, "feature": f, "offset_s": c, "acc": a, "ci_lo": l_, "ci_hi": h_, "chance": chance})
            bo = []
            for b in range(2000):
                kk = corr[f][sub][idx[b]].sum(0)
                lob = np.array([S16.wilson(kk[j], len(sub))[1] for j in range(len(C_GRID_AFTER))])
                ob = S16.stable_onset(lob > chance)
                bo.append(np.nan if ob is None else C_GRID_AFTER[ob])
            bo = np.array(bo)
            onset_rows.append({"table": t, "feature": f, "onset_s": None if o is None else C_GRID_AFTER[o],
                               "boot_lo": float(np.nanquantile(bo, 0.025)) if (~np.isnan(bo)).any() else None,
                               "boot_hi": float(np.nanquantile(bo, 0.975)) if (~np.isnan(bo)).any() else None,
                               "boot_never_share": float(np.isnan(bo).mean())})
    S16.write_csv(RES / "stable_tau.csv", rows)
    S16.write_csv(RES / "onset_after_contact.csv", onset_rows)
    S16.write_csv(RES / "curve_after_contact.csv", curve)
    pooled = rows[-1]
    verdict = "PASS" if pooled["diff_ci_hi"] < 0 else "FAIL"
    with (RES / "rules.md").open("a") as fh:
        fh.write(f"\n## H4-c (exploratory, run 18/11 after Step 2)\n\n| rule | value | 95% CI | verdict |\n|---|---:|---|---|\n"
                 f"| H4-c: median per-clip tau(F+H) - tau(F+), pooled A-D | {pooled['median_diff_FH_minus_F']:+.2f} s | "
                 f"[{pooled['diff_ci_lo']:+.2f}, {pooled['diff_ci_hi']:+.2f}] | {verdict} (upper bound < 0 required) |\n")
    print(json.dumps({"stable_tau": rows, "onset": onset_rows}, indent=1))
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
    sub.add_parser("tau").set_defaults(fn=cmd_tau)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
