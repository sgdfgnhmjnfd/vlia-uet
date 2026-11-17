#!/usr/bin/env python3
"""Stage 16: does the model read the action or the table? (prereg: results/prereg_stage16_scene_motion.md)

  extract  (GPU, one job) SigLIP so400m embeddings of every 4 fps grid frame of the 90 own clips, the 8-frame windows of
           the untrained Stage 13 F pipeline, and the T1 text embeddings of the 10 LIBERO descriptions
  analyze  (CPU) Part A, Part A2, continuous curves (aligned on start_s and on t_on), onset / lead, stability, Part B
  selftest

The classifier is the Stage 15 logistic regression (torch L-BFGS, C * summed CE + 0.5 ||W||^2). Features are
standardised on the training fold only. C is chosen by inner leave-one-rep-out on the training persons. LOPO is
within table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = Path("/media/dhqg/d3/datasets/own_libero10")
WORK = Path("/media/dhqg/d3/vlia_outputs/stage16")
EMB = WORK / "emb"
RES = ROOT / "results/stage16_scene_motion"
ENCODER = "google/siglip-so400m-patch14-384"
TABLES = {"A": (0, 1, 7), "B": (4, 6), "C": (2, 8), "D": (3, 9), "E": (5,)}
SHARED = ("A", "B", "C", "D")
PERSONS = ("p1", "p2", "p3")
PS = (0, 25, 50, 75, 100)
HS = (2.0, 1.0, 0.5, 0.25)
C_START = [round(-2.5 + 0.25 * i, 2) for i in range(19)]  # -2.5 .. +2.0
D_TON = [round(-1.0 + 0.25 * i, 2) for i in range(17)]  # -1.0 .. +3.0
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATS = ("S", "M", "F+")
MAXF = 16
ASK_RATE = 0.20
STAGE12_TAU = 0.0106
B_RULE, B_CURVE = 10000, 2000


# ----------------------------------------------------------------------
# metadata and windows (numpy only)
# ----------------------------------------------------------------------
def load_meta() -> dict:
    inv = {r["video_id"]: r for r in csv.DictReader((RES / "inventory.csv").open())}
    ton = {r["video_id"]: float(r["t_on_manual"]) for r in csv.DictReader((RES / "t_on_manual.csv").open())}
    out = {}
    for r in csv.DictReader((DATA / "labels_all.csv").open()):
        v = r["video_id"]
        dur = float(inv[v]["duration_s"])
        out[v] = {"video_id": v, "table": r["scene"], "task": int(r["task"]), "person": r["person"], "rep": int(r["rep"]),
                  "start": float(r["start_s"]), "end": float(r["end_s"]), "t_on": ton[v], "dur": dur, "t_end": dur - 1.0,
                  "file": r["file"]}
    return dict(sorted(out.items()))


def grid_times(dur: float) -> np.ndarray:
    return np.round(np.arange(0, dur - 0.05 + 1e-9, 0.25), 3)


def pick16(idx: np.ndarray) -> np.ndarray:
    if len(idx) <= MAXF:
        return idx
    return idx[np.round(np.linspace(0, len(idx) - 1, MAXF)).astype(int)]


def ref_idx(times: np.ndarray, t_on: float, cut: float | None = None):
    r = min(1.0, t_on - 0.25, cut if cut is not None else np.inf)
    idx = np.where(times <= r + 1e-9)[0]
    return (idx, r) if len(idx) >= 3 else (None, r)


def features(z: np.ndarray, ref: np.ndarray, obs: np.ndarray) -> dict:
    zr = z[ref].mean(0)
    if len(obs):
        zo = z[pick16(obs)]
        return {"S": zr, "M": (zo - zr).mean(0), "F+": np.concatenate([zr, zo.mean(0)])}
    return {"S": zr, "M": np.zeros_like(zr), "F+": np.concatenate([zr, zr])}


def query_cut(meta: dict, times: np.ndarray, z: np.ndarray, cut: float):
    """Features from frames with timestamp <= cut. Returns (features, info) or (None, info) if excluded."""
    ref, r = ref_idx(times, meta["t_on"], cut)
    info = {"cut": cut, "hand_visible": int(cut >= meta["t_on"] - 1e-9), "max_frame_minus_cut": None}
    if ref is None or cut < 0:
        info["excluded"] = "fewer than 3 reference frames"
        return None, info
    obs = np.where((times > r + 1e-9) & (times <= cut + 1e-9))[0]
    used = np.concatenate([ref, pick16(obs)]) if len(obs) else ref
    info["max_frame_minus_cut"] = float(times[used].max() - cut)
    info["n_obs"] = int(len(pick16(obs)))
    return features(z, ref, obs), info


def query_p(meta: dict, times: np.ndarray, z: np.ndarray, p: int):
    ref, _ = ref_idx(times, meta["t_on"])
    a, b = meta["t_on"], meta["t_on"] + p / 100 * (meta["t_end"] - meta["t_on"])
    obs = np.where((times >= a - 1e-9) & (times <= b + 1e-9))[0] if p > 0 else np.array([], int)
    return features(z, ref, obs), {"seconds": b - a}


# ----------------------------------------------------------------------
# classifier (Stage 15 LogReg, copied) and LOPO
# ----------------------------------------------------------------------
def _lbfgs(params, closure_loss, max_iter=500):
    import torch

    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        return loss

    opt.step(closure)


class LogReg:
    def __init__(self, C: float, k: int):
        self.C, self.k = C, k

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F

        X = torch.as_tensor(X, dtype=torch.float64)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        W = torch.zeros(X.shape[1], self.k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(self.k, dtype=torch.float64, requires_grad=True)
        _lbfgs([W, b], lambda: self.C * F.cross_entropy(X @ W + b, y, reduction="sum") + 0.5 * (W * W).sum())
        self.W, self.b = W.detach(), b.detach()
        return self

    def scores(self, X):
        import torch

        return torch.softmax(torch.as_tensor(X, dtype=torch.float64) @ self.W + self.b, dim=1).numpy()


def fit_predict(Xtr, ytr, Xte, C, k):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    return LogReg(C, k).fit((Xtr - mu) / sd, ytr).scores((Xte - mu) / sd)


def choose_C(X, y, reps, k):
    def score(C):
        hit = 0
        for r in np.unique(reps):
            tr, te = reps != r, reps == r
            hit += int((fit_predict(X[tr], y[tr], X[te], C, k).argmax(1) == y[te]).sum())
        return hit

    grid = list(C_GRID)
    s = {C: score(C) for C in grid}
    best = min(grid, key=lambda C: (-s[C], C))
    ext = None
    if best == grid[0] or best == grid[-1]:
        extra = grid[0] / 10 if best == grid[0] else grid[-1] * 10
        s[extra] = score(extra)
        ext = extra
        best = min(s, key=lambda C: (-s[C], C))
    return best, ext


def lopo(X, y, persons, reps, k, return_inner=False):
    """Outer leave-one-person-out. Returns probs (n, k), chosen C per person, edge extensions, and (optionally) the
    inner leave-one-person-out probabilities of each fold's training persons with the fold's C (for calibration)."""
    probs = np.zeros((len(y), k))
    Cs, exts, inner = {}, {}, {}
    for P in sorted(set(persons)):
        tr, te = persons != P, persons == P
        C, ext = choose_C(X[tr], y[tr], reps[tr], k)
        Cs[P], exts[P] = C, ext
        probs[te] = fit_predict(X[tr], y[tr], X[te], C, k)
        if return_inner:
            ip = np.zeros((int(tr.sum()), k))
            pt, yt, Xt = persons[tr], y[tr], X[tr]
            for Q in sorted(set(pt)):
                a, b = pt != Q, pt == Q
                ip[b] = fit_predict(Xt[a], yt[a], Xt[b], C, k)
            inner[P] = (ip, yt)
    return (probs, Cs, exts, inner) if return_inner else (probs, Cs, exts)


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def strat_boot_idx(strata, B, seed):
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    parts = [np.where(strata == s)[0] for s in sorted(set(strata))]
    return np.concatenate([p[rng.integers(0, len(p), size=(B, len(p)))] for p in parts], axis=1)


def paired_boot(a, b, strata, B=B_RULE, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = strat_boot_idx(strata, B, seed)
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def macro_f1(y, yhat, labels):
    f = []
    for c in labels:
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    e = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins) if i else (conf >= 0) & (conf <= 1 / bins)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def stable_onset(ok: np.ndarray):
    """Index of the earliest position from which every later entry is True; None if the last is False."""
    if not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return i


# ----------------------------------------------------------------------
# extract (GPU)
# ----------------------------------------------------------------------
def cmd_extract(args) -> int:
    import torch

    import extract_ladder_embeddings as X
    import run_stage13_human as S13
    import run_stage16_step0 as S0

    EMB.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    model, proc = S0.load_encoder()
    dev = model.device
    t0 = time.time()
    for v, m in meta.items():
        out = EMB / f"{v}.npz"
        if out.exists():
            continue
        path = DATA / m["file"]
        times = grid_times(m["dur"])
        fr, tt = S0.decode_times(path, list(times))
        assert len(tt) == len(times), (v, len(tt), len(times))
        z = S0.embed(model, proc, fr, bs=16)
        probe = S13.probe(path)
        fw = {}
        for name, b in (("f_p0", m["start"]), ("f_p50", m["start"] + 0.5 * (m["end"] - m["start"]))):
            idx = S13.window_indices(0.0, b, probe["fps"], probe["n_frames"])
            frames, _, _ = S13.decode(path, wanted=set(int(i) for i in idx))
            fw[name] = S0.embed(model, proc, [frames[int(i)] for i in idx], bs=8)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, times=times, z=z.astype(np.float32), **{k: x.astype(np.float32) for k, x in fw.items()})
        os.replace(tmp, out)
        print(f"{v}: {len(times)} frames ({time.time() - t0:.0f} s)", flush=True)
    tfile = EMB / "text_T1.npy"
    if not tfile.exists():
        original = {int(k): x for k, x in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
        desc = [S13.TEMPLATES["T1"].format(why=original[t]) for t in range(10)]
        text, trunc = X.text_features(model, proc, ENCODER, desc, dev, 32)
        assert trunc == 0
        np.save(tfile, text.astype(np.float32))
    (EMB / "extract_done.json").write_text(json.dumps({"n_clips": len(meta), "seconds": time.time() - t0,
                                                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}))
    print("EXTRACT DONE")
    return 0


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------
def load_emb(meta):
    return {v: dict(np.load(EMB / f"{v}.npz")) for v in meta}


def run_job(job):
    """One LOPO over the clips of a table for one feature set and one query definition. Returns per-clip rows."""
    import torch

    torch.set_num_threads(1)
    name, table, feat, Xd, ys, persons, reps, k, inner = job
    keep = [i for i, x in enumerate(Xd) if x is not None]
    if not keep:
        return name, table, feat, None
    X = np.stack([Xd[i] for i in keep]).astype(np.float64)
    res = lopo(X, np.asarray(ys)[keep], np.asarray(persons)[keep], np.asarray(reps)[keep], k, return_inner=inner)
    return name, table, feat, (keep, res)


def build_jobs(meta, E):
    """Every LOPO the analysis needs. Each job key = (name, table, feat)."""
    jobs, qinfo = [], {}
    tabs = {t: [v for v in meta if meta[v]["table"] == t] for t in TABLES}
    allv = list(meta)

    def add(name, table, clips, labels_of, featfn, inner=False):
        tasks = sorted(set(labels_of(v) for v in clips))
        lab = {t: i for i, t in enumerate(tasks)}
        rows = [featfn(v) for v in clips]
        for f in FEATS:
            Xd = [None if r is None else r[f] for r in rows]
            jobs.append((name, table, f, Xd, [lab[labels_of(v)] for v in clips], [meta[v]["person"] for v in clips],
                         [meta[v]["rep"] for v in clips], len(tasks), inner and f == "F+"))
        qinfo[(name, table)] = {"clips": clips, "tasks": tasks}

    task_of = lambda v: meta[v]["task"]  # noqa: E731
    soup = lambda v: 0 if meta[v]["task"] in (0, 7) else 1  # noqa: E731
    for p in PS:
        for t in SHARED:
            add(f"p{p}", t, tabs[t], task_of, lambda v, p=p: query_p(meta[v], E[v]["times"], E[v]["z"], p)[0], inner=p in (25, 50))
        add(f"p{p}", "all10", allv, task_of, lambda v, p=p: query_p(meta[v], E[v]["times"], E[v]["z"], p)[0])

    def cutfeat(v, cut, key):
        f, info = query_cut(meta[v], E[v]["times"], E[v]["z"], cut)
        qinfo[(key, v)] = info
        return f

    for H in HS:
        for t in SHARED:
            add(f"H{H}", t, tabs[t], task_of, lambda v, H=H: cutfeat(v, meta[v]["start"] - H, f"H{H}"))
        add(f"H{H}_soup", "A", tabs["A"], soup, lambda v, H=H: cutfeat(v, meta[v]["start"] - H, f"H{H}"))
    for c in C_START:
        for t in SHARED:
            add(f"cs{c}", t, tabs[t], task_of, lambda v, c=c: cutfeat(v, meta[v]["start"] + c, f"cs{c}"))
    for d in D_TON:
        for t in SHARED:
            add(f"ct{d}", t, tabs[t], task_of, lambda v, d=d: cutfeat(v, meta[v]["t_on"] + d, f"ct{d}"))
    return jobs, qinfo


def untrained_F(meta, E):
    import run_stage12_e2e as S12

    text = np.load(EMB / "text_T1.npy").astype(np.float64)
    out = {}
    for v, m in meta.items():
        others = [u for u in meta if u != v and meta[u]["table"] == m["table"] and meta[u]["person"] == m["person"]]
        mu = np.concatenate([E[u]["f_p0"] for u in others]).mean(0)
        q = S12.query_vec(E[v]["f_p50"].astype(np.float64), mu)
        s = q @ text.T
        cands = list(TABLES[m["table"]])
        sc = s[cands]
        order = np.argsort(-sc, kind="stable")
        out[v] = {"pred": cands[int(order[0])], "correct": bool(S12.strict_correct(sc, cands.index(m["task"]))) if len(cands) > 1 else None,
                  "margin": float(sc[order[0]] - sc[order[1]]) if len(cands) > 1 else np.nan}
    return out


def cmd_analyze(args) -> int:
    from multiprocessing import Pool

    meta = load_meta()
    E = load_emb(meta)
    jobs, qinfo = build_jobs(meta, E)
    # causality log (all cut-based queries)
    mx = max(i["max_frame_minus_cut"] for k, i in qinfo.items() if isinstance(i, dict) and i.get("max_frame_minus_cut") is not None)
    assert mx <= 1e-9, mx
    excluded = sorted({(k[0], k[1]) for k, i in qinfo.items() if isinstance(i, dict) and i.get("excluded")})
    print(f"{len(jobs)} LOPO jobs; max(frame time - cut) = {mx:.3g}; excluded queries: {len(excluded)}", flush=True)
    t0 = time.time()
    with Pool(int(os.environ.get("S16_PROCS", "12"))) as pool:
        done = pool.map(run_job, jobs, chunksize=4)
    print(f"LOPO done in {time.time() - t0:.0f} s", flush=True)
    R = {}
    for name, table, feat, out in done:
        R[(name, table, feat)] = out
    F_untr = untrained_F(meta, E)
    analyze(meta, R, qinfo, F_untr, excluded, mx, args)
    return 0


def per_clip(meta, R, qinfo, name, table, feat):
    """{video_id: (correct, pred_task, probs)} for one job."""
    out = R[(name, table, feat)]
    info = qinfo[(name, table)]
    if out is None:
        return {}
    keep, res = out
    probs = res[0]
    tasks = info["tasks"]
    lab = {t: i for i, t in enumerate(tasks)}
    soup = name.endswith("_soup")
    d = {}
    for j, i in enumerate(keep):
        v = info["clips"][i]
        y = (0 if meta[v]["task"] in (0, 7) else 1) if soup else lab[meta[v]["task"]]
        d[v] = (int(probs[j].argmax() == y), int(probs[j].argmax()), probs[j], y)
    return d


def pooled(meta, R, qinfo, name, feat, tables=SHARED):
    rows = {}
    for t in tables:
        rows.update(per_clip(meta, R, qinfo, name, t, feat))
    return rows


def acc_row(meta, rows, tables=SHARED):
    vs = [v for v in rows if meta[v]["table"] in tables]
    k = sum(rows[v][0] for v in vs)
    n = len(vs)
    a, lo, hi = wilson(k, n)
    chance = float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in vs])) if vs else np.nan
    f1s, w = [], []
    for t in tables:
        tv = [v for v in vs if meta[v]["table"] == t]
        if tv:
            f1s.append(macro_f1(np.array([rows[v][3] for v in tv]), np.array([rows[v][1] for v in tv]), sorted(set(rows[v][3] for v in tv))))
            w.append(len(tv))
    return {"n": n, "acc": a, "ci_lo": lo, "ci_hi": hi, "chance": chance, "macro_f1": float(np.average(f1s, weights=w)) if f1s else np.nan}


def analyze(meta, R, qinfo, F_untr, excluded, mx, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RES.mkdir(parents=True, exist_ok=True)
    shared = [v for v in meta if meta[v]["table"] in SHARED]
    strata_of = lambda vs: [meta[v]["table"] for v in vs]  # noqa: E731
    chance_pool = float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in shared]))
    cfg = {"n_clips": len(meta), "n_shared": len(shared), "pooled_chance": chance_pool, "max_frame_minus_cut": mx,
           "excluded_queries": [list(x) for x in excluded], "C_chosen": {}, "C_edge_extensions": {}}
    for (name, table, feat), out in R.items():
        if out is not None:
            cfg["C_chosen"][f"{name}|{table}|{feat}"] = out[1][1]
            ex = {p: e for p, e in out[1][2].items() if e is not None}
            if ex:
                cfg["C_edge_extensions"][f"{name}|{table}|{feat}"] = ex

    # ---------------- Part A ----------------
    acc_rows = []
    for p in PS:
        secs = [p / 100 * (meta[v]["t_end"] - meta[v]["t_on"]) for v in shared]
        for f in FEATS:
            rows = pooled(meta, R, qinfo, f"p{p}", f)
            for t in SHARED + ("pooled",):
                tabs = SHARED if t == "pooled" else (t,)
                a = acc_row(meta, rows, tabs)
                acc_rows.append({"part": "A", "table": t, "feature": f, "p": p, "seconds_mean": float(np.mean(secs)),
                                 "seconds_min": float(np.min(secs)), "seconds_max": float(np.max(secs)), **a})
            r10 = per_clip(meta, R, qinfo, f"p{p}", "all10", f)
            a = acc_row(meta, r10, tuple(TABLES))
            acc_rows.append({"part": "A-10way", "table": "all", "feature": f, "p": p, "seconds_mean": float(np.mean(secs)), **a, "chance": 0.1})
    # untrained F at p50 and most-frequent
    for t in SHARED + ("pooled",):
        tabs = SHARED if t == "pooled" else (t,)
        vs = [v for v in shared if meta[v]["table"] in tabs]
        k = sum(F_untr[v]["correct"] for v in vs)
        a, lo, hi = wilson(k, len(vs))
        acc_rows.append({"part": "A", "table": t, "feature": "untrained F (Stage 13 pipeline)", "p": 50, "n": len(vs), "acc": a, "ci_lo": lo, "ci_hi": hi,
                         "chance": float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in vs]))})
        mf = 0
        for v in vs:
            tr = [u for u in meta if meta[u]["table"] == meta[v]["table"] and meta[u]["person"] != meta[v]["person"]]
            cnt = {x: sum(meta[u]["task"] == x for u in tr) for x in TABLES[meta[v]["table"]]}
            maj = min(cnt, key=lambda x: (-cnt[x], x))
            mf += int(maj == meta[v]["task"])
        a, lo, hi = wilson(mf, len(vs))
        acc_rows.append({"part": "A", "table": t, "feature": "most-frequent", "p": 50, "n": len(vs), "acc": a, "ci_lo": lo, "ci_hi": hi,
                         "chance": float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in vs]))})
    write_csv(RES / "accuracy.csv", acc_rows)

    # rules H1-a/b/c
    rS = pooled(meta, R, qinfo, "p50", "S")
    rF = pooled(meta, R, qinfo, "p50", "F+")
    rM = pooled(meta, R, qinfo, "p50", "M")
    h1a = paired_boot([rF[v][0] for v in shared], [rS[v][0] for v in shared], strata_of(shared))
    sA = acc_row(meta, rS)
    mA = acc_row(meta, rM)
    h1b = "PASS" if sA["ci_lo"] <= chance_pool else "FAIL"
    if sA["ci_hi"] > chance_pool + 0.25:
        h1b += " (inconclusive: upper bound > chance + 0.25)"
    rules = [("H1-a: acc(F+, p50) - acc(S), paired, CI lower bound > 0", h1a, "PASS" if h1a[1] > 0 else "FAIL"),
             ("H1-b: Wilson lower bound of pooled S accuracy <= pooled chance", (sA["acc"], sA["ci_lo"], sA["ci_hi"]), h1b),
             ("H1-c: Wilson lower bound of pooled acc(M, p50) > pooled chance", (mA["acc"], mA["ci_lo"], mA["ci_hi"]), "PASS" if mA["ci_lo"] > chance_pool else "FAIL")]

    # per person and confusion at p50
    pp = []
    for P in PERSONS:
        for f, rr in (("F+", rF), ("S", rS)):
            vs = [v for v in shared if meta[v]["person"] == P]
            k = sum(rr[v][0] for v in vs)
            a, lo, hi = wilson(k, len(vs))
            pp.append({"person": P, "feature": f, "p": 50, "n": len(vs), "acc": a, "ci_lo": lo, "ci_hi": hi})
    write_csv(RES / "per_person.csv", pp)
    conf = []
    for t in SHARED:
        tasks = TABLES[t]
        for a_ in tasks:
            for b_ in tasks:
                n = sum(1 for v in shared if meta[v]["table"] == t and meta[v]["task"] == a_ and tasks[rF[v][1]] == b_)
                conf.append({"table": t, "true_task": a_, "pred_task": b_, "n": n})
    write_csv(RES / "confusion_p50.csv", conf)

    # ---------------- Part A2 ----------------
    a2 = []
    for H in HS:
        name = f"H{H}"
        rS_, rF_ = pooled(meta, R, qinfo, name, "S"), pooled(meta, R, qinfo, name, "F+")
        vs_all = [v for v in shared if v in rF_]
        hv = [v for v in vs_all if qinfo[(name, v)]["hand_visible"]]
        share = len(hv) / len(shared)
        for f in FEATS:
            rr = pooled(meta, R, qinfo, name, f)
            for subset, vs in (("all", vs_all), ("hand-visible", hv), ("no-hand", [v for v in vs_all if v not in hv])):
                if not vs:
                    a2.append({"H": H, "feature": f, "subset": subset, "n": 0})
                    continue
                k = sum(rr[v][0] for v in vs)
                a, lo, hi = wilson(k, len(vs))
                ytrue = np.array([rr[v][3] for v in vs])
                f1 = np.mean([macro_f1(np.array([rr[v][3] for v in vs if meta[v]["table"] == t]), np.array([rr[v][1] for v in vs if meta[v]["table"] == t]),
                                       sorted(set(rr[v][3] for v in vs if meta[v]["table"] == t))) for t in SHARED if any(meta[v]["table"] == t for v in vs)])
                a2.append({"H": H, "feature": f, "subset": subset, "n": len(vs), "hand_visible_share": share, "acc": a, "ci_lo": lo, "ci_hi": hi,
                           "macro_f1": float(f1), "chance": float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in vs]))})
        soupF = per_clip(meta, R, qinfo, f"H{H}_soup", "A", "F+")
        soupS = per_clip(meta, R, qinfo, f"H{H}_soup", "A", "S")
        for f, rr in (("F+", soupF), ("S", soupS)):
            k = sum(x[0] for x in rr.values())
            a, lo, hi = wilson(k, len(rr))
            a2.append({"H": H, "feature": f, "subset": "table A secondary 2-class (soup first vs 1)", "n": len(rr), "acc": a, "ci_lo": lo, "ci_hi": hi, "chance": 0.5})
        if H == 0.5:
            if share < 0.6:
                d = paired_boot([rF_[v][0] for v in hv], [rS_[v][0] for v in hv], strata_of(hv)) if hv else (np.nan,) * 3
                rules.append((f"H1-d: hand-visible at H=0.5 (share {share:.1%} < 60%): F+ - S", d, "NOT TESTABLE (pre-registered)"))
            else:
                d = paired_boot([rF_[v][0] for v in hv], [rS_[v][0] for v in hv], strata_of(hv))
                rules.append(("H1-d: hand-visible at H=0.5: F+ - S, CI lower bound > 0", d, "PASS" if d[1] > 0 else "FAIL"))
    write_csv(RES / "a2_cutoffs.csv", a2)

    # ---------------- curves, onset, lead ----------------
    onset_rows = []
    curves = {}
    for align, grid, prefix in (("start_s", C_START, "cs"), ("t_on", D_TON, "ct")):
        rows_c = []
        corr = {f: np.zeros((len(shared), len(grid)), int) for f in FEATS}
        valid = np.ones((len(shared), len(grid)), bool)
        for j, c in enumerate(grid):
            name = f"{prefix}{c}"
            rr = {f: pooled(meta, R, qinfo, name, f) for f in FEATS}
            for i, v in enumerate(shared):
                if v not in rr["F+"]:
                    valid[i, j] = False
                    continue
                for f in FEATS:
                    corr[f][i, j] = rr[f][v][0]
            vs = [v for v in shared if v in rr["F+"]]
            hv = np.mean([qinfo[(name, v)]["hand_visible"] for v in vs])
            d = paired_boot([rr["F+"][v][0] for v in vs], [rr["S"][v][0] for v in vs], strata_of(vs), B=B_CURVE)
            for f in FEATS:
                a = acc_row(meta, rr[f])
                rows_c.append({"align": align, "offset_s": c, "regime": ("before touch" if c <= 0 else "after touch") if align == "start_s" else "",
                               "feature": f, "hand_visible_share": float(hv), **a,
                               **({"F+_minus_S": d[0], "diff_lo": d[1], "diff_hi": d[2]} if f == "F+" else {})})
            for t in SHARED:
                for f in FEATS:
                    a = acc_row(meta, rr[f], (t,))
                    rows_c.append({"align": align, "offset_s": c, "regime": "per-table", "table": t, "feature": f, **a})
        write_csv(RES / f"curve_{'start' if align == 'start_s' else 'ton'}.csv", rows_c)
        curves[align] = (grid, rows_c)
        strata = np.array(strata_of(shared))
        # point onsets
        point = {}
        for f in FEATS:
            kk, n_ = (corr[f] * valid).sum(0), valid.sum(0)
            lo = np.array([wilson(kk[j], n_[j])[1] for j in range(len(grid))])
            point[f] = stable_onset(lo > chance_pool)
        # tau_gain with bootstrap CI (point estimate)
        gain_ok = []
        for j in range(len(grid)):
            m = valid[:, j]
            d = paired_boot(corr["F+"][m, j], corr["S"][m, j], strata[m], B=B_CURVE, seed=j)
            gain_ok.append(d[1] > 0)
        point["gain"] = stable_onset(np.array(gain_ok))
        # 80% onset
        accF = np.array([(corr["F+"][valid[:, j], j]).mean() for j in range(len(grid))])
        point["F+>=0.80"] = stable_onset(accF >= 0.80)
        # bootstrap intervals of the onsets
        idx = strat_boot_idx(strata, B_CURVE, seed=1)
        boots = {k: [] for k in ("S", "M", "F+", "gain")}
        for b in range(B_CURVE):
            ii = idx[b]
            vm = valid[ii]
            n_ = vm.sum(0)
            for f in FEATS:
                kk = (corr[f][ii] * vm).sum(0)
                lo = np.array([wilson(kk[j], n_[j])[1] for j in range(len(grid))])
                o = stable_onset(lo > chance_pool)
                boots[f].append(np.nan if o is None else grid[o])
            dd = (corr["F+"][ii] - corr["S"][ii]) * vm
            ok = []
            for j in range(len(grid)):
                m = vm[:, j]
                dj = dd[m, j]
                st = strata[ii][m]
                var = sum((np.var(dj[st == s], ddof=1) / (st == s).sum()) * ((st == s).sum() / len(dj)) ** 2 for s in set(st) if (st == s).sum() > 1)
                ok.append(dj.mean() - 1.959963984540054 * np.sqrt(var) > 0)
            o = stable_onset(np.array(ok))
            boots["gain"].append(np.nan if o is None else grid[o])
        for key in ("S", "M", "F+", "gain", "F+>=0.80"):
            o = point[key]
            bb = np.array(boots.get(key, []), float)
            row = {"align": align, "quantity": {"S": "tau_S", "M": "tau_M", "F+": "tau_F+", "gain": "tau_gain (F+ - S > 0)", "F+>=0.80": "F+ >= 80%"}[key],
                   "onset_offset_s": None if o is None else grid[o], "never": o is None}
            if len(bb):
                row.update({"boot_never_share": float(np.isnan(bb).mean()),
                            "boot_lo": float(np.nanquantile(bb, 0.025)) if (~np.isnan(bb)).any() else None,
                            "boot_hi": float(np.nanquantile(bb, 0.975)) if (~np.isnan(bb)).any() else None})
            onset_rows.append(row)
        # lead of motion over table
        oS, oF = point["S"], point["F+"]
        lead = {"align": align, "quantity": "lead of F+ over S (s)"}
        if oS is not None and oF is not None:
            lead["onset_offset_s"] = grid[oS] - grid[oF]
        elif oS is None and oF is not None:
            lead["onset_offset_s"] = f">= {grid[-1] - grid[oF]:.2f} (S never stably above chance; F+ from {grid[oF]:+.2f} s)"
        else:
            lead["onset_offset_s"] = "F+ never stably above chance" if oF is None else "S onset before F+"
        onset_rows.append(lead)
    write_csv(RES / "onset.csv", onset_rows)

    # stability on the pre-touch grid (alignment start_s, c <= 0), F+
    pre = [c for c in C_START if c <= 0]
    seq = {v: [] for v in shared}
    for c in pre:
        rr = pooled(meta, R, qinfo, f"cs{c}", "F+")
        for v in shared:
            seq[v].append(rr[v] if v in rr else None)
    st_rows = []
    sw, nsw, leads, leads_on, none = 0, 0, [], [], 0
    for v in shared:
        s = [x for x in seq[v]]
        cs_ = [c for c, x in zip(pre, s) if x is not None]
        s = [x for x in s if x is not None]
        preds = [x[1] for x in s]
        corr_ = np.array([x[0] for x in s], bool)
        sw += sum(int(a != b) for a, b in zip(preds, preds[1:]))
        nsw += max(0, len(preds) - 1)
        o = stable_onset(corr_) if len(corr_) else None
        if o is None:
            none += 1
            leads.append(0.0)
            lead_on = None
        else:
            leads.append(-cs_[o])
            lead_on = meta[v]["start"] + cs_[o] - meta[v]["t_on"]
            leads_on.append(lead_on)
        st_rows.append({"video_id": v, "table": meta[v]["table"], "task": meta[v]["task"], "person": meta[v]["person"],
                        "switches": sum(int(a != b) for a, b in zip(preds, preds[1:])), "n_cutoffs": len(preds),
                        "stable_correct_lead_s": leads[-1] if o is not None else 0.0, "no_stable_correct": int(o is None),
                        "stable_correct_minus_t_on_s": lead_on})
    st_sum = {"video_id": "SUMMARY", "switches": sw, "n_cutoffs": nsw, "switch_rate": sw / nsw if nsw else np.nan,
              "lead_median_s": float(np.median(leads)), "lead_iqr_lo": float(np.quantile(leads, 0.25)), "lead_iqr_hi": float(np.quantile(leads, 0.75)),
              "share_no_stable_correct": none / len(shared),
              "stable_correct_minus_t_on_median_s": float(np.median(leads_on)) if leads_on else None}
    write_csv(RES / "stability.csv", st_rows + [st_sum])

    # ---------------- Part B ----------------
    ask_rows, h3 = [], {}
    for p in (25, 50):
        allc, raw_conf, cal_conf, cor, conf_acc_c, rand_exp, strata_b, asked_all = [], [], [], [], [], [], [], []
        rates = []
        for t in SHARED:
            keep, res = R[(f"p{p}", t, "F+")]
            probs, Cs, exts, inner = res
            info = qinfo[(f"p{p}", t)]
            clips = [info["clips"][i] for i in keep]
            lab = {x: i for i, x in enumerate(info["tasks"])}
            y = np.array([lab[meta[v]["task"]] for v in clips])
            persons = np.array([meta[v]["person"] for v in clips])
            for P in PERSONS:
                te = persons == P
                ip, iy = inner[P]
                Ts = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
                lp = np.log(np.clip(ip, 1e-12, 1))
                nll = [-np.mean(np.log(np.clip(softmax_T(lp, T)[np.arange(len(iy)), iy], 1e-12, 1))) for T in Ts]
                T = float(Ts[int(np.argmin(nll))])
                tau = float(np.quantile(softmax_T(lp, T).max(1), ASK_RATE))
                pt = probs[te]
                pc = softmax_T(np.log(np.clip(pt, 1e-12, 1)), T)
                c_ = (pt.argmax(1) == y[te]).astype(int)
                ask = pc.max(1) < tau
                r = ask.mean()
                rates.append(r)
                for i in range(int(te.sum())):
                    allc.append(clips[np.where(te)[0][i]])
                raw_conf += list(pt.max(1))
                cal_conf += list(pc.max(1))
                cor += list(c_)
                conf_acc_c += list(np.where(ask, 1, c_))
                asked_all += list(ask)
                rand_exp += list(c_ + (1 - c_) * r)
                strata_b += [t] * int(te.sum())
        cor, conf_acc_c, rand_exp = np.array(cor), np.array(conf_acc_c), np.array(rand_exp)
        d = paired_boot(conf_acc_c, rand_exp, strata_b)
        e0, e1 = ece(raw_conf, cor), ece(cal_conf, cor)
        # random draws at the same per-fold rate (1000), for the mean and 95% interval
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(1000):
            draws.append(np.mean([1 if rng.random() < rr_ else cc for cc, rr_ in zip(cor, rand_rate_list(strata_b, allc, meta, rates))]))
        h3[p] = {"acc_no_ask": float(cor.mean()), "acc_confidence_ask": float(conf_acc_c.mean()), "ask_rate": float(np.mean(asked_all)),
                 "acc_random_ask_expected": float(rand_exp.mean()), "random_draws_mean": float(np.mean(draws)),
                 "random_draws_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                 "H3a_diff": d, "ECE_before": e0, "ECE_after": e1}
        # ask curve 0-50%
        order = np.argsort(np.array(cal_conf), kind="stable")
        for q in np.round(np.arange(0, 0.501, 0.05), 2):
            k = int(round(q * len(cor)))
            asked = np.zeros(len(cor), bool)
            asked[order[:k]] = True
            ask_rows.append({"p": p, "ask_rate": q, "acc_confidence": float(np.where(asked, 1, cor).mean()),
                             "acc_random_expected": float((cor + (1 - cor) * q).mean())})
    # Stage 12 margin rule on untrained F (reference)
    mF = np.array([F_untr[v]["margin"] for v in shared])
    cF = np.array([int(F_untr[v]["correct"]) for v in shared])
    askF = mF < STAGE12_TAU
    h3["stage12_margin_rule_on_untrained_F"] = {"ask_rate": float(askF.mean()), "acc_no_ask": float(cF.mean()), "acc_ask": float(np.where(askF, 1, cF).mean())}
    write_csv(RES / "ask_curve.csv", ask_rows)
    for p in (25, 50):
        x = h3[p]
        rules.append((f"H3-a (p{p}): confidence ask - random ask at the same rate, CI lower bound > 0", tuple(x["H3a_diff"]), "PASS" if x["H3a_diff"][1] > 0 else "FAIL"))
        rules.append((f"H3-b (p{p}): ECE after < ECE before", (x["ECE_after"], x["ECE_before"], np.nan), "PASS" if x["ECE_after"] < x["ECE_before"] else "FAIL"))

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, align in zip(axes, ("start_s", "t_on")):
        grid, rows_c = curves[align]
        for f, col in zip(FEATS, ("#8c8c8c", "#2a7fb8", "#d1495b")):
            rr = [r for r in rows_c if r.get("regime") != "per-table" and r["feature"] == f]
            x = [r["offset_s"] for r in rr]
            ax.plot(x, [r["acc"] for r in rr], "-o", ms=3, color=col, label=f)
            ax.fill_between(x, [r["ci_lo"] for r in rr], [r["ci_hi"] for r in rr], color=col, alpha=0.15)
        ax.axhline(chance_pool, ls=":", color="k", lw=1)
        ax.axvline(0, ls="--", color="k", lw=1)
        ax.set_xlabel(f"seconds relative to {'first touch (start_s)' if align == 'start_s' else 'hand appears (t_on)'}")
        ax.set_ylabel("within-table accuracy (pooled A-D)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "curve_seconds.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    for p, col in ((25, "#2a7fb8"), (50, "#d1495b")):
        rr = [r for r in ask_rows if r["p"] == p]
        ax.plot([r["ask_rate"] for r in rr], [r["acc_confidence"] for r in rr], "-o", ms=3, color=col, label=f"confidence ask, p{p}")
        ax.plot([r["ask_rate"] for r in rr], [r["acc_random_expected"] for r in rr], "--", color=col, label=f"random ask, p{p}")
    ax.set_xlabel("asking rate")
    ax.set_ylabel("accuracy (simulated ask-back)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "ask_curve.png", dpi=120)
    plt.close(fig)

    # ---------------- rules.md, config ----------------
    lines = ["# Stage 16 rules (pre-registered)", "", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for name, v, verdict in rules:
        lines.append(f"| {name} | {fmt(v[0])} | [{fmt(v[1])}, {fmt(v[2])}] | {verdict} |")
    lines += ["", f"Pooled chance (shared tables A-D, clip-weighted): {chance_pool:.3f}.", "",
              "## Main question: onset of stable recognition (pooled A-D)", "", "| alignment | quantity | onset (s) | bootstrap 95% | never in bootstrap |", "|---|---|---|---|---|"]
    for r in onset_rows:
        lines.append(f"| {r['align']} | {r['quantity']} | {r.get('onset_offset_s')} | [{r.get('boot_lo')}, {r.get('boot_hi')}] | {r.get('boot_never_share', '')} |")
    lines += ["", "## Part B", "", "```", json.dumps(h3, indent=1, default=float), "```"]
    (RES / "rules.md").write_text("\n".join(lines) + "\n")
    cfg.update({"part_b": h3, "stability_summary": st_sum, "t_on": "manual (t_on_manual.csv)", "classifier": "Stage 15 LogReg (torch L-BFGS), standardised on training fold",
                "C_grid": C_GRID, "max_frames": MAXF, "fps_grid": 4, "B_rule": B_RULE, "B_curve": B_CURVE,
                "macro_f1_pooled": "clip-weighted mean of per-table macro-F1",
                "tau_gain_in_bootstrap": "per-cutoff paired CI by stratified normal approximation inside each resample (point estimate uses the bootstrap CI)"})
    (RES / "config.json").write_text(json.dumps(cfg, indent=1, default=lambda x: x if not isinstance(x, np.generic) else x.item()))
    print("\n".join(lines))


def softmax_T(logp, T):
    x = logp / T
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def rand_rate_list(strata, clips, meta, rates):
    """Per-clip asking rate of its (table, person) fold, in the order the clips were appended."""
    out, i = [], 0
    folds = []
    for t in SHARED:
        for P in PERSONS:
            folds.append((t, P))
    rate_of = dict(zip(folds, rates))
    return [rate_of[(meta[v]["table"], meta[v]["person"])] for v in clips]


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:+.3f}" if isinstance(x, (float, np.floating)) else str(x)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in r.items()})


def cmd_selftest(_) -> int:
    assert stable_onset(np.array([False, True, False, True, True])) == 3
    assert stable_onset(np.array([True, True])) == 0 and stable_onset(np.array([True, False])) is None
    lo = wilson(5, 10)
    assert abs(lo[0] - 0.5) < 1e-12 and lo[1] < 0.5 < lo[2]
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (12, 5)), rng.normal(2, 1, (12, 5))])
    y = np.array([0] * 12 + [1] * 12)
    persons = np.array(["p1", "p2", "p3"] * 8)
    reps = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1] * 2)
    probs, Cs, _ = lopo(X, y, persons, reps, 2)
    assert (probs.argmax(1) == y).mean() > 0.8
    t = np.round(np.arange(0, 6, 0.25), 3)
    m = {"t_on": 2.5}
    z = rng.normal(size=(len(t), 4))
    f, info = query_cut(m, t, z, 2.6)
    assert info["max_frame_minus_cut"] <= 0 and info["hand_visible"] == 1
    f, info = query_cut(m, t, z, 0.3)
    assert f is None
    assert ece([0.9, 0.9], [1, 1]) < 0.11
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract").set_defaults(fn=cmd_extract)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
