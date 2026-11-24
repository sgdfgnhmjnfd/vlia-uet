"""Stage 19 (method B): frozen VideoMAE-base fine-tuned on SSv2, scene / motion / video on own videos and EgoIntent.

Pre-registration: results/stage19_temporal/prereg.md (hash in prereg_hash.txt).

Sources (read with `git show`, no checkout):
  * own-video protocol (labels, t_on, windows, per-table LOPO, C by inner leave-one-rep-out):
    scripts/run_stage16_scene_motion.py @ e41e72c (claude/stage16-scene-motion); results/stage16_scene_motion/
    inventory.csv and t_on_manual.csv at the same commit.
  * EgoIntent units, labels, folds, LogReg, bootstrap: scripts/stage18_scene_motion.py (this branch).

CPU only (2 threads). Subcommands: speed, extract, eval_b, table, selftest.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage18_scene_motion as S18  # noqa: E402

RES = ROOT / "results/stage19_temporal"
WORK = Path("/media/dhqg/d3/vlia_outputs/stage19")
FEAT = WORK / "feat"
MODEL = "MCG-NJU/videomae-base-finetuned-ssv2"
OWN = Path("/media/dhqg/d3/datasets/own_libero10")
S16_EMB = Path("/media/dhqg/d3/vlia_outputs/stage16/emb")
S16_COMMIT = "e41e72c"
TABLES = {"A": (0, 1, 7), "B": (4, 6), "C": (2, 8), "D": (3, 9), "E": (5,)}
SHARED = ("A", "B", "C", "D")
OWN_P = (25, 50, 100)
EGO_P = (50, 100)
NF = 16
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
B = 10000
log = S18.log


def state(line: str):
    with (WORK / "STATE.md").open("a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} | {line}\n")


def git_show(path: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{S16_COMMIT}:{path}"], capture_output=True, text=True,
                          check=True).stdout


# ----------------------------------------------------------------------
# own videos: Stage 16 metadata and windows (copied)
# ----------------------------------------------------------------------
def own_meta() -> dict:
    inv = {r["video_id"]: r for r in csv.DictReader(io.StringIO(git_show("results/stage16_scene_motion/inventory.csv")))}
    ton = {r["video_id"]: float(r["t_on_manual"]) for r in
           csv.DictReader(io.StringIO(git_show("results/stage16_scene_motion/t_on_manual.csv")))}
    out = {}
    for r in csv.DictReader((OWN / "labels_all.csv").open()):
        v = r["video_id"]
        dur = float(inv[v]["duration_s"])
        out[v] = {"video_id": v, "table": r["scene"], "task": int(r["task"]), "person": r["person"], "rep": int(r["rep"]),
                  "t_on": ton[v], "dur": dur, "t_end": dur - 1.0, "file": str(OWN / r["file"])}
    return dict(sorted(out.items()))


def own_window(m: dict, p: int):
    """Stage 16 query_p observed window [t_on, t_on + p% (t_end - t_on)] in seconds."""
    return m["t_on"], m["t_on"] + p / 100 * (m["t_end"] - m["t_on"])


# ----------------------------------------------------------------------
# EgoIntent windows (match Stage 18: p50 = up to cached frame 4 of 8, p100 = last frame)
# ----------------------------------------------------------------------
def ego_end_frame(n: int, p: int) -> int:
    idx8 = np.linspace(0, n - 1, num=min(8, n), dtype=int)
    if len(idx8) < 8:  # Stage 18 cache repeated frames to 8 (linspace-round over the decoded frames)
        idx8 = idx8[np.linspace(0, len(idx8) - 1, 8, dtype=np.float32).round().astype(int)]
    return int(idx8[3]) if p == 50 else int(idx8[-1])


# ----------------------------------------------------------------------
# decoding and model
# ----------------------------------------------------------------------
def prep(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    h, w = frame_bgr.shape[:2]
    s = 224 / min(h, w)
    img = cv2.resize(frame_bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    y0, x0 = (h - 224) // 2, (w - 224) // 2
    img = img[y0:y0 + 224, x0:x0 + 224, ::-1].astype(np.float32) / 255.0
    return ((img - MEAN) / STD).transpose(2, 0, 1)  # (3, 224, 224)


def read_frames(path: str, wanted: list[int]) -> dict:
    """Sequential decode; returns {frame index: preprocessed frame} for the wanted indices (clamped to last)."""
    import cv2

    cap = cv2.VideoCapture(path)
    need, out, i, last = set(wanted), {}, 0, None
    top = max(wanted)
    while i <= top:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            out[i] = prep(fr)
        last = (i, fr)
        i += 1
    cap.release()
    for w in wanted:  # indices past the real end -> last decoded frame
        if w not in out:
            out[w] = prep(last[1])
    return out


def clip16(idxs) -> np.ndarray:
    return np.round(np.linspace(idxs[0], idxs[1], NF)).astype(int)


class Encoder:
    def __init__(self):
        import torch
        from transformers import VideoMAEForVideoClassification

        self.torch = torch
        self.m = VideoMAEForVideoClassification.from_pretrained(MODEL).eval()
        # transformers 5.5 does not map the checkpoint's q_bias / v_bias (qkv_bias, key bias fixed at 0) and leaves
        # query/key/value biases randomly initialised; copy them from the checkpoint by hand.
        from huggingface_hub import hf_hub_download

        sd = torch.load(hf_hub_download(MODEL, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        with torch.no_grad():
            for i, layer in enumerate(self.m.videomae.encoder.layer):
                att = layer.attention.attention
                pre = f"videomae.encoder.layer.{i}.attention.attention."
                att.query.bias.copy_(sd[pre + "q_bias"])
                att.key.bias.zero_()
                att.value.bias.copy_(sd[pre + "v_bias"])

    def logits(self, clips: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.m(pixel_values=self.torch.from_numpy(np.ascontiguousarray(clips))).logits.numpy()

    def __call__(self, clips: np.ndarray) -> np.ndarray:
        """clips (b, 16, 3, 224, 224) -> L2-normalised fc_norm(mean token), (b, 768)."""
        with self.torch.inference_mode():
            x = self.torch.from_numpy(np.ascontiguousarray(clips))
            seq = self.m.videomae(pixel_values=x).last_hidden_state
            f = self.m.fc_norm(seq.mean(1)).double().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)


def units():
    """[(unit key, video path, {variant: frame indices})], own videos first, then EgoIntent kept steps."""
    import cv2

    out = []
    for v, m in own_meta().items():
        fps = cv2.VideoCapture(m["file"]).get(cv2.CAP_PROP_FPS)
        var = {"S": np.zeros(NF, int)}
        for p in OWN_P:
            a, b = own_window(m, p)
            var[f"V{p}"] = clip16((round(a * fps), round(b * fps)))
        out.append((f"own/{v}", m["file"], var))
    rows, _, _, _, _, _ = S18.ego_data()
    for i, r in enumerate(rows):
        n = int(cv2.VideoCapture(r["video_path"]).get(cv2.CAP_PROP_FRAME_COUNT))
        var = {"S": np.zeros(NF, int)}
        for p in EGO_P:
            var[f"V{p}"] = clip16((0, ego_end_frame(n, p)))
        out.append((f"ego/{i:04d}", r["video_path"], var))
    return out


def encode_unit(enc, path, var, variants):
    wanted = sorted({int(j) for k in variants for j in var[k]})
    fr = read_frames(path, wanted)
    clips = np.stack([np.stack([fr[int(j)] for j in var[k]]) for k in variants])
    return dict(zip(variants, enc(clips)))


def cmd_speed(_args) -> int:
    enc = Encoder()
    us = units()
    ego = [u for u in us if u[0].startswith("ego/")][:5]
    t0 = time.time()
    for key, path, var in ego:
        encode_unit(enc, path, var, ["S", "V50", "V100"])
    per = (time.time() - t0) / len(ego)
    n_ego, n_own = sum(u[0].startswith("ego/") for u in us), sum(u[0].startswith("own/") for u in us)
    msg = (f"speed: {per:.1f} s per EgoIntent step for 3 clips (S, V50, V100) -> ~{per / 3:.1f} s/clip; "
           f"own {n_own} clips x 4 ~ {n_own * 4 * per / 3 / 60:.0f} min; ego {n_ego} x (S+V50) ~ "
           f"{n_ego * 2 * per / 3 / 3600:.2f} h, +V100 ~ {n_ego * per / 3 / 3600:.2f} h")
    log(msg)
    state(msg)
    return 0


def cmd_extract(args) -> int:
    """args.part: own | ego50 | ego100. Per-unit npz, *.tmp then rename; existing variants are skipped."""
    enc = Encoder()
    us = [u for u in units() if u[0].startswith("own/" if args.part == "own" else "ego/")]
    variants = {"own": ["S", "V25", "V50", "V100"], "ego50": ["S", "V50"], "ego100": ["V100"]}[args.part]
    t0, done = time.time(), 0
    for j, (key, path, var) in enumerate(us):
        fp = FEAT / f"{key}.npz"
        old = dict(np.load(fp)) if fp.exists() else {}
        todo = [k for k in variants if k not in old]
        if not todo:
            continue
        old.update(encode_unit(enc, path, var, todo))
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_name(fp.name + ".tmp.npz")
        np.savez(tmp, **old)
        os.replace(tmp, fp)
        done += 1
        if done == 50:
            eta = (time.time() - t0) / 50 * (len(us) - j - 1)
            state(f"extract {args.part}: 50 units in {time.time() - t0:.0f}s, ETA {eta / 60:.0f} min")
    state(f"extract {args.part} done ({len(us)} units, {time.time() - t0:.0f}s new)")
    log(f"extract {args.part} done in {time.time() - t0:.0f}s")
    return 0


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def load_feat(prefix: str, keys, variant: str) -> np.ndarray:
    return np.stack([np.load(FEAT / f"{prefix}/{k}.npz")[variant] for k in keys])


def variant_X(prefix, keys, var, p):
    if var == "S":
        return load_feat(prefix, keys, "S")
    V = load_feat(prefix, keys, f"V{p}")
    return V if var == "S+M" else V - load_feat(prefix, keys, "S")


def choose_C_reps(X, y, reps, k):
    """Stage 16 choose_C (inner leave-one-rep-out, accuracy count), using the Stage 18 balanced LogReg."""
    def score(C):
        hit = 0
        for r in np.unique(reps):
            tr, te = reps != r, reps == r
            hit += int((S18.fit_predict(X[tr], y[tr], X[te], C, k).argmax(1) == y[te]).sum())
        return hit

    grid = list(S18.C_GRID)
    s = {C: score(C) for C in grid}
    best = min(grid, key=lambda C: (-s[C], C))
    ext = None
    if best == grid[0] or best == grid[-1]:
        ext = grid[0] / 10 if best == grid[0] else grid[-1] * 10
        s[ext] = score(ext)
        best = min(s, key=lambda C: (-s[C], C))
    return best, ext


def own_lopo(X, y, persons, reps, k):
    probs, Cs = np.zeros((len(y), k)), {}
    for P in sorted(set(persons)):
        tr, te = persons != P, persons == P
        C, ext = choose_C_reps(X[tr], y[tr], reps[tr], k)
        Cs[P] = (C, ext)
        probs[te] = S18.fit_predict(X[tr], y[tr], X[te], C, k)
    return probs, Cs


def own_eval(X_of, meta, shuffle=False):
    """Per-table LOPO over the shared tables. X_of(video ids) -> features. Returns {video: correct}, choices."""
    hits, ch = {}, {}
    rng = np.random.default_rng(0)
    for t in SHARED:
        vs = [v for v in meta if meta[v]["table"] == t]
        tasks = sorted({meta[v]["task"] for v in vs})
        y = np.array([tasks.index(meta[v]["task"]) for v in vs])
        if shuffle:
            y = y[rng.permutation(len(y))]
        X = X_of(vs).astype(np.float64)
        probs, ch[t] = own_lopo(X, y, np.array([meta[v]["person"] for v in vs]),
                                np.array([meta[v]["rep"] for v in vs]), len(tasks))
        for v, pr, yy in zip(vs, probs.argmax(1), y):
            hits[v] = int(pr == yy)
    return hits, ch


def s16_fplus_p50(meta, p: int = 50, feat: str = "F+") -> dict:
    """Recompute Stage 16 F+ p50 per clip from its cached SigLIP embeddings (Stage 16 features, our LogReg = same
    objective without class weights is Stage 16's; here we keep Stage 16 exactly: unweighted)."""
    def feats_p(v):
        e = np.load(S16_EMB / f"{v}.npz")
        times, z = e["times"], e["z"].astype(np.float64)
        m = meta[v]
        r = min(1.0, m["t_on"] - 0.25)
        ref = np.where(times <= r + 1e-9)[0]
        a, b = own_window(m, p)
        obs = np.where((times >= a - 1e-9) & (times <= b + 1e-9))[0]
        if len(obs) > 16:
            obs = obs[np.round(np.linspace(0, len(obs) - 1, 16)).astype(int)]
        if feat == "M":  # Stage 16 M = mean(z_obs - z_ref)
            return (z[obs] - z[ref].mean(0)).mean(0)
        return np.concatenate([z[ref].mean(0), z[obs].mean(0)])

    orig = S18.LogReg.fit

    def fit_unweighted(self, X, y):  # Stage 16 LogReg had no class weights
        import torch
        import torch.nn.functional as F

        X = torch.as_tensor(X, dtype=torch.float64)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        W = torch.zeros(X.shape[1], self.k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(self.k, dtype=torch.float64, requires_grad=True)
        S18._lbfgs([W, b], lambda: self.C * F.cross_entropy(X @ W + b, yt, reduction="sum") + 0.5 * (W * W).sum())
        self.W, self.b = W.detach(), b.detach()
        return self

    S18.LogReg.fit = fit_unweighted
    try:
        hits, _ = own_eval(lambda vs: np.stack([feats_p(v) for v in vs]), meta)
    finally:
        S18.LogReg.fit = orig
    return hits


def cmd_eval_b(args) -> int:
    t0 = time.time()
    out = {}
    if args.part == "own":
        meta = {v: m for v, m in own_meta().items() if m["table"] in SHARED}
        for var, p in [("S", 0)] + [(f, p) for p in OWN_P for f in ("M", "S+M")]:
            hits, ch = own_eval(lambda vs, var=var, p=p: variant_X("own", vs, var, p), meta)
            out[f"{var}|{p}"] = {"hits": hits, "C": {t: c for t, c in ch.items()}}
            log(f"own B {var} p{p}: acc {np.mean(list(hits.values())):.3f}")
        hits, _ = own_eval(lambda vs: variant_X("own", vs, "S+M", 50), meta, shuffle=True)
        out["shuffled|S+M|50"] = {"hits": hits}
        out["stage16_F+|50"] = {"hits": s16_fplus_p50(meta)}
        log(f"own stage16 F+ p50 recomputed: {np.mean(list(out['stage16_F+|50']['hits'].values())):.3f} (Stage 16: 0.741)")
        # train accuracy (descriptive): per table, C = 100
        tr = {}
        for var, p in (("S", 0), ("M", 50), ("S+M", 50)):
            h = []
            for t in SHARED:
                vs = [v for v in meta if meta[v]["table"] == t]
                tasks = sorted({meta[v]["task"] for v in vs})
                y = np.array([tasks.index(meta[v]["task"]) for v in vs])
                X = variant_X("own", vs, var, p)
                h += list(S18.fit_predict(X, y, X, 100.0, len(tasks)).argmax(1) == y)
            tr[f"{var}|{p}"] = float(np.mean(h))
        out["train_acc"] = tr
    else:
        rows, _, y, vids, outer, _ = S18.ego_data()
        keys = [f"{i:04d}" for i in range(len(y))]
        K = len(S18.KEEP)
        metric = lambda a, b: S18.bal_acc(a, b, K)  # noqa: E731
        ps = [p for p in EGO_P if all((FEAT / f"ego/{k}.npz").exists() and f"V{p}" in np.load(FEAT / f"ego/{k}.npz")
                                      for k in keys)]
        jobs = [("S", 0)] + [(f, p) for p in ps for f in ("M", "S+M")]
        for var, p in jobs:
            X = variant_X("ego", keys, var, p)
            probs, ch = S18.nested(X, y, outer, lambda tr: S18.video_inner(vids[tr]), K, metric,
                                   WORK / f"ego_{var.replace('+', 'p')}_p{p}.json")
            out[f"{var}|{p}"] = {"pred": probs.argmax(1).tolist(), "C": {f: c["C"] for f, c in ch.items()},
                                 "ext": {f: c["ext"] for f, c in ch.items() if c["ext"]}}
            log(f"ego B {var} p{p}: bal_acc {metric(y, probs.argmax(1)):.3f} ({time.time() - t0:.0f}s)")
        yp = y[np.random.default_rng(0).permutation(len(y))]
        probs, _ = S18.nested(variant_X("ego", keys, "S+M", 50), yp, outer, lambda tr: S18.video_inner(vids[tr]), K,
                              metric, WORK / "ego_shuffled_SpM_p50.json")
        out["shuffled|S+M|50"] = {"pred": probs.argmax(1).tolist(), "y": yp.tolist()}
        tr = {}
        for v, p in (("S", 0), ("M", 50), ("S+M", 50)):
            X = variant_X("ego", keys, v, p)
            tr[f"{v}|{p}"] = S18.bal_acc(y, S18.fit_predict(X, y, X, 100.0, K).argmax(1), K)
        out["train_acc"] = tr
    S18.atomic_json(WORK / f"eval_{args.part}.json", out)
    state(f"eval_b {args.part} done in {time.time() - t0:.0f}s")
    return 0


def cmd_posthoc_p100(_args) -> int:
    """Post-hoc (not pre-registered, asked by Trang after seeing results): own videos, B S+M p100 - Stage 16 F+ p100,
    paired on the same 81 clips, bootstrap of clips within table, 10,000, seed 0."""
    meta = {v: m for v, m in own_meta().items() if m["table"] in SHARED}
    vs = sorted(meta)
    E = json.loads((WORK / "eval_own.json").read_text())
    b = np.array([E["S+M|100"]["hits"][v] for v in vs], float)
    h16 = s16_fplus_p50(meta, p=100)
    s = np.array([h16[v] for v in vs], float)
    idx = S18.strat_boot_idx([meta[v]["table"] for v in vs], seed=0)
    d = (b - s)[idx].mean(1)
    res = {"B_SpM_p100": float(b.mean()), "stage16_Fplus_p100_recomputed": float(s.mean()),
           "stage16_Fplus_p100_committed": 0.7283950617283951, "diff": float((b - s).mean()),
           "ci": S18.ci(d), "n_both_right": int(((b == 1) & (s == 1)).sum()), "n_B_only": int(((b == 1) & (s == 0)).sum()),
           "n_S16_only": int(((b == 0) & (s == 1)).sum()), "n_both_wrong": int(((b == 0) & (s == 0)).sum())}
    S18.atomic_json(WORK / "posthoc_p100.json", res)
    print(json.dumps(res, indent=1))
    state(f"posthoc p100 (own): diff {res['diff']:+.3f} CI [{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}]")
    return 0


def cmd_posthoc_p25(_args) -> int:
    """Post-hoc (not pre-registered, asked by Trang after seeing the data): own videos p25, B vs Stage 16 for S+M (vs
    F+) and M (vs M), paired on the same 81 clips, bootstrap of clips within table, 10,000, seed 0."""
    meta = {v: m for v, m in own_meta().items() if m["table"] in SHARED}
    vs = sorted(meta)
    E = json.loads((WORK / "eval_own.json").read_text())
    idx = S18.strat_boot_idx([meta[v]["table"] for v in vs], seed=0)
    res = {}
    for bkey, s16feat, committed in (("S+M|25", "F+", 0.7160493827160493), ("M|25", "M", 0.7530864197530864)):
        b = np.array([E[bkey]["hits"][v] for v in vs], float)
        h16 = s16_fplus_p50(meta, p=25, feat=s16feat)
        s = np.array([h16[v] for v in vs], float)
        d = (b - s)[idx].mean(1)
        res[f"B {bkey} vs Stage16 {s16feat}|25"] = {
            "B": float(b.mean()), "stage16_recomputed": float(s.mean()), "stage16_committed": committed,
            "diff": float((b - s).mean()), "ci": S18.ci(d), "n_both_right": int(((b == 1) & (s == 1)).sum()),
            "n_B_only": int(((b == 1) & (s == 0)).sum()), "n_S16_only": int(((b == 0) & (s == 1)).sum()),
            "n_both_wrong": int(((b == 0) & (s == 0)).sum())}
    S18.atomic_json(WORK / "posthoc_p25.json", res)
    print(json.dumps(res, indent=1))
    state("posthoc p25 (own): " + "; ".join(f"{k}: {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
                                            for k, r in res.items()))
    return 0


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def cmd_table(_args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc, rules, cmp_rows = [], [], []
    ci = S18.ci
    # --- own videos
    if (WORK / "eval_own.json").exists():
        E = json.loads((WORK / "eval_own.json").read_text())
        meta = {v: m for v, m in own_meta().items() if m["table"] in SHARED}
        vs = sorted(meta)
        strata = [meta[v]["table"] for v in vs]
        idx = S18.strat_boot_idx(strata, seed=0)
        chance = float(np.mean([1 / len(TABLES[meta[v]["table"]]) for v in vs]))
        H = {k: np.array([d["hits"][v] for v in vs], float) for k, d in E.items() if "hits" in d}
        for k, h in H.items():
            bb = h[idx].mean(1)
            parts = k.split("|")
            acc.append({"dataset": "own", "method": "B" if not k.startswith(("stage16", "shuffled")) else parts[0],
                        "variant": parts[-2], "p": int(parts[-1]), "n": len(vs), "acc": float(h.mean()), "bal_acc": "",
                        "macro_f1": "", "ci_lo": ci(bb)[0], "ci_hi": ci(bb)[1], "chance": chance})
        S, SM, M = H["S|0"], H["S+M|50"], H["M|50"]
        d2 = (SM - S)[idx].mean(1)
        d4 = (SM - H["stage16_F+|50"])[idx].mean(1)
        rules.append(("own", S.mean() >= 0.8, ((SM - S).mean(), *ci(d2)), (M.mean(), ci(M[idx].mean(1))[0], chance),
                      ((SM - H["stage16_F+|50"]).mean(), *ci(d4))))
    # --- EgoIntent
    if (WORK / "eval_ego.json").exists():
        E = json.loads((WORK / "eval_ego.json").read_text())
        _, _, y, vids, _, _ = S18.ego_data()
        K = len(S18.KEEP)
        uv = np.unique(vids)
        gid = np.searchsorted(uv, vids)
        W = S18.video_boot_weights(uv, seed=0)
        boots = {}
        for k, d in E.items():
            if "pred" not in d:
                continue
            yy = np.array(d.get("y", y))
            yh = np.array(d["pred"])
            bb = S18.boot_bal(yy, yh, gid, W, K)
            boots[k] = bb
            parts = k.split("|")
            acc.append({"dataset": "EgoIntent", "method": "B" if not k.startswith("shuffled") else "shuffled",
                        "variant": parts[-2], "p": int(parts[-1]), "n": len(yy), "acc": float(np.mean(yh == yy)),
                        "bal_acc": S18.bal_acc(yy, yh, K), "macro_f1": S18.macro_f1(yy, yh, K), "ci_lo": ci(bb)[0],
                        "ci_hi": ci(bb)[1], "chance": 1 / K})
        s18 = np.load(S18.WORK / "ego_oof.npz")
        y18 = s18["S+M|50"].argmax(1)
        b18 = S18.boot_bal(y, y18, gid, W, K)
        acc.append({"dataset": "EgoIntent", "method": "stage18", "variant": "S+M", "p": 50, "n": len(y),
                    "acc": float(np.mean(y18 == y)), "bal_acc": S18.bal_acc(y, y18, K),
                    "macro_f1": S18.macro_f1(y, y18, K), "ci_lo": ci(b18)[0], "ci_hi": ci(b18)[1], "chance": 1 / K})
        bal = {r["variant"] + "|" + str(r["p"]): r["bal_acc"] for r in acc if r["dataset"] == "EgoIntent" and r["method"] == "B"}
        d2 = boots["S+M|50"] - boots["S|0"]
        d4 = boots["S+M|50"] - b18
        rules.append(("EgoIntent", bal["S|0"] >= 0.8, (bal["S+M|50"] - bal["S|0"], *ci(d2)),
                      (bal["M|50"], ci(boots["M|50"])[0], 1 / K),
                      (bal["S+M|50"] - S18.bal_acc(y, y18, K), *ci(d4))))
        cm = np.zeros((K, K), int)
        np.add.at(cm, (y, np.array(E["S+M|50"]["pred"])), 1)
        with (RES / "confusion_ego_p50.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + list(S18.KEEP))
            for i, c in enumerate(S18.KEEP):
                w.writerow([c] + cm[i].tolist())
    with (RES / "accuracy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(acc[0]))
        w.writeheader()
        w.writerows(acc)

    def verdict(r):
        r1, r2, r3 = r[1], r[2][1] > 0, r[3][1] > r[3][2]
        if r1:
            return "Scene suffices; cannot claim action understanding."
        return "Evidence the model uses action, not only scene." if (r2 and r3) else "Insufficient evidence."

    lines = ["# Stage 19 rules (pre-registered)", "", "| dataset | rule | value | 95% CI | verdict |", "|---|---|---|---|---|"]
    for r in rules:
        lines += [f"| {r[0]} | R1 S >= 0.80 | | | {'HOLDS' if r[1] else 'does not hold'} |",
                  f"| {r[0]} | R2 (S+M p50) - S | {r[2][0]:+.3f} | [{r[2][1]:+.3f}, {r[2][2]:+.3f}] | {'PASS' if r[2][1] > 0 else 'FAIL'} |",
                  f"| {r[0]} | R3 M p50 > chance ({r[3][2]:.3f}) | {r[3][0]:.3f} | lo {r[3][1]:.3f} | {'PASS' if r[3][1] > r[3][2] else 'FAIL'} |",
                  f"| {r[0]} | R4 B S+M p50 - Stage 18/16 S+M p50 | {r[4][0]:+.3f} | [{r[4][1]:+.3f}, {r[4][2]:+.3f}] | {'PASS' if r[4][1] > 0 else 'FAIL'} |"]
    lines += [""] + [f"- {r[0]}: {verdict(r)}" for r in rules]
    (RES / "rules.md").write_text("\n".join(lines) + "\n")
    # comparison table
    s18acc = {(r["dataset"], r["feature"], int(r["p"])): r for r in csv.DictReader((S18.RES / "accuracy_ego.csv").open())}
    s16 = {(r["feature"], int(r["p"])): r for r in S18.s16_rows()}

    def c(a, lo, hi):
        return f"{100 * float(a):.1f} [{100 * float(lo):.1f}, {100 * float(hi):.1f}]"

    md = ["# Stage 19 vs Stage 18: scene / motion / both (%, 95% CI)", "",
          "| dataset | features | metric | chance | S | M p50 | S+M p50 | M p100 | S+M p100 |", "|---|---|---|---:|---|---|---|---|---|"]
    if any(r["dataset"] == "own" for r in acc):
        g = lambda f, p: c(s16[(f, p)]["acc"], s16[(f, p)]["ci_lo"], s16[(f, p)]["ci_hi"])  # noqa: E731
        md.append(f"| own videos | Stage 16 SigLIP per frame (S+M = F+) | acc | 44.4 | {g('S', 0)} | {g('M', 50)} | {g('F+', 50)} | "
                  f"{g('M', 100)} | {g('F+', 100)} |")
        bo = {(r["variant"], r["p"]): r for r in acc if r["dataset"] == "own" and r["method"] == "B"}
        h = lambda f, p: c(bo[(f, p)]["acc"], bo[(f, p)]["ci_lo"], bo[(f, p)]["ci_hi"]) if (f, p) in bo else "-"  # noqa: E731
        md.append(f"| own videos | Stage 19 B VideoMAE-SSv2 | acc | 44.4 | {h('S', 0)} | {h('M', 50)} | {h('S+M', 50)} | "
                  f"{h('M', 100)} | {h('S+M', 100)} |")
    if any(r["dataset"] == "EgoIntent" for r in acc):
        g = lambda f, p: c(s18acc[("EgoIntent", f, p)]["bal_acc"], s18acc[("EgoIntent", f, p)]["ci_lo"],  # noqa: E731
                           s18acc[("EgoIntent", f, p)]["ci_hi"])
        md.append(f"| EgoIntent | Stage 18 SigLIP per frame | bal_acc | 10.0 | {g('S', 0)} | {g('M', 50)} | {g('S+M', 50)} | "
                  f"{g('M', 100)} | {g('S+M', 100)} |")
        be = {(r["variant"], r["p"]): r for r in acc if r["dataset"] == "EgoIntent" and r["method"] == "B"}
        h = lambda f, p: c(be[(f, p)]["bal_acc"], be[(f, p)]["ci_lo"], be[(f, p)]["ci_hi"]) if (f, p) in be else "-"  # noqa: E731
        md.append(f"| EgoIntent | Stage 19 B VideoMAE-SSv2 | bal_acc | 10.0 | {h('S', 0)} | {h('M', 50)} | {h('S+M', 50)} | "
                  f"{h('M', 100)} | {h('S+M', 100)} |")
    md += ["", "Own videos: Stage 16 S = mean of idle frames; Stage 19 S = first frame repeated 16x (static clip)."]
    md += [f"- {r[0]}: {verdict(r)}" for r in rules]
    sh = [r for r in acc if r["method"] == "shuffled"]
    md += [f"- shuffled-label check ({r['dataset']}, S+M p50): {100 * float(r['bal_acc'] or r['acc']):.1f}" for r in sh]
    (RES / "bang_so_sanh.md").write_text("\n".join(md) + "\n")
    # curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, ds, m in ((ax[0], "own", "acc"), (ax[1], "EgoIntent", "bal_acc")):
        rr = [r for r in acc if r["dataset"] == ds and r["method"] == "B"]
        if not rr:
            continue
        ps = sorted({r["p"] for r in rr if r["p"]})
        S = [r for r in rr if r["variant"] == "S"][0][m]
        a.plot(ps, [S] * len(ps), marker="o", label="S (static clip)")
        for v in ("M", "S+M"):
            a.plot(ps, [[r for r in rr if r["variant"] == v and r["p"] == p][0][m] for p in ps], marker="o", label=v)
        a.axhline(rr[0]["chance"], color="k", lw=0.6, label="chance")
        a.set(title=f"{ds} - VideoMAE-SSv2 ({m})", xlabel="observed p (%)", ylim=(0, 1.02), xticks=ps)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "curve.png", dpi=120)
    print("\n".join(md))
    print("\n".join(lines))
    state("table done")
    return 0


# ----------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------
def cmd_selftest(_args) -> int:
    # static clip: S does not depend on frame order or on p (same 16 indices = frame 0)
    for u in units()[:3] + [x for x in units() if x[0].startswith("ego/")][:3]:
        assert (u[2]["S"] == 0).all()
        for k, v in u[2].items():
            assert len(v) == NF and (np.diff(v) >= 0).all(), (u[0], k)
    # EgoIntent windows match Stage 18 (frame 4 of 8 for p50) and are causal
    assert ego_end_frame(71, 50) == int(np.linspace(0, 70, 8, dtype=int)[3]) and ego_end_frame(71, 100) == 70
    assert ego_end_frame(5, 50) <= ego_end_frame(5, 100) <= 4
    # own windows: start at t_on, causal (end <= t_end)
    m = next(iter(own_meta().values()))
    a, b = own_window(m, 50)
    assert a == m["t_on"] and b <= m["t_end"]
    # own LOPO: test person never in training
    rng = np.random.default_rng(0)
    persons = np.repeat(["p1", "p2", "p3"], 6)
    reps = np.tile([1, 2, 3, 1, 2, 3], 3)
    y = np.tile([0, 0, 0, 1, 1, 1], 3)
    X = rng.normal(size=(18, 4)) + y[:, None] * 3
    seen = []
    orig = S18.fit_predict

    def spy(Xtr, ytr, Xte, C, k):
        seen.append(len(Xtr))
        return orig(Xtr, ytr, Xte, C, k)

    S18.fit_predict = spy
    try:
        probs, _ = own_lopo(X, y, persons, reps, 2)
    finally:
        S18.fit_predict = orig
    assert max(seen) <= 12 and np.mean(probs.argmax(1) == y) > 0.8
    # EgoIntent: no locked test video; scene folds video-pure (Stage 18 data)
    _, _, ye, vids, outer, _ = S18.ego_data()
    assert not (set(vids) & S18.locked_uids())
    for f in np.unique(outer):
        assert not (set(vids[outer == f]) & set(vids[outer != f]))
    log("selftest PASS")
    state("selftest PASS")
    return 0


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["speed", "extract", "eval_b", "table", "selftest", "posthoc_p100", "posthoc_p25"])
    ap.add_argument("--part", default="own", choices=["own", "ego50", "ego100", "ego"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    return {"speed": cmd_speed, "extract": cmd_extract, "eval_b": cmd_eval_b, "table": cmd_table,
            "selftest": cmd_selftest, "posthoc_p100": cmd_posthoc_p100, "posthoc_p25": cmd_posthoc_p25}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
