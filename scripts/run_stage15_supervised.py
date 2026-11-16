#!/usr/bin/env python3
"""Stage 15: supervised early intention recognition (LIBERO-10 now, Trang's clips later).

Subcommands: selftest | extract | cv | plan | analyze | human.
Pre-registration: results/prereg_stage15_supervised_early.md.
Rollouts run through scripts/libero_goal_ablation.py (conditions E2E-sup-p25 / E2E-sup-p50,
--e2e-tokens results/stage15_supervised_early/e2e_sup_tokens.json).
Models are written in torch (no scikit-learn in the lab environments; prereg Part 2).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import run_stage12_e2e as S12  # noqa: E402

N_TASKS = 10
N_FRAMES = 8
STEP_S = 0.25  # 4 frames per second
DS_FPS = 10.0
EVAL_P = (0, 10, 25, 50, 75, 100)
TRAIN_P = (10, 25, 50, 75, 100)
SELECT_P = (25, 50, 75, 100)
ROBOT_P = (25, 50)
N_OUTER, N_INNER = 5, 4
N_UNIT = 30
MODELS = ("logreg", "svm-lin", "svm-rbf", "tcnn")  # this order also breaks ties
GRIDS = {
    "svm-lin": [{"C": c} for c in (0.1, 1, 10, 100, 1000)],
    "svm-rbf": [{"C": c, "g": g} for c in (0.1, 1, 10, 100, 1000) for g in (0.25, 1, 4, 16)],
    "logreg": [{"C": c} for c in (0.01, 0.1, 1, 10, 100)],
    "tcnn": [{}],
}
TCNN_SEEDS = (0, 1, 2)
CAMS = {"agent": "observation.images.image", "wrist": "observation.images.image2"}
ENCODER = S12.ENCODER
WORK = Path("/media/dhqg/d3/vlia_outputs/stage15")
RES = ROOT / "results/stage15_supervised_early"
EVAL = S12.EVAL
TABLE = S12.TABLE
VLIA = S12.VLIA
STAGE12_REC = ROOT / "results/stage12_end_to_end/recognition.csv"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------
# frames, prefixes, features (numpy only)
# ----------------------------------------------------------------------

def frame_grid(n_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(frame indices, times) of a 4 fps decode of a video with n_frames at fps; the last frame is always kept."""
    last = n_frames - 1
    idx = np.round(np.arange(0.0, last / fps + 1e-9, STEP_S) * fps).astype(int)
    idx = np.unique(np.minimum(idx, last))
    if idx[-1] != last:
        idx = np.r_[idx, last]
    return idx, idx / fps


def sample8(times: np.ndarray, end: float) -> np.ndarray:
    """Positions of 8 frames spread evenly over the frames with time <= end (at least the first frame)."""
    n = max(1, int(np.sum(times <= end + 1e-9)))
    return np.round(np.linspace(0, n - 1, N_FRAMES)).astype(int)


def libero_end(times: np.ndarray, p: int) -> float:
    """Prefix end time for a demo: p% of its duration (p = 0 -> the first frame only)."""
    return p / 100 * float(times[-1])


def human_end(start_s: float, end_s: float, p: int) -> float:
    """Prefix end for a clip: p% of the action part; p = 0 -> the idle part only."""
    return start_s + p / 100 * (end_s - start_s)


def mean_feat(seq: np.ndarray) -> np.ndarray:
    return S12.normalize(seq.mean(axis=-2))


def strat_folds(labels, k: int, seed: int) -> np.ndarray:
    """Fold id per item: within each class, shuffle and deal to folds 0..k-1 in turn."""
    labels = np.asarray(labels)
    out = np.full(len(labels), -1)
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        ids = ids[rng.permutation(len(ids))]
        out[ids] = np.arange(len(ids)) % k
    return out


def val_split(labels, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of a stratified validation part (about frac of each class, at least 1 if a class has >= 2)."""
    labels = np.asarray(labels)
    mask = np.zeros(len(labels), bool)
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        n = int(round(frac * len(ids)))
        if len(ids) >= 2:
            n = min(max(n, 1), len(ids) - 1)
        mask[ids[rng.permutation(len(ids))[:n]]] = True
    return mask


def strat_boot(groups, values, B: int = 10000, seed: int = 0):
    """Mean of values and percentile 95% CI, resampling items with replacement within each group."""
    groups, values = np.asarray(groups), np.asarray(values, float)
    rng = np.random.default_rng(seed)
    total = np.zeros(B)
    for g in np.unique(groups):
        v = values[groups == g]
        total += v[rng.integers(0, len(v), size=(B, len(v)))].sum(axis=1)
    boots = total / len(values)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def strict_hits(scores: np.ndarray, y) -> np.ndarray:
    return np.array([S12.strict_correct(s, int(t)) for s, t in zip(scores, y)], float)


def macro_f1(pred, y, k: int = N_TASKS) -> float:
    pred, y = np.asarray(pred), np.asarray(y)
    f = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def strict_pred(scores: np.ndarray) -> np.ndarray:
    """Arg-max, or -1 when the top is tied (counted wrong everywhere)."""
    out = np.argmax(scores, axis=1)
    top = scores[np.arange(len(scores)), out]
    ties = (scores == top[:, None]).sum(axis=1) > 1
    return np.where(ties, -1, out)


def two_way(scores: np.ndarray, y, a: int = 0, b: int = 1) -> np.ndarray:
    """1 if the true task (a or b) scores strictly higher than the other one."""
    y = np.asarray(y)
    other = np.where(y == a, b, a)
    return (scores[np.arange(len(y)), y] > scores[np.arange(len(y)), other]).astype(float)


def within_scene_hits(scores: np.ndarray, y, cands_of) -> np.ndarray:
    import run_stage13_human as H

    return np.array([H.masked_correct(s, int(t), cands_of(int(t))) for s, t in zip(scores, y)], float)


# ----------------------------------------------------------------------
# models (torch)
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


def _ovr(y, k):
    import torch

    Y = -torch.ones(len(y), k, dtype=torch.float64)
    Y[torch.arange(len(y)), torch.as_tensor(np.asarray(y), dtype=torch.long)] = 1.0
    return Y


class LinearSVM:
    def __init__(self, C: float, k: int = N_TASKS):
        self.C, self.k = C, k

    def fit(self, X, y):
        import torch

        X = torch.as_tensor(X, dtype=torch.float64)
        Y = _ovr(y, self.k)
        W = torch.zeros(X.shape[1], self.k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(self.k, dtype=torch.float64, requires_grad=True)
        _lbfgs([W, b], lambda: self.C * torch.clamp(1 - Y * (X @ W + b), min=0).pow(2).sum() + 0.5 * (W * W).sum())
        self.W, self.b = W.detach(), b.detach()
        return self

    def scores(self, X):
        import torch

        return (torch.as_tensor(X, dtype=torch.float64) @ self.W + self.b).numpy()


def sq_dists(A, B):
    import torch

    return torch.clamp((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, min=0)


class KernelSVM:
    """One-vs-rest RBF SVM, squared hinge, primal over kernel coefficients (prereg Part 2)."""

    def __init__(self, C: float, g: float, k: int = N_TASKS):
        self.C, self.g, self.k = C, g, k

    def fit(self, X, y):
        import torch

        X = torch.as_tensor(X, dtype=torch.float64)
        self.gamma = self.g / (X.shape[1] * float(X.var()))
        K = torch.exp(-self.gamma * sq_dists(X, X))
        Y = _ovr(y, self.k)
        A = torch.zeros(len(X), self.k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(self.k, dtype=torch.float64, requires_grad=True)

        def loss():
            KA = K @ A
            return self.C * torch.clamp(1 - Y * (KA + b), min=0).pow(2).sum() + 0.5 * (A * KA).sum()

        _lbfgs([A, b], loss)
        self.X, self.A, self.b = X, A.detach(), b.detach()
        return self

    def scores(self, X):
        import torch

        X = torch.as_tensor(X, dtype=torch.float64)
        return (torch.exp(-self.gamma * sq_dists(X, self.X)) @ self.A + self.b).numpy()


class LogReg:
    def __init__(self, C: float, k: int = N_TASKS):
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


def _tcnn_net(d: int, k: int):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Conv1d(d, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Conv1d(256, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.3))
            self.head = nn.Linear(256, k)

        def forward(self, x):  # x (n, T, d)
            return self.head(self.body(x.transpose(1, 2)).amax(dim=2))

    return Net()


class TCNN:
    """Temporal CNN, 3 seeds averaged; early stopping on a stratified 20% validation split of the training demos."""

    def __init__(self, k: int = N_TASKS, epochs: int = 100, patience: int = 10, val_seed: int = 0):
        self.k, self.epochs, self.patience, self.val_seed = k, epochs, patience, val_seed

    def fit(self, X, y, groups, group_label):
        """X (n, 8, d) sequences; groups = demo id per row; group_label = {demo: task} for the stratified split."""
        import torch
        import torch.nn.functional as F

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        demos = np.array(sorted(group_label))
        vmask_demo = val_split([group_label[d] for d in demos], 0.2, self.val_seed)
        vdemos = set(demos[vmask_demo].tolist())
        vrow = np.array([g in vdemos for g in groups])
        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=dev)
        tr, va = np.where(~vrow)[0], np.where(vrow)[0]
        self.nets, self.best_epochs = [], []
        for s in TCNN_SEEDS:
            torch.manual_seed(s)
            net = _tcnn_net(X.shape[2], self.k).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            rng = np.random.default_rng(s)
            best, best_state, best_ep, bad = (-1.0, float("inf")), None, 0, 0
            for ep in range(self.epochs):
                net.train()
                perm = tr[rng.permutation(len(tr))]
                for i in range(0, len(perm), 64):
                    j = torch.as_tensor(perm[i:i + 64], device=dev)
                    opt.zero_grad()
                    F.cross_entropy(net(Xt[j]), yt[j]).backward()
                    opt.step()
                if len(va) == 0:  # no class has 2 training videos (tiny clip folds): train all epochs, keep the last
                    best_state, best_ep = {k: v.detach().clone() for k, v in net.state_dict().items()}, ep
                    continue
                net.eval()
                with torch.no_grad():
                    out = net(Xt[va])
                    vacc = float((out.argmax(1) == yt[va]).float().mean())
                    vloss = float(F.cross_entropy(out, yt[va]))
                if best_state is None or vacc > best[0] or (vacc == best[0] and vloss < best[1]):
                    best, best_state, best_ep, bad = (vacc, vloss), {k: v.detach().clone() for k, v in net.state_dict().items()}, ep, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
            self.best_epochs.append(best_ep)
        self.dev = dev
        return self

    def scores(self, X):
        import torch

        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
            return np.mean([torch.softmax(n(Xt), 1).double().cpu().numpy() for n in self.nets], axis=0)


def make_model(name: str, hp: dict, k: int, val_seed: int = 0):
    if name == "svm-lin":
        return LinearSVM(hp["C"], k)
    if name == "svm-rbf":
        return KernelSVM(hp["C"], hp["g"], k)
    if name == "logreg":
        return LogReg(hp["C"], k)
    return TCNN(k, val_seed=val_seed)


# ----------------------------------------------------------------------
# cross-validation (generic over videos)
# ----------------------------------------------------------------------

def build(seqs: dict, vids, label: dict, ps):
    """Rows for every (video, p): mean features, sequences, labels, video ids, p."""
    rows = [(v, p) for v in vids for p in ps]
    S = np.stack([seqs[v][p] for v, p in rows]) if rows else np.zeros((0, N_FRAMES, 1))
    return {"mean": mean_feat(S), "seq": S, "y": np.array([label[v] for v, _ in rows]),
            "vid": np.array([v for v, _ in rows]), "p": np.array([p for _, p in rows])}


def fit_score(name, hp, tr, te, label, k, val_seed):
    m = make_model(name, hp, k, val_seed)
    if name == "tcnn":
        m.fit(tr["seq"], tr["y"], tr["vid"], {v: label[v] for v in set(tr["vid"].tolist())})
        return m, m.scores(te["seq"])
    m.fit(tr["mean"], tr["y"])
    return m, m.scores(te["mean"])


def run_cv(seqs: dict, label: dict, outer: dict, inner_fn, k: int = N_TASKS, models=MODELS, grids=GRIDS, verbose=True):
    """Nested CV. outer = {video: outer fold}. inner_fn(train videos) -> {video: inner fold}.
    Returns out-of-fold scores {(model, video, p): scores} plus per-fold choices."""
    vids = sorted(outer)
    folds = sorted(set(outer.values()))
    oof, choices = {}, []
    for f in folds:
        test = [v for v in vids if outer[v] == f]
        train = [v for v in vids if outer[v] != f]
        assert not set(test) & set(train)
        inner = inner_fn(train)
        assert set(inner) == set(train), "inner folds must use training videos only"
        inner_scores = {m: [] for m in models}
        for m in models:
            for gi, hp in enumerate(grids[m]):
                hits = []
                for j in sorted(set(inner.values())):
                    itr = build(seqs, [v for v in train if inner[v] != j], label, TRAIN_P)
                    ite = build(seqs, [v for v in train if inner[v] == j], label, SELECT_P)
                    _, s = fit_score(m, hp, itr, ite, label, k, val_seed=1000 * (f + 1) + j)
                    hits.append(strict_hits(s, ite["y"]))
                inner_scores[m].append(float(np.concatenate(hits).mean()))
        chosen = {}
        for m in models:
            sc = inner_scores[m]
            best = max(range(len(sc)), key=lambda i: (sc[i], -i))  # grid is ordered by C then g: first best wins
            chosen[m] = {"hp": grids[m][best], "inner_acc": sc[best], "grid_acc": sc}
        primary = max(models, key=lambda m: (chosen[m]["inner_acc"], -models.index(m)))
        tr = build(seqs, train, label, TRAIN_P)
        te = build(seqs, test, label, EVAL_P)
        for m in models:
            model, s = fit_score(m, chosen[m]["hp"], tr, te, label, k, val_seed=1000 * (f + 1) + 99)
            if m == "tcnn":
                chosen[m]["best_epochs"] = model.best_epochs
            for i, (v, p) in enumerate(zip(te["vid"], te["p"])):
                oof[(m, v, int(p))] = s[i]
        for i, (v, p) in enumerate(zip(te["vid"], te["p"])):
            oof[("primary", v, int(p))] = oof[(primary, v, int(p))]
        choices.append({"fold": int(f), "n_train": len(train), "n_test": len(test), "primary": primary,
                        "models": {m: {kk: vv for kk, vv in c.items()} for m, c in chosen.items()}, "test": test})
        if verbose:
            log(f"fold {f}: primary {primary}; " + ", ".join(f"{m} {chosen[m]['hp']} {chosen[m]['inner_acc']:.3f}" for m in models))
    return oof, choices


def leak_check(outer: dict, choices: list) -> list[str]:
    errs = []
    tested = [v for c in choices for v in c["test"]]
    if sorted(tested) != sorted(outer):
        errs.append("each video must be tested exactly once")
    for c in choices:
        if set(c["test"]) & {v for v in outer if outer[v] != c["fold"]}:
            errs.append(f"fold {c['fold']}: train/test overlap")
    return errs


def most_frequent_hp(choices, m):
    hs = [json.dumps(c["models"][m]["hp"], sort_keys=True) for c in choices]
    return json.loads(max(sorted(set(hs)), key=hs.count))


# ----------------------------------------------------------------------
# LIBERO data
# ----------------------------------------------------------------------

def video_loc(eps, ep, cam):
    r = eps.loc[ep]
    return (int(r[f"videos/{cam}/chunk_index"]), int(r[f"videos/{cam}/file_index"]),
            float(r[f"videos/{cam}/from_timestamp"]), int(r["length"]))


def demo_key(t: int, d: int) -> str:
    return f"t{t}_d{d}"


def load_demos():
    snap, eps, demos, _others, original = S12.load_meta()
    return snap, eps, demos, original


def siglip():
    import torch

    import extract_ladder_embeddings as X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = X.load_encoder(ENCODER, device)
    return lambda frames: S12.encode(model, processor, frames, device, X)


def cmd_extract(args) -> int:
    import av

    cam = CAMS[args.camera]
    snap, eps, demos, _ = load_demos()
    out = WORK / "feat" / args.camera
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t, v in demos.items():
        for d, ep in enumerate(v):
            if not (out / f"{demo_key(t, d)}.npz").exists():
                jobs.append((t, d, ep))
    log(f"{args.camera}: {sum(map(len, demos.values()))} demos, {len(jobs)} to extract")
    if not jobs:
        return 0
    embed = siglip()
    by_file = {}
    for t, d, ep in jobs:
        chunk, fi, t0, length = video_loc(eps, ep, cam)
        idx, times = frame_grid(length, DS_FPS)
        by_file.setdefault((chunk, fi), []).append((t, d, ep, int(round(t0 * DS_FPS)), idx, times))
    t_start = time.time()
    for n, ((chunk, fi), items) in enumerate(sorted(by_file.items())):
        want = {s + int(i) for _, _, _, s, idx, _ in items for i in idx}
        last = max(want)
        got = {}
        with av.open(str(Path(snap) / f"videos/{cam}/chunk-{chunk:03d}/file-{fi:03d}.mp4")) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for frame in c.decode(s):
                i = int(round(float(frame.pts * s.time_base) * DS_FPS))
                if i in want:
                    got[i] = frame.to_ndarray(format="rgb24")
                if i >= last:
                    break
        missing = want - set(got)
        if missing:
            raise SystemExit(f"STOP: {len(missing)} frames not decoded in {cam} file {fi}")
        for t, d, ep, s0, idx, times in items:
            emb = embed(np.stack([got[s0 + int(i)] for i in idx]))
            np.savez(out / f"{demo_key(t, d)}.npz", emb=emb.astype(np.float32), times=times, frame_index=idx,
                     episode=ep, task=t, demo=d, length=len(idx) and int(idx[-1]) + 1)
        log(f"file {n + 1}/{len(by_file)}: {len(items)} demos, {len(want)} frames ({time.time() - t_start:.0f}s)")
    return 0


def gate_f1(feat: dict, lengths: dict) -> list[str]:
    errs = []
    for k, (emb, times) in feat.items():
        if times[0] != 0 or not (np.diff(times) > 0).all() or not np.isclose(times[-1], (lengths[k] - 1) / DS_FPS):
            errs.append(f"{k}: bad times")
        if np.abs(np.linalg.norm(emb, axis=1) - 1).max() >= 1e-4:
            errs.append(f"{k}: embeddings not unit norm")
    return errs


def load_libero(camera: str):
    """seqs {key: {p: (8, D)}}, label {key: task}, demos meta. Applies gates M1 (via load_meta) and F1."""
    _snap, eps, demos, original = load_demos()
    feat, lengths, label, info = {}, {}, {}, {}
    fdir = WORK / "feat" / camera
    for t, v in demos.items():
        for d, ep in enumerate(v):
            k = demo_key(t, d)
            z = np.load(fdir / f"{k}.npz")
            feat[k] = (z["emb"], z["times"])
            lengths[k] = video_loc(eps, ep, CAMS[camera])[3]
            label[k], info[k] = t, {"task": t, "demo": d, "episode": ep}
    errs = gate_f1(feat, lengths)
    print(f"GATE F1 ({camera}, {len(feat)} demos): {'PASS' if not errs else 'FAIL'}")
    if errs:
        raise SystemExit("STOP: F1 failed: " + "; ".join(errs[:5]))
    seqs = {k: {p: emb[sample8(times, libero_end(times, p))] for p in EVAL_P} for k, (emb, times) in feat.items()}
    return seqs, label, info, {t: len(v) for t, v in demos.items()}, original


def libero_outer(label: dict) -> dict:
    keys = sorted(label)
    f = strat_folds([label[k] for k in keys], N_OUTER, 0)
    return dict(zip(keys, f.tolist()))


def libero_inner_fn(outer: dict, label: dict):
    def fn(train):
        f = outer[train[0]]
        tr = sorted(train)
        g = strat_folds([label[k] for k in tr], N_INNER, 1 + f)
        return dict(zip(tr, g.tolist()))

    return fn


def save_oof(path: Path, oof: dict):
    keys = sorted(oof)
    np.savez(path, model=np.array([k[0] for k in keys]), vid=np.array([k[1] for k in keys]),
             p=np.array([k[2] for k in keys]), scores=np.stack([oof[k] for k in keys]))


def load_oof(path: Path) -> dict:
    z = np.load(path)
    return {(str(m), str(v), int(p)): s for m, v, p, s in zip(z["model"], z["vid"], z["p"], z["scores"])}


def cmd_cv(args) -> int:
    t0 = time.time()
    seqs, label, info, n_demos, _ = load_libero(args.camera)
    outer = libero_outer(label)
    oof, choices = run_cv(seqs, label, outer, libero_inner_fn(outer, label))
    errs = leak_check(outer, choices)
    print(f"GATE L0 ({args.camera}): {'PASS' if not errs else 'FAIL'}")
    if errs:
        raise SystemExit("STOP: L0 failed: " + "; ".join(errs))
    save_oof(WORK / f"oof_{args.camera}.npz", oof)
    (WORK / f"cv_{args.camera}.json").write_text(json.dumps({
        "camera": args.camera, "outer_fold": outer, "info": info, "n_demos": n_demos, "choices": choices,
        "seconds": time.time() - t0}, indent=1))
    log(f"cv {args.camera} done in {time.time() - t0:.0f}s")
    return 0


# ----------------------------------------------------------------------
# accuracy tables (shared by LIBERO and clips)
# ----------------------------------------------------------------------

def acc_rows(oof: dict, label: dict, models, ps, extra: dict) -> list[dict]:
    rows = []
    vids = sorted(label)
    y = np.array([label[v] for v in vids])
    for m in models:
        for p in ps:
            s = np.stack([oof[(m, v, p)] for v in vids])
            h = strict_hits(s, y)
            pt, lo, hi = strat_boot(y, h)
            rows.append({**extra, "model": m, "p": p, "n": len(vids), "accuracy": pt, "ci95_lo": lo, "ci95_hi": hi,
                         "macro_f1": macro_f1(strict_pred(s), y)})
    return rows


def confusion(oof: dict, label: dict, m: str, p: int, k: int = N_TASKS) -> np.ndarray:
    c = np.zeros((k, k), int)
    for v in label:
        c[label[v], int(np.argmax(oof[(m, v, p)]))] += 1
    return c


def write_confusion(path: Path, c: np.ndarray):
    with path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["true\\pred"] + list(range(len(c))))
        for t in range(len(c)):
            w.writerow([t] + c[t].tolist())


def confused_pairs(c: np.ndarray, min_count: int = 1) -> list[tuple[int, int, int]]:
    out = [(int(i), int(j), int(c[i, j])) for i in range(len(c)) for j in range(len(c)) if i != j and c[i, j] >= min_count]
    return sorted(out, key=lambda x: -x[2])


def plot_curve(path: Path, rows: list[dict], refs: dict, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in [r for r in dict.fromkeys(r["model"] for r in rows)]:
        rr = [r for r in rows if r["model"] == m]
        ax.errorbar([r["p"] for r in rr], [100 * r["accuracy"] for r in rr],
                    yerr=[[100 * (r["accuracy"] - r["ci95_lo"]) for r in rr], [100 * (r["ci95_hi"] - r["accuracy"]) for r in rr]],
                    marker="o", capsize=3, label=m, lw=2.2 if m == "primary" else 1.2)
    for name, (xs, ys, style) in refs.items():
        ax.plot(xs, ys, style, label=name, color="gray")
    ax.set_xlabel("% of the video seen (p)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# robot plan and summary (shared)
# ----------------------------------------------------------------------

def libero_units(n_demos: dict) -> list[tuple[int, int]]:
    return [(t, e) for t in range(N_TASKS) for e in range(N_UNIT) if e < n_demos[t]]


def plan_tokens(pred: dict, units, original: dict, prefix: str = "E2E-sup") -> dict:
    """pred[(p, t, e)] = predicted task. Units predicted right reuse I-only; the rest need rollouts."""
    plan = {"tokens": {}, "reuse": {}, "pred": {}, "new": {}}
    for p in ROBOT_P:
        c = f"{prefix}-p{p}"
        plan["tokens"][c] = {f"{t},{e}": original[pred[(p, t, e)]] for t, e in units if pred[(p, t, e)] >= 0}
        plan["pred"][c] = {f"{t},{e}": int(pred[(p, t, e)]) for t, e in units}
        plan["reuse"][c] = [f"{t},{e}" for t, e in units if pred[(p, t, e)] == t]
        plan["new"][c] = [f"{t},{e}" for t, e in units if pred[(p, t, e)] != t]
    return plan


def check_reuse(plan: dict, original: dict, eval_dir: Path = EVAL) -> list[str]:
    errs = []
    for c, keys in plan["reuse"].items():
        for k in keys:
            t, e = map(int, k.split(","))
            r = json.loads((eval_dir / f"episodes/I-only_t{t}_e{e}.json").read_text())
            if r["z_int_task"] != original[t] or r["seed"] != 1000 + e or r["init_state_id"] != e:
                errs.append(f"{c} {k}: I-only episode does not match the unit")
            if plan["tokens"][c][k] != original[t]:
                errs.append(f"{c} {k}: reused unit token is not the true task")
    return errs


def unit_success(plan: dict, c: str, t: int, e: int, eval_dir: Path = EVAL) -> tuple[float, str]:
    k = f"{t},{e}"
    if k in plan["reuse"][c]:
        return float(json.loads((eval_dir / f"episodes/I-only_t{t}_e{e}.json").read_text())["success"]), "reused"
    path = eval_dir / f"episodes/{c}_t{t}_e{e}.json"
    if not path.exists():
        raise SystemExit(f"STOP: missing rollout {path.name}")
    r = json.loads(path.read_text())
    assert r["z_int_task"] == plan["tokens"][c][k] and r["seed"] == 1000 + e and r["init_state_id"] == e, path
    return float(r["success"]), "new"


def robot_tables(plan: dict, units, prefix: str, refs=("I-only", "Stage12 E2E", "I-wrong"), eval_dir: Path = EVAL):
    succ, src = {}, {}
    for p in ROBOT_P:
        c = f"{prefix}-p{p}"
        res = [unit_success(plan, c, t, e, eval_dir) for t, e in units]
        succ[c] = np.array([r[0] for r in res])
        src[c] = {"reused": sum(r[1] == "reused" for r in res), "new": sum(r[1] == "new" for r in res)}
    files = {"I-only": "I-only", "Stage12 E2E": "E2E", "I-wrong": "I-wrong"}
    for name in refs:
        succ[name] = np.array([float(json.loads((eval_dir / f"episodes/{files[name]}_t{t}_e{e}.json").read_text())["success"])
                               for t, e in units])
    summary = []
    for c, s in succ.items():
        k, n = int(s.sum()), len(units)
        lo, hi = G.wilson(k, n)
        row = {"condition": c, "successes": k, "n": n, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi}
        row.update({f"episodes_{kk}": vv for kk, vv in src.get(c, {}).items()})
        summary.append(row)
    paired = []
    for p in ROBOT_P:
        c = f"{prefix}-p{p}"
        for b in refs:
            x = G.paired_bootstrap(succ[c] - succ[b])
            paired.append({"rule": "R4" if (p == 25 and b == "I-wrong") else "descriptive", "comparison": f"{c} - {b}",
                           "delta": x[0], "ci95_lo": x[1], "ci95_hi": x[2], "n": len(units)})
    return succ, summary, paired, src


def write_csv(path: Path, rows):
    S12.write_csv(path, rows)


# ----------------------------------------------------------------------
# LIBERO plan / analyze
# ----------------------------------------------------------------------

def primary_pred(oof: dict, info: dict, p: int) -> dict:
    """(t, d) -> strict top-1 of the primary model at p (-1 on a tie)."""
    out = {}
    for k, v in info.items():
        s = oof[("primary", k, p)]
        out[(v["task"], v["demo"])] = int(strict_pred(s[None])[0])
    return out


def cmd_plan(_) -> int:
    oof = load_oof(WORK / "oof_agent.npz")
    cv = json.loads((WORK / "cv_agent.json").read_text())
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    units = libero_units({int(t): n for t, n in cv["n_demos"].items()})
    pred = {}
    for p in ROBOT_P:
        pp = primary_pred(oof, cv["info"], p)
        pred.update({(p, t, e): pp[(t, e)] for t, e in units})
    plan = plan_tokens(pred, units, original)
    errs = check_reuse(plan, original)
    print(f"GATE reuse ({sum(len(x) for x in plan['reuse'].values())} reused units): {'PASS' if not errs else 'FAIL'}")
    if errs:
        raise SystemExit("STOP: " + "; ".join(errs[:5]))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e2e_sup_tokens.json").write_text(json.dumps(plan, indent=1))
    for c in plan["new"]:
        print(f"{c}: {len(units)} units, reuse I-only {len(plan['reuse'][c])}, new rollouts {len(plan['new'][c])}: {plan['new'][c]}")
    return 0


def stage12_refs() -> dict:
    rows = [r for r in csv.DictReader(STAGE12_REC.open()) if r["context"] == "C2" and r["template"] == "T1"]
    return {int(r["p"]): float(r["accuracy"]) for r in rows}


def rules_libero(oof: dict, label: dict) -> dict:
    vids = sorted(label)
    y = np.array([label[v] for v in vids])
    out = {}
    for rule, p, thr in (("R1", 100, 0.90), ("R2", 50, 0.50)):
        h = strict_hits(np.stack([oof[("primary", v, p)] for v in vids]), y)
        pt, lo, hi = strat_boot(y, h)
        out[rule] = {"pass": lo > thr, "accuracy": pt, "ci95": [lo, hi], "rule": f"accuracy p={p} CI lower bound > {thr:.0%}"}
    v01 = [v for v in vids if label[v] in (0, 1)]
    y01 = np.array([label[v] for v in v01])
    h = two_way(np.stack([oof[("primary", v, 50)] for v in v01]), y01)
    pt, lo, hi = strat_boot(y01, h)
    out["R3"] = {"pass": lo > 0.5, "accuracy": pt, "ci95": [lo, hi], "n": len(v01),
                 "rule": "tasks 0 vs 1, 2-way accuracy p=50 CI lower bound > 50%"}
    return out


def cmd_analyze(args) -> int:
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    cams = [c for c in CAMS if (WORK / f"oof_{c}.npz").exists()]
    acc, cvs, oofs = [], {}, {}
    for cam in cams:
        oofs[cam] = load_oof(WORK / f"oof_{cam}.npz")
        cvs[cam] = json.loads((WORK / f"cv_{cam}.json").read_text())
        label = {k: v["task"] for k, v in cvs[cam]["info"].items()}
        acc += acc_rows(oofs[cam], label, ("primary",) + MODELS, EVAL_P, {"camera": cam})
    write_csv(RES / "accuracy.csv", acc)
    cv = cvs["agent"]
    oof = oofs["agent"]
    label = {k: v["task"] for k, v in cv["info"].items()}
    ref12 = stage12_refs()
    plot_curve(RES / "curve.png", [r for r in acc if r["camera"] == "agent"],
               {"zero-shot F (Stage 12)": (sorted(ref12), [100 * ref12[p] for p in sorted(ref12)], "s--"),
                "chance 10%": (list(EVAL_P), [10] * len(EVAL_P), ":")},
               "LIBERO-10, agent view, out-of-fold (5-fold)")
    conf = {}
    for p in (25, 50):
        conf[p] = confusion(oof, label, "primary", p)
        write_confusion(RES / f"confusion_p{p}.csv", conf[p])
    verdicts = rules_libero(oof, label)
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    units = libero_units({int(t): n for t, n in cv["n_demos"].items()})
    plan = json.loads((RES / "e2e_sup_tokens.json").read_text())
    robot = None
    try:
        succ, summary, paired, src = robot_tables(plan, units, "E2E-sup")
        r4 = next(r for r in paired if r["rule"] == "R4")
        verdicts["R4"] = {"pass": r4["ci95_lo"] > 0, "delta": r4["delta"], "ci95": [r4["ci95_lo"], r4["ci95_hi"]],
                          "rule": "success(E2E-sup, p=25) - success(I-wrong) CI lower bound > 0 (paired_bootstrap)"}
        write_csv(RES / "summary.csv", summary)
        lines = ["# Stage 15 robot: VLIA with the supervised label", "", f"{len(units)} paired units (Stage 12 amendment 1).", "",
                 "| condition | successes / n | rate | Wilson 95% | episodes reused / new |", "|---|---:|---:|---|---|"]
        lines += [f"| {r['condition']} | {r['successes']} / {r['n']} | {r['success_rate']:.3f} | [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] | "
                  f"{r.get('episodes_reused', '')} / {r.get('episodes_new', '')} |" for r in summary]
        lines += ["", "| rule | comparison | delta [95% CI] |", "|---|---|---|"]
        lines += [f"| {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in paired]
        (RES / "robot_summary.md").write_text("\n".join(lines) + "\n")
        robot = {"summary": summary, "paired": paired, "episodes": src}
    except SystemExit as exc:
        print(f"robot part not ready: {exc}")
    names = {"R1": "full video", "R2": "half video", "R3": "not only the scene", "R4": "robot"}
    lines = ["# Stage 15 rules (primary model, agent view, out-of-fold)", "", "| rule | what | value | 95% CI | verdict |", "|---|---|---:|---|---|"]
    for k, v in verdicts.items():
        val = v.get("accuracy", v.get("delta"))
        lines.append(f"| {k} | {names[k]}: {v['rule']} | {val:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {'PASS' if v['pass'] else 'FAIL'} |")
    lines += ["", "Primary model per outer fold: " + ", ".join(f"fold {c['fold']} {c['primary']}" for c in cv["choices"]), ""]
    lines += ["Confused pairs at p=25 (true -> predicted: count): " + ", ".join(f"{a}->{b}: {n}" for a, b, n in confused_pairs(conf[25])), ""]
    (RES / "rules.md").write_text("\n".join(lines))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"git_commit": commit, "encoder": ENCODER, "vlia_checkpoint": VLIA, "token_table": TABLE, "n_demos": cv["n_demos"],
           "outer_fold": {cam: cvs[cam]["outer_fold"] for cam in cams},
           "folds": {cam: [{kk: vv for kk, vv in c.items() if kk != "test"} for c in cvs[cam]["choices"]] for cam in cams},
           "cv_seconds": {cam: cvs[cam]["seconds"] for cam in cams}, "stage12_zero_shot": ref12, "verdicts": verdicts,
           "robot": robot, "analyze_seconds": time.time() - t0}
    if robot:
        secs = [json.loads((EVAL / "episodes/{}_t{}_e{}.json".format(c, *k.split(","))).read_text())["seconds"]
                for c in plan["new"] for k in plan["new"][c]]
        cfg["seconds_per_new_episode"] = float(np.mean(secs)) if secs else None
        cfg["n_new_episodes"] = len(secs)
    (RES / "config.json").write_text(json.dumps(cfg, indent=2))
    for r in acc:
        print(f"{r['camera']:5s} {r['model']:8s} p={r['p']:3d}: {100 * r['accuracy']:5.1f} [{100 * r['ci95_lo']:.1f}; {100 * r['ci95_hi']:.1f}] F1 {r['macro_f1']:.3f}")
    print((RES / "rules.md").read_text())
    if robot:
        print((RES / "robot_summary.md").read_text())
    return 0


# ----------------------------------------------------------------------
# Trang's clips
# ----------------------------------------------------------------------

def human_extract(videos: Path, labels: list[dict], out: Path, embed, rotate_deg: int = 0) -> dict:
    """4 fps features of every labelled clip (resumable). Returns {video_id: meta}."""
    import run_stage13_human as H

    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for r in labels:
        v = r["video_id"]
        m = H.probe(videos / f"{v}.mp4")
        meta[v] = m
        if (out / f"{v}.npz").exists():
            continue
        idx, times = frame_grid(m["n_frames"], m["fps"])
        fr, _, _ = H.decode(videos / f"{v}.mp4", wanted={int(i) for i in idx})
        frames = H.rotate(np.stack([fr[int(i)] for i in idx]), rotate_deg)
        np.savez(out / f"{v}.npz", emb=embed(frames).astype(np.float32), times=times, frame_index=idx)
    return meta


def human_seqs(labels: list[dict], fdir: Path) -> dict:
    seqs = {}
    for r in labels:
        z = np.load(fdir / f"{r['video_id']}.npz")
        emb, times = z["emb"], z["times"]
        seqs[r["video_id"]] = {p: emb[sample8(times, human_end(r["start_s"], r["end_s"], p))] for p in EVAL_P}
    return seqs


def human_rules(oof: dict, label: dict, scene_of) -> dict:
    import run_stage13_human as H

    vids = sorted(label)
    y = np.array([label[v] for v in vids])
    out = {}
    for rule, p, thr in (("R1", 100, 0.90), ("R2", 50, 0.50)):
        h = strict_hits(np.stack([oof[("primary", v, p)] for v in vids]), y)
        pt, lo, hi = strat_boot(y, h)
        out[rule] = {"pass": lo > thr, "accuracy": pt, "ci95": [lo, hi], "rule": f"accuracy p={p} CI lower bound > {thr:.0%}"}
    multi = [v for v in vids if len(H.SCENE_TASKS[scene_of[v]]) > 1]
    ym = np.array([label[v] for v in multi])
    h = within_scene_hits(np.stack([oof[("primary", v, 50)] for v in multi]), ym, lambda t: H.SCENE_TASKS[H.SCENE_OF[t]])
    chance = float(np.mean([1 / len(H.SCENE_TASKS[scene_of[v]]) for v in multi]))
    pt, lo, hi = strat_boot(ym, h)
    out["R3"] = {"pass": lo > chance, "accuracy": pt, "ci95": [lo, hi], "chance": chance, "n": len(multi),
                 "rule": f"within-scene accuracy p=50 CI lower bound > chance {chance:.3f}"}
    return out


def cmd_human(args, embed=None) -> int:
    """Same pipeline on Trang's clips: --step extract | cv | plan | analyze."""
    import run_stage13_human as H

    videos, work = Path(args.videos), Path(args.work)
    res = Path(args.results)
    labels = H.read_labels(Path(args.labels) if args.labels else videos / "labels.csv")
    fdir = work / "feat"
    if args.step == "extract":
        meta = {v: H.probe(videos / f"{v}.mp4") for v in (r["video_id"] for r in labels) if (videos / f"{v}.mp4").exists()}
        errs = H.gate_l1(labels, {v: m["duration"] for v, m in meta.items()}, partial=args.allow_partial)
        print(f"GATE L1: {'PASS' if not errs else 'FAIL'}")
        if errs:
            raise SystemExit("STOP: L1 failed: " + "; ".join(errs[:5]))
        human_extract(videos, labels, fdir, embed or siglip(), args.rotate)
        (work / "extract_meta.json").write_text(json.dumps({"rotate": args.rotate, "clips": meta}, indent=1))
        return 0
    prim = [r for r in labels if r["person"] == H.PRIMARY_PERSON]
    label = {r["video_id"]: r["task"] for r in prim}
    rep = {r["video_id"]: r["rep"] for r in prim}
    scene_of = {r["video_id"]: r["scene"] for r in prim}
    if args.step == "cv":
        t0 = time.time()
        seqs = human_seqs(labels, fdir)
        outer = {v: rep[v] - 1 for v in label}

        def inner_fn(train):
            reps = sorted({rep[v] for v in train})
            return {v: reps.index(rep[v]) for v in train}

        oof, choices = run_cv(seqs, label, outer, inner_fn)
        errs = leak_check(outer, choices)
        print(f"GATE L0 (clips, leave one rep out): {'PASS' if not errs else 'FAIL'}")
        if errs:
            raise SystemExit("STOP: L0 failed")
        save_oof(work / "oof_human.npz", oof)
        extra = {"choices": choices, "seconds": time.time() - t0, "persons": {}, "cross_domain": None}
        # other persons: train on all clips of the other persons with the most frequent choice (descriptive)
        persons = sorted({r["person"] for r in labels})
        if len(persons) > 1:
            lab_all = {r["video_id"]: r["task"] for r in labels}
            for pp in persons:
                te = [r["video_id"] for r in labels if r["person"] == pp]
                tr = [r["video_id"] for r in labels if r["person"] != pp]
                rows = []
                for m in MODELS:
                    _, s = fit_score(m, most_frequent_hp(choices, m), build(seqs, tr, lab_all, TRAIN_P),
                                     build(seqs, te, lab_all, EVAL_P), lab_all, N_TASKS, 7)
                    b = build(seqs, te, lab_all, EVAL_P)
                    for p in EVAL_P:
                        sel = b["p"] == p
                        rows.append({"model": m, "p": p, "n": int(sel.sum()), "accuracy": float(strict_hits(s[sel], b["y"][sel]).mean())})
                extra["persons"][pp] = rows
        # cross-domain: refit on all LIBERO demos (agent view), test on every primary clip (descriptive)
        if args.cross_domain and (WORK / "cv_agent.json").exists():
            lseqs, llabel, _, _, _ = load_libero("agent")
            lcv = json.loads((WORK / "cv_agent.json").read_text())["choices"]
            b = build(seqs, sorted(label), label, EVAL_P)
            rows = []
            for m in MODELS:
                _, s = fit_score(m, most_frequent_hp(lcv, m), build(lseqs, sorted(llabel), llabel, TRAIN_P), b, llabel, N_TASKS, 11)
                for p in EVAL_P:
                    sel = b["p"] == p
                    rows.append({"model": m, "p": p, "n": int(sel.sum()), "accuracy": float(strict_hits(s[sel], b["y"][sel]).mean())})
            extra["cross_domain"] = rows
        (work / "cv_human.json").write_text(json.dumps(extra, indent=1))
        return 0
    oof = load_oof(work / "oof_human.npz")
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    units = H.units()
    if args.step == "plan":
        pred = {}
        for p in ROBOT_P:
            for t, e in units:
                pred[(p, t, e)] = int(strict_pred(oof[("primary", f"t{t}_r{H.rep_of(e)}", p)][None])[0])
        plan = plan_tokens(pred, units, original, "E2E-sup-human")
        plan["clip"] = {f"{t},{e}": f"t{t}_r{H.rep_of(e)}" for t, e in units}
        errs = check_reuse(plan, original, Path(args.eval_dir))
        print(f"GATE reuse: {'PASS' if not errs else 'FAIL'}")
        if errs:
            raise SystemExit("STOP: " + "; ".join(errs[:5]))
        res.mkdir(parents=True, exist_ok=True)
        (res / "e2e_sup_human_tokens.json").write_text(json.dumps(plan, indent=1))
        for c in plan["new"]:
            print(f"{c}: reuse {len(plan['reuse'][c])}, new {len(plan['new'][c])}")
        return 0
    # analyze
    res.mkdir(parents=True, exist_ok=True)
    cvh = json.loads((work / "cv_human.json").read_text())
    acc = acc_rows(oof, label, ("primary",) + MODELS, EVAL_P, {"camera": "phone"})
    write_csv(res / "accuracy.csv", acc)
    plot_curve(res / "curve.png", acc, {"chance 10%": (list(EVAL_P), [10] * len(EVAL_P), ":")}, "Trang's clips, leave one rep out")
    for p in (25, 50):
        write_confusion(res / f"confusion_p{p}.csv", confusion(oof, label, "primary", p))
    verdicts = human_rules(oof, label, scene_of)
    plan_path = res / "e2e_sup_human_tokens.json"
    robot = None
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text())
            # 300 clip units: Stage 12 E2E has no unit (8, 29), so the references are I-only and I-wrong
            _, summary, paired, src = robot_tables(plan, units, "E2E-sup-human", refs=("I-only", "I-wrong"), eval_dir=Path(args.eval_dir))
            write_csv(res / "summary.csv", summary)
            r4 = next(r for r in paired if r["rule"] == "R4")
            verdicts["R4"] = {"pass": r4["ci95_lo"] > 0, "delta": r4["delta"], "ci95": [r4["ci95_lo"], r4["ci95_hi"]],
                              "rule": "success(E2E-sup-human, p=25) - success(I-wrong) CI lower bound > 0"}
            robot = {"summary": summary, "paired": paired, "episodes": src}
        except SystemExit as exc:
            print(f"robot part not ready: {exc}")
    lines = ["# Stage 15 on Trang's clips (primary model, leave one rep out)", "", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for k, v in verdicts.items():
        lines.append(f"| {k}: {v['rule']} | {v.get('accuracy', v.get('delta')):.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {'PASS' if v['pass'] else 'FAIL'} |")
    (res / "rules.md").write_text("\n".join(lines) + "\n")
    (res / "config.json").write_text(json.dumps({"folds": [{k: v for k, v in c.items() if k != "test"} for c in cvh["choices"]],
                                                 "persons": cvh["persons"], "cross_domain": cvh["cross_domain"],
                                                 "verdicts": verdicts, "robot": robot}, indent=2))
    print((res / "rules.md").read_text())
    return 0


# ----------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------

def _toy(n_per=12, k=4, d=16, seed=0, sep=3.0):
    """Toy videos whose frames drift from a shared start towards a class direction: early prefixes are hard."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d))
    seqs, label = {}, {}
    for c in range(k):
        for i in range(n_per):
            v = f"c{c}_v{i}"
            n = 20
            times = np.arange(n) * 0.25
            frac = (np.arange(n) / (n - 1))[:, None]
            emb = S12.normalize(rng.normal(size=d) * 0.1 + sep * frac * centers[c] + 0.5 * rng.normal(size=(n, d)))
            seqs[v] = {p: emb[sample8(times, libero_end(times, p))] for p in EVAL_P}
            label[v] = c
    return seqs, label


def cmd_selftest(_) -> int:
    # frame grid: 4 fps on a 10 fps demo, last frame kept, strictly increasing
    idx, times = frame_grid(101, 10.0)
    assert idx[0] == 0 and idx[-1] == 100 and (np.diff(idx) > 0).all() and np.isclose(times[-1], 10.0)
    assert len(idx) == 41
    idx, times = frame_grid(23, 10.0)
    assert idx.tolist()[-2:] == [20, 22] and times[-1] == 2.2
    # prefix sampling: 8 frames, last prefix frame included, p = 0 -> first frame repeated
    times = np.arange(41) * 0.25
    s = sample8(times, libero_end(times, 50))
    assert len(s) == 8 and s[0] == 0 and s[-1] == 20 and (np.diff(s) >= 0).all()
    assert sample8(times, libero_end(times, 0)).tolist() == [0] * 8
    assert sample8(times, libero_end(times, 100))[-1] == 40
    assert sample8(np.arange(3) * 0.25, 10).tolist() == [0, 0, 1, 1, 1, 1, 2, 2]
    assert sample8(times, human_end(2.0, 6.0, 0))[-1] == 8 and sample8(times, human_end(2.0, 6.0, 50))[-1] == 16
    # stratified folds: every class spread, sizes balanced, deterministic
    lab = np.repeat(np.arange(3), [7, 5, 10])
    f = strat_folds(lab, 5, 0)
    assert (f >= 0).all() and (f == strat_folds(lab, 5, 0)).all()
    for c in range(3):
        cnt = np.bincount(f[lab == c], minlength=5)
        assert cnt.max() - cnt.min() <= 1
    vm = val_split(lab, 0.2, 0)
    assert [int(vm[lab == c].sum()) for c in range(3)] == [1, 1, 2]
    # bootstrap within groups
    pt, lo, hi = strat_boot(np.r_[np.zeros(10), np.ones(10)], np.r_[np.ones(10), np.zeros(10)])
    assert pt == 0.5 and lo == 0.5 and hi == 0.5, (pt, lo, hi)
    pt, lo, hi = strat_boot(np.zeros(40), np.r_[np.ones(20), np.zeros(20)])
    assert pt == 0.5 and 0.3 < lo < 0.5 < hi < 0.7
    # strict top-1, macro-F1, two-way
    sc = np.array([[0.5, 0.5, 0.1], [0.1, 0.9, 0.0], [0.0, 0.2, 0.7]])
    assert strict_hits(sc, [0, 1, 2]).tolist() == [0, 1, 1] and strict_pred(sc).tolist() == [-1, 1, 2]
    assert np.isclose(macro_f1([0, 1, 1], [0, 1, 0], 2), np.mean([2 / 3, 2 / 3]))
    assert two_way(np.array([[0.2, 0.1, 0.9], [0.3, 0.1, 0.0]]), [0, 1]).tolist() == [1.0, 0.0]
    # models separate an easy toy problem
    seqs, label = _toy(n_per=10, k=3, d=12, sep=6.0)
    tr = build(seqs, sorted(label), label, TRAIN_P)
    for name, hp in (("svm-lin", {"C": 10}), ("svm-rbf", {"C": 10, "g": 1}), ("logreg", {"C": 10})):
        _, s = fit_score(name, hp, tr, tr, label, 3, 0)
        assert strict_hits(s, tr["y"])[tr["p"] == 100].mean() > 0.9, name
    # nested CV: inner folds never see the outer test videos, each video tested once
    seen = []

    def spy_inner(train):
        seen.append(set(train))
        return dict(zip(sorted(train), strat_folds([label[v] for v in sorted(train)], 2, 1).tolist()))

    outer = dict(zip(sorted(label), strat_folds([label[v] for v in sorted(label)], 3, 0).tolist()))
    small = {"svm-lin": [{"C": 1}, {"C": 100}], "logreg": [{"C": 1}], "svm-rbf": [{"C": 10, "g": 1}], "tcnn": [{}]}
    oof, choices = run_cv(seqs, label, outer, spy_inner, k=3, models=("logreg", "svm-lin", "svm-rbf"), grids=small, verbose=False)
    assert not leak_check(outer, choices)
    for c, tr_set in zip(choices, seen):
        assert not tr_set & set(c["test"]), "inner CV saw a test video"
    assert all(("primary", v, p) in oof for v in label for p in EVAL_P)
    bad = [dict(c, test=c["test"] + [choices[0]["test"][0]] if i == 1 else c["test"]) for i, c in enumerate(choices)]
    assert leak_check(outer, bad)
    # the primary choice depends on inner scores only: scrambling the test videos' features changes nothing
    seqs2 = {v: ({p: -x for p, x in s.items()} if outer[v] == 0 else s) for v, s in seqs.items()}
    _, ch2 = run_cv(seqs2, label, outer, spy_inner, k=3, models=("logreg", "svm-lin", "svm-rbf"), grids=small, verbose=False)
    assert ch2[0]["primary"] == choices[0]["primary"] and ch2[0]["models"] == choices[0]["models"]
    # I-only reuse logic
    original = {t: f"task {t}" for t in range(N_TASKS)}
    units = [(0, 0), (1, 0), (2, 3)]
    pred = {(p, t, e): t for p in ROBOT_P for t, e in units}
    pred[(25, 1, 0)] = 6
    pred[(50, 2, 3)] = -1
    plan = plan_tokens(pred, units, original)
    assert plan["reuse"]["E2E-sup-p25"] == ["0,0", "2,3"] and plan["new"]["E2E-sup-p25"] == ["1,0"]
    assert plan["tokens"]["E2E-sup-p25"]["1,0"] == "task 6" and "2,3" not in plan["tokens"]["E2E-sup-p50"]
    assert plan["new"]["E2E-sup-p50"] == ["2,3"]
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "extract", "cv", "plan", "analyze", "human"])
    ap.add_argument("--camera", choices=sorted(CAMS), default="agent")
    ap.add_argument("--step", choices=["extract", "cv", "plan", "analyze"], help="human: which step")
    ap.add_argument("--videos", default="/media/dhqg/d3/datasets/own_libero10")
    ap.add_argument("--labels")
    ap.add_argument("--work", default=str(WORK / "human"))
    ap.add_argument("--results", default=str(ROOT / "results/stage15_supervised_early/human"))
    ap.add_argument("--eval-dir", default=str(EVAL))
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--cross-domain", type=int, default=1)
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "extract": cmd_extract, "cv": cmd_cv, "plan": cmd_plan, "analyze": cmd_analyze,
            "human": cmd_human}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
