"""Stage 18: scene vs motion on LIBERO-10, own videos (Stage 16, copied) and EgoIntent (verbs of WHAT).

Pre-registration: results/stage18_scene_motion_ego/prereg.md (hash in prereg_hash.txt).

Sources of copied code (read with `git show`, no checkout):
  * LogReg, _lbfgs, choose-C edge extension: scripts/run_stage16_scene_motion.py @ e41e72c
    (claude/stage16-scene-motion), itself copied from Stage 15; class-balanced weights added here.
  * strat_folds, sample8, libero_end, strat_boot: scripts/run_stage15_supervised.py @ 2e865a6
    (origin/claude/stage15-supervised-early); outer folds from results/stage15_supervised_early/config.json there.
  * EgoIntent folds: scripts/run_scene_cv.assign_folds (8 scene-grouped folds, as run_ladder_v1.Data).

CPU only. Subcommands: labels, selftest, ego, libero, table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results/stage18_scene_motion_ego"
WORK = Path("/media/dhqg/d3/vlia_outputs/stage18")
MANIFEST = ROOT / "results/hoi_detection_v1/manifest.jsonl"
LOCK = ROOT / "results/test_protocol_full_v1/locked_protocol.json"
EGO_CACHE = Path("/media/dhqg/d3/vlia_outputs/ladder/siglip-so400m-patch14-384")
LIB_FEAT = Path("/media/dhqg/d3/vlia_outputs/stage15/feat/agent")
S15_BRANCH, S15_COMMIT = "origin/claude/stage15-supervised-early", "2e865a6"
S16_COMMIT = "e41e72c"

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
B = 10000
EGO_P = {25: 2, 50: 4, 100: 8}  # p -> number of cached frames observed
LIB_P = (10, 25, 50, 100)
N_FRAMES = 8
N_FOLDS_EGO, N_INNER_EGO = 8, 4
N_INNER_LIB = 4
FEATS = ("S", "M", "S+M", "F+")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def state(line: str):
    with (WORK / "STATE.md").open("a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} | {line}\n")


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# labels (fixed rule, see label_rule.md)
# ----------------------------------------------------------------------
SKIP = {"then", "now", "i", "the", "person", "a", "an", "and", "he", "she", "they", "c", "camera", "wearer", "user",
        "also", "first", "next", "again", "still", "just", "to"}
ADV = {"carefully", "slowly", "gently", "quickly", "firmly", "briefly", "repeatedly", "continuously"}
PART = {"up", "down", "out", "off", "away", "back"}
IRR = {"took": "take", "taken": "take", "held": "hold", "put": "put", "putting": "put", "puts": "put", "cut": "cut",
       "cutting": "cut", "set": "set", "setting": "set", "sets": "set", "got": "get", "getting": "get", "gets": "get",
       "made": "make", "making": "make", "threw": "throw", "thrown": "throw", "stood": "stand", "sat": "sit",
       "sitting": "sit", "ran": "run", "running": "run", "wrote": "write", "writing": "write", "written": "write",
       "shook": "shake", "shaking": "shake", "brought": "bring", "dug": "dig", "digging": "dig", "hung": "hang",
       "left": "leave", "laid": "lay", "lying": "lie", "wiping": "wipe", "placing": "place", "moving": "move",
       "using": "use", "removing": "remove", "closing": "close", "taking": "take", "pouring": "pour",
       "dipping": "dip", "stirring": "stir", "scrubbing": "scrub", "rubbing": "rub", "tapping": "tap",
       "flipping": "flip", "dropping": "drop", "grabbing": "grab", "wrapping": "wrap", "trimming": "trim",
       "sweeping": "sweep", "adjusting": "adjust", "rinsing": "rinse", "squeezing": "squeeze",
       "arranging": "arrange", "dragging": "drag", "tying": "tie", "rolling": "roll", "folding": "fold",
       "pressing": "press", "checking": "check", "holding": "hold", "picking": "pick", "opening": "open",
       "washing": "wash", "turning": "turn", "pulling": "pull", "pushing": "push", "lifting": "lift",
       "hanging": "hang", "applying": "apply", "drying": "dry", "mixing": "mix", "inspecting": "inspect",
       "reaching": "reach", "releasing": "release", "returning": "return", "cleaning": "clean"}
KEEP = ("place", "pick_up", "grasp", "wipe", "hold", "position", "adjust", "remove", "align", "walk")  # STOP A, ok


def lemma(w: str) -> str:
    if w in IRR:
        return IRR[w]
    for suf, rep in (("ies", "y"), ("ied", "y")):
        if w.endswith(suf) and len(w) > 4:
            return w[:-3] + rep
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[:-len(suf)]
            if len(b) > 2 and b[-1] == b[-2] and b[-1] not in "ls":
                b = b[:-1]
            return b
    if w.endswith("es") and w[:-2].endswith(("sh", "ch", "ss", "x", "z")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def verb(text: str):
    ws = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    i = 0
    while i < len(ws) and (ws[i] in SKIP or ws[i] in ADV):
        i += 1
    if i >= len(ws):
        return None
    v = lemma(ws[i])
    if i + 1 < len(ws) and ws[i + 1] in PART:
        v += "_" + ws[i + 1]
    return v


def locked_uids() -> set:
    lk = json.loads(LOCK.read_text())
    return set(lk["candidate_test_video_uids"]) | set(lk["excluded_test_video_uids"])


def load_manifest():
    rows = [json.loads(line) for line in MANIFEST.open() if line.strip()]
    assert all(r["split"] in ("train", "val") for r in rows), "locked split in manifest"
    assert not ({r["video_uid"] for r in rows} & locked_uids()), "locked test video in development manifest"
    return rows


def durations(rows) -> np.ndarray:
    path = WORK / "durations.json"
    if path.exists():
        d = json.loads(path.read_text())
    else:
        import cv2
        d = {}
        for r in rows:
            cap = cv2.VideoCapture(r["video_path"])
            n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            d[r["sample_id"]] = (n - 1) / fps if fps > 0 and n > 1 else 0.0
        atomic_json(path, d)
    return np.array([d[r["sample_id"]] for r in rows])


def cmd_labels(_args) -> int:
    rows = load_manifest()
    v = [verb(r["what"]) for r in rows]
    c, vids = Counter(v), defaultdict(set)
    for x, r in zip(v, rows):
        vids[x].add(r["video_uid"])
    keep = [x for x, n in c.most_common() if n >= 30 and len(vids[x]) >= 8][:10]
    assert tuple(keep) == KEEP, keep
    with (RES / "labels_ego.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "verb", "steps", "videos", "kept"])
        for i, (x, n) in enumerate(c.most_common(25)):
            w.writerow([i + 1, x, n, len(vids[x]), int(x in KEEP)])
    kept = [x for x in v if x in KEEP]
    per_vid = defaultdict(set)
    for x, r in zip(v, rows):
        if x in KEEP:
            per_vid[r["video_uid"]].add(x)
    dur = durations(rows)
    (RES / "label_rule.md").write_text(
        "# EgoIntent label rule (fixed before any accuracy)\n\n"
        "Label = first verb of the WHAT sentence (`what` / `local_intent`) of each step:\n\n"
        "1. lower-case, replace every non-letter by a space, split on spaces;\n"
        f"2. skip leading words in {sorted(SKIP)} or adverbs in {sorted(ADV)};\n"
        "3. lemma by inflection only (irregular table + -ies/-ied -> y, -ing/-ed with doubled consonant undone, "
        "-es after sh/ch/ss/x/z, -s); no merging by meaning (place != put);\n"
        f"4. if the next word is a particle in {sorted(PART)}, append it (`pick up` -> `pick_up`); "
        "on/in/over are not merged (usually prepositions: `place on table`).\n"
        "5. keep verbs with >= 30 steps and >= 8 videos, at most the 10 most frequent; other steps are dropped.\n\n"
        "Deviation from the first draft (before any accuracy): skipping every word ending in -ly also skipped "
        "`apply`; replaced by the fixed adverb list above. The kept set did not change.\n\n"
        f"Result: K = {len(KEEP)} ({', '.join(KEEP)}), {len(kept)} of {len(rows)} steps kept, largest class "
        f"{Counter(kept).most_common(1)[0][1] / len(kept):.3f}; distinct kept labels per video: mean "
        f"{np.mean([len(s) for s in per_vid.values()]):.2f}, min {min(len(s) for s in per_vid.values())}, "
        f"{len(per_vid)} videos.\n\nWHY is not used: {len(set(r['why'] for r in rows))} distinct sentences over "
        f"{len(rows)} steps. Mean step duration {dur.mean():.2f} s (median {np.median(dur):.2f} s).\n")
    state(f"labels: labels_ego.csv, label_rule.md (K={len(KEEP)}, kept {len(kept)})")
    log("labels done")
    return 0


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def feats(seq: np.ndarray, k: int) -> dict:
    """seq (T, D) observed-order frames; k = number observed (k <= T)."""
    seq = np.asarray(seq, np.float64)
    z1 = seq[0]
    out = {"S": z1, "F+": np.concatenate([z1, seq[:k].mean(0)])}
    if k >= 2:
        m = (seq[1:k] - z1).mean(0)
        out["M"], out["S+M"] = m, np.concatenate([z1, m])
    return out


# ----------------------------------------------------------------------
# classifier (Stage 16 LogReg, copied; balanced sample weights added)
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

        y = np.asarray(y)
        cnt = np.bincount(y, minlength=self.k).astype(float)
        cw = np.where(cnt > 0, len(y) / (self.k * np.maximum(cnt, 1)), 0.0)
        sw = torch.as_tensor(cw[y], dtype=torch.float64)
        X = torch.as_tensor(X, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.long)
        W = torch.zeros(X.shape[1], self.k, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(self.k, dtype=torch.float64, requires_grad=True)
        _lbfgs([W, b], lambda: self.C * (sw * F.cross_entropy(X @ W + b, yt, reduction="none")).sum()
               + 0.5 * (W * W).sum())
        self.W, self.b = W.detach(), b.detach()
        return self

    def scores(self, X):
        import torch

        return torch.softmax(torch.as_tensor(X, dtype=torch.float64) @ self.W + self.b, dim=1).numpy()


def fit_predict(Xtr, ytr, Xte, C, k):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    return LogReg(C, k).fit((Xtr - mu) / sd, ytr).scores((Xte - mu) / sd)


def bal_acc(y, yhat, k):
    r = [np.mean(yhat[y == c] == c) for c in range(k) if (y == c).any()]
    return float(np.mean(r))


def choose_C(X, y, inner, k, metric):
    """inner: inner-fold id per training row. metric(y, yhat) on pooled inner predictions."""
    def score(C):
        pred = np.empty(len(y), int)
        for g in np.unique(inner):
            tr, te = inner != g, inner == g
            pred[te] = fit_predict(X[tr], y[tr], X[te], C, k).argmax(1)
        return metric(y, pred)

    grid = list(C_GRID)
    s = {C: score(C) for C in grid}
    best = min(grid, key=lambda C: (-s[C], C))
    ext = None
    if best == grid[0] or best == grid[-1]:
        ext = grid[0] / 10 if best == grid[0] else grid[-1] * 10
        s[ext] = score(ext)
        best = min(s, key=lambda C: (-s[C], C))
    return best, ext, {str(c): v for c, v in s.items()}


def nested(X, y, outer, inner_of, k, metric, done_path: Path):
    """Outer CV with per-fold resume. inner_of(train_idx) -> inner fold id per training row."""
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    probs = np.zeros((len(y), k))
    for f in sorted(np.unique(outer)):
        key = str(f)
        te, tr = np.flatnonzero(outer == f), np.flatnonzero(outer != f)
        if key not in done:
            t0 = time.time()
            C, ext, grid = choose_C(X[tr], y[tr], inner_of(tr), k, metric)
            p = fit_predict(X[tr], y[tr], X[te], C, k)
            done[key] = {"C": C, "ext": ext, "grid": grid, "probs": p.tolist(), "sec": time.time() - t0}
            atomic_json(done_path, done)
        probs[te] = np.asarray(done[key]["probs"])
    return probs, {f: {kk: v for kk, v in d.items() if kk != "probs"} for f, d in done.items()}


# ----------------------------------------------------------------------
# metrics / bootstrap
# ----------------------------------------------------------------------
def macro_f1(y, yhat, k):
    f = []
    for c in range(k):
        if not ((y == c).any() or (yhat == c).any()):
            continue
        tp = np.sum((y == c) & (yhat == c))
        pr = tp / max(np.sum(yhat == c), 1)
        rc = tp / max(np.sum(y == c), 1)
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def video_boot_weights(groups, seed=0, b=B):
    """(b, G) multiplicities of whole groups resampled with replacement."""
    G = len(np.unique(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(b, G))
    return np.stack([np.bincount(d, minlength=G) for d in draws])


def boot_bal(y, yhat, gid, W, k):
    """Balanced accuracy for each bootstrap weight row; gid = group index 0..G-1 per item."""
    G = W.shape[1]
    hit = np.zeros((G, k))
    tot = np.zeros((G, k))
    np.add.at(tot, (gid, y), 1)
    np.add.at(hit, (gid, y), (y == yhat).astype(float))
    T, H = W @ tot, W @ hit
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(T > 0, H / T, np.nan)
    return np.nanmean(r, axis=1)


def strat_boot_idx(strata, seed=0, b=B):
    """(b, n) indices resampling items within each stratum (Stage 15 strat_boot, index form)."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = np.empty((b, len(strata)), int)
    for s in np.unique(strata):
        ids = np.flatnonzero(strata == s)
        out[:, ids] = ids[rng.integers(0, len(ids), size=(b, len(ids)))]
    return out


def ci(v):
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


# ----------------------------------------------------------------------
# EgoIntent
# ----------------------------------------------------------------------
def ego_data():
    from run_scene_cv import assign_folds

    rows = load_manifest()
    ids = json.loads((EGO_CACHE / "sample_ids.json").read_text())
    assert ids == [r["sample_id"] for r in rows], "cache order != manifest order"
    emb = np.load(EGO_CACHE / "frame_emb.npy", mmap_mode="r")
    fm = assign_folds([r["scene"] for r in rows], N_FOLDS_EGO)  # on all 2,503 steps, as run_ladder_v1.Data
    v = [verb(r["what"]) for r in rows]
    keep = np.array([x in KEEP for x in v])
    idx = np.flatnonzero(keep)
    dur = durations(rows)[idx]
    rows = [rows[i] for i in idx]
    y = np.array([KEEP.index(v[i]) for i in idx])
    Z = np.asarray(emb[idx], np.float64)
    vids = np.array([r["video_uid"] for r in rows])
    outer = np.array([fm[r["scene"]] for r in rows])
    assert not (set(vids) & locked_uids())
    return rows, Z, y, vids, outer, dur


def video_inner(vids_tr: np.ndarray, n=N_INNER_EGO) -> np.ndarray:
    """Whole training videos -> n inner folds, greedy balance by step count, deterministic."""
    cnt = Counter(vids_tr.tolist())
    load, fold = [0] * n, {}
    for vid in sorted(cnt, key=lambda s: (-cnt[s], s)):
        f = min(range(n), key=lambda i: (load[i], i))
        fold[vid], load[f] = f, load[f] + cnt[vid]
    return np.array([fold[x] for x in vids_tr])


def ego_matrix(Z, feat, k, vids):
    if feat == "S-vid":
        z1 = Z[:, 0]
        means = {u: z1[vids == u].mean(0) for u in np.unique(vids)}
        return np.stack([means[u] for u in vids])
    return np.stack([feats(z, k)[feat] for z in Z])


def cmd_ego(_args) -> int:
    t0 = time.time()
    rows, Z, y, vids, outer, dur = ego_data()
    K = len(KEEP)
    for f in np.unique(outer):
        assert not (set(vids[outer == f]) & set(vids[outer != f])), "video crosses folds"
        assert len(np.unique(y[outer == f])) >= 2
    metric = lambda a, b: bal_acc(a, b, K)  # noqa: E731
    jobs = [("S", 0), ("S-vid", 0)] + [(f, p) for p in EGO_P for f in ("M", "S+M", "F+")]
    out = {}
    for feat, p in jobs:
        kf = EGO_P.get(p, 1)
        X = ego_matrix(Z, feat, kf, vids)
        path = WORK / f"ego_{feat.replace('+', 'p')}_p{p}.json"
        probs, choice = nested(X, y, outer, lambda tr: video_inner(vids[tr]), K, metric, path)
        out[(feat, p)] = (probs, choice)
        log(f"ego {feat} p{p}: bal_acc {metric(y, probs.argmax(1)):.3f} ({time.time() - t0:.0f}s)")
        if len(out) == 1:
            state(f"ego: first job done in {time.time() - t0:.0f}s")
    np.savez(WORK / "ego_oof.npz", y=y, vids=vids, outer=outer, dur=dur,
             **{f"{f}|{p}": out[(f, p)][0] for f, p in out})
    atomic_json(WORK / "ego_choices.json", {f"{f}|{p}": out[(f, p)][1] for f, p in out})
    state(f"ego done in {time.time() - t0:.0f}s: {WORK}/ego_oof.npz")
    return 0


# ----------------------------------------------------------------------
# LIBERO
# ----------------------------------------------------------------------
def strat_folds(labels, k: int, seed: int) -> np.ndarray:
    """Fold id per item: within each class, shuffle and deal to folds 0..k-1 in turn. (Stage 15, copied.)"""
    labels = np.asarray(labels)
    out = np.full(len(labels), -1)
    rng = np.random.default_rng(seed)
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        ids = ids[rng.permutation(len(ids))]
        out[ids] = np.arange(len(ids)) % k
    return out


def sample8(times: np.ndarray, end: float) -> np.ndarray:
    """Stage 15, copied: positions of 8 frames spread evenly over the frames with time <= end."""
    n = max(1, int(np.sum(times <= end + 1e-9)))
    return np.round(np.linspace(0, n - 1, N_FRAMES)).astype(int)


def s15_config() -> dict:
    txt = subprocess.run(["git", "-C", str(ROOT), "show", f"{S15_COMMIT}:results/stage15_supervised_early/config.json"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(txt)


def s15_accuracy() -> dict:
    txt = subprocess.run(["git", "-C", str(ROOT), "show", f"{S15_COMMIT}:results/stage15_supervised_early/accuracy.csv"],
                         capture_output=True, text=True, check=True).stdout
    return {int(r["p"]): float(r["accuracy"]) for r in csv.DictReader(txt.splitlines())
            if r["camera"] == "agent" and r["model"] == "primary"}


def libero_data():
    outer_map = s15_config()["outer_fold"]["agent"]
    keys = sorted(outer_map)
    seqs, y, outer, T = {}, [], [], []
    for kk in keys:
        z = np.load(LIB_FEAT / f"{kk}.npz")
        emb, times = z["emb"], z["times"]
        seqs[kk] = {p: emb[sample8(times, p / 100 * float(times[-1]))] for p in LIB_P}
        y.append(int(z["task"]))
        outer.append(outer_map[kk])
        T.append(float(times[-1]))
    return keys, seqs, np.array(y), np.array(outer), np.array(T)


def cmd_libero(_args) -> int:
    t0 = time.time()
    keys, seqs, y, outer, T = libero_data()
    assert len(keys) == 379
    K = 10
    metric = lambda a, b: float(np.mean(a == b))  # noqa: E731

    def inner_of(tr):  # Stage 15 libero_inner_fn, verbatim: seed 1 + outer fold of the first sorted training key
        return strat_folds(y[tr], N_INNER_LIB, 1 + int(outer[tr[0]]))

    jobs = [("S", 0)] + [(f, p) for p in LIB_P for f in ("M", "S+M", "F+")]
    out = {}
    for feat, p in jobs:
        pp = p if p else LIB_P[0]
        X = np.stack([feats(seqs[kk][pp], N_FRAMES)[feat] for kk in keys])
        path = WORK / f"lib_{feat.replace('+', 'p')}_p{p}.json"
        probs, choice = nested(X, y, outer, inner_of, K, metric, path)
        out[(feat, p)] = (probs, choice)
        log(f"libero {feat} p{p}: acc {metric(y, probs.argmax(1)):.3f} ({time.time() - t0:.0f}s)")
        if feat == "S":
            s = metric(y, probs.argmax(1))
            ref = s15_accuracy()[0]
            if abs(s - ref) > 0.03:
                state(f"libero STOP: S {s:.3f} vs Stage 15 {ref:.3f}")
                raise SystemExit(f"STOP: LIBERO S {s:.3f} differs from Stage 15 {ref:.3f} by > 3 points")
    s25, ref25 = metric(y, out[("S+M", 25)][0].argmax(1)), s15_accuracy()[25]
    if abs(s25 - ref25) > 0.03:
        state(f"libero STOP: S+M p25 {s25:.3f} vs Stage 15 {ref25:.3f}")
        raise SystemExit(f"STOP: LIBERO S+M p25 {s25:.3f} differs from Stage 15 {ref25:.3f}")
    np.savez(WORK / "lib_oof.npz", y=y, keys=np.array(keys), outer=outer, T=T,
             **{f"{f}|{p}": out[(f, p)][0] for f, p in out})
    atomic_json(WORK / "lib_choices.json", {f"{f}|{p}": out[(f, p)][1] for f, p in out})
    state(f"libero done in {time.time() - t0:.0f}s: {WORK}/lib_oof.npz")
    return 0


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def s16_rows():
    txt = subprocess.run(["git", "-C", str(ROOT), "show", f"{S16_COMMIT}:results/stage16_scene_motion/accuracy.csv"],
                         capture_output=True, text=True, check=True).stdout
    return [r for r in csv.DictReader(txt.splitlines()) if r["part"] == "A" and r["table"] == "pooled"]


def conclusion(r1, r2, r3):
    if r1:
        return "Scene suffices; cannot claim action understanding."
    if r2 and r3:
        return "Evidence the model uses action, not only scene."
    return "Insufficient evidence."


def cmd_table(_args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = 10
    acc_cols = ["dataset", "feature", "p", "seconds", "n", "acc", "bal_acc", "macro_f1", "ci_metric", "ci_lo", "ci_hi",
                "chance", "majority"]
    # --- EgoIntent
    e = np.load(WORK / "ego_oof.npz")
    y, vids, outer, dur = e["y"], e["vids"], e["outer"], e["dur"]
    uv = np.unique(vids)
    gid = np.searchsorted(uv, vids)
    W = video_boot_weights(uv, seed=0)
    maj = np.empty(len(y), int)
    for f in np.unique(outer):
        maj[outer == f] = np.bincount(y[outer != f], minlength=K).argmax()
    ego, ego_boot = [], {}
    for key in [k for k in e.files if "|" in k]:
        feat, p = key.split("|")
        p = int(p)
        yh = e[key].argmax(1)
        bb = boot_bal(y, yh, gid, W, K)
        ego_boot[(feat, p)] = bb
        sec = float(np.mean((EGO_P[p] - 1) / 7 * dur)) if p else 0.0
        ego.append({"dataset": "EgoIntent", "feature": feat, "p": p, "seconds": round(sec, 3), "n": len(y),
                    "acc": float(np.mean(yh == y)), "bal_acc": bal_acc(y, yh, K), "macro_f1": macro_f1(y, yh, K),
                    "ci_metric": "bal_acc", "ci_lo": ci(bb)[0], "ci_hi": ci(bb)[1], "chance": 1 / K,
                    "majority": float(np.mean(maj == y))})
    ego.sort(key=lambda r: (r["p"], FEATS.index(r["feature"]) if r["feature"] in FEATS else 9))
    with (RES / "accuracy_ego.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, acc_cols)
        w.writeheader()
        w.writerows(ego)
    eg = {(r["feature"], r["p"]): r for r in ego}
    d = ego_boot[("S+M", 50)] - ego_boot[("S", 0)]
    ego_r = {"R1": eg[("S", 0)]["bal_acc"] >= 0.80,
             "R2": (eg[("S+M", 50)]["bal_acc"] - eg[("S", 0)]["bal_acc"], *ci(d)),
             "R3": (eg[("M", 50)]["bal_acc"], eg[("M", 50)]["ci_lo"])}
    ego_v = conclusion(ego_r["R1"], ego_r["R2"][1] > 0, ego_r["R3"][1] > 1 / K)
    # per video / confusion / class recall
    yS, ySM, yM = (e[f"{k}"].argmax(1) for k in ("S|0", "S+M|50", "M|50"))
    with (RES / "per_video_ego.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_uid", "scene", "fold", "n", "n_labels", "acc_S", "acc_SM_p50", "bal_S", "bal_SM_p50"])
        scene = {r["video_uid"]: r["scene"] for r in load_manifest()}
        for u in uv:
            m = vids == u
            w.writerow([u, scene[u], int(outer[m][0]), int(m.sum()), len(np.unique(y[m])),
                        round(float(np.mean(yS[m] == y[m])), 4), round(float(np.mean(ySM[m] == y[m])), 4),
                        round(bal_acc(y[m], yS[m], K), 4), round(bal_acc(y[m], ySM[m], K), 4)])
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, ySM), 1)
    with (RES / "confusion_ego_p50.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(KEEP))
        for i, c in enumerate(KEEP):
            w.writerow([c] + cm[i].tolist())
    recM = {KEEP[c]: float(np.mean(yM[y == c] == c)) for c in range(K)}
    # --- LIBERO
    L = np.load(WORK / "lib_oof.npz")
    ly, lT, louter = L["y"], L["T"], L["outer"]
    idx = strat_boot_idx(ly, seed=0)
    lmaj = np.empty(len(ly), int)
    for f in np.unique(louter):
        lmaj[louter == f] = np.bincount(ly[louter != f], minlength=K).argmax()
    lib, lib_hits = [], {}
    s15 = s15_accuracy()
    for key in [k for k in L.files if "|" in k]:
        feat, p = key.split("|")
        p = int(p)
        yh = L[key].argmax(1)
        hit = (yh == ly).astype(float)
        lib_hits[(feat, p)] = hit
        bh = hit[idx].mean(1)
        lib.append({"dataset": "LIBERO-10", "feature": feat, "p": p, "seconds": round(float(np.mean(p / 100 * lT)), 3),
                    "n": len(ly), "acc": float(hit.mean()), "bal_acc": bal_acc(ly, yh, K),
                    "macro_f1": macro_f1(ly, yh, K), "ci_metric": "acc", "ci_lo": ci(bh)[0], "ci_hi": ci(bh)[1],
                    "chance": 1 / K, "majority": float(np.mean(lmaj == ly)),
                    "stage15_primary": s15.get(p if feat != "S" else 0, "") if feat in ("S", "F+", "S+M") else ""})
    lib.sort(key=lambda r: (r["p"], FEATS.index(r["feature"])))
    with (RES / "accuracy_libero.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, acc_cols + ["stage15_primary"])
        w.writeheader()
        w.writerows(lib)
    lb = {(r["feature"], r["p"]): r for r in lib}
    dl = (lib_hits[("S+M", 50)] - lib_hits[("S", 0)])[idx].mean(1)
    lib_r = {"R1": lb[("S", 0)]["acc"] >= 0.80,
             "R2": (lb[("S+M", 50)]["acc"] - lb[("S", 0)]["acc"], *ci(dl)),
             "R3": (lb[("M", 50)]["acc"], lb[("M", 50)]["ci_lo"])}
    lib_v = conclusion(lib_r["R1"], lib_r["R2"][1] > 0, lib_r["R3"][1] > 1 / K)
    # --- own videos (Stage 16, copied)
    own = {(r["feature"], int(r["p"])): r for r in s16_rows()}
    own_S = float(own[("S", 0)]["acc"])
    own_v = conclusion(own_S >= 0.80, True, True) + " (Stage 16 H1-a/b/c PASS, copied)"

    def cell(r, m):
        return f"{100 * float(r[m]):.1f} [{100 * float(r['ci_lo']):.1f}, {100 * float(r['ci_hi']):.1f}]"

    rows3 = [
        {"dataset": "LIBERO-10 (robot demos)", "split": "5-fold by demo (Stage 15 folds)", "metric": "acc",
         "chance": 0.10, "S": cell(lb[("S", 0)], "acc"), "M_p25": cell(lb[("M", 25)], "acc"),
         "M_p50": cell(lb[("M", 50)], "acc"), "SM_p25": cell(lb[("S+M", 25)], "acc"),
         "SM_p50": cell(lb[("S+M", 50)], "acc"), "conclusion": lib_v},
        {"dataset": "Own videos (shared tables)", "split": "leave-one-person-out (Stage 16, not rerun)",
         "metric": "acc", "chance": float(own[("S", 0)]["chance"]), "S": cell(own[("S", 0)], "acc"),
         "M_p25": cell(own[("M", 25)], "acc"), "M_p50": cell(own[("M", 50)], "acc"),
         "SM_p25": cell(own[("F+", 25)], "acc") + " (F+)", "SM_p50": cell(own[("F+", 50)], "acc") + " (F+)",
         "conclusion": own_v},
        {"dataset": "EgoIntent (first-person, verb of step)", "split": "unseen videos and scenes (8 scene folds)",
         "metric": "bal_acc", "chance": 0.10, "S": cell(eg[("S", 0)], "bal_acc"),
         "M_p25": cell(eg[("M", 25)], "bal_acc"), "M_p50": cell(eg[("M", 50)], "bal_acc"),
         "SM_p25": cell(eg[("S+M", 25)], "bal_acc"), "SM_p50": cell(eg[("S+M", 50)], "bal_acc"),
         "conclusion": ego_v},
    ]
    with (RES / "bang_3x3.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows3[0]))
        w.writeheader()
        w.writerows(rows3)
    md = ["# Stage 18: scene vs motion (% , 95% CI)", "",
          "| dataset | split | metric | chance | S | M p25 | M p50 | S+M p25 | S+M p50 | conclusion (rule) |",
          "|---|---|---|---:|---|---|---|---|---|---|"]
    md += [f"| {r['dataset']} | {r['split']} | {r['metric']} | {100 * r['chance']:.1f} | {r['S']} | {r['M_p25']} | "
           f"{r['M_p50']} | {r['SM_p25']} | {r['SM_p50']} | {r['conclusion']} |" for r in rows3]
    md += ["", "Own videos: S+M column is Stage 16 F+ = [S, mean z_t] (not [z_1, M]); S there is the idle part before "
           "the action. Source: " + f"{S16_COMMIT}:results/stage16_scene_motion/accuracy.csv (A, pooled), Stage 16 "
           "prereg sha256 4e041e2e.",
           f"EgoIntent S-vid (video identity only): {cell(eg[('S-vid', 0)], 'bal_acc')}; "
           f"majority-class accuracy {100 * eg[('S', 0)]['majority']:.1f}."]
    (RES / "bang_3x3.md").write_text("\n".join(md) + "\n")
    rules = ["# Stage 18 rules (pre-registered)", "", "| dataset | rule | value | 95% CI | verdict |", "|---|---|---|---|---|"]
    for name, r in (("EgoIntent", ego_r), ("LIBERO-10", lib_r)):
        S = eg[("S", 0)]["bal_acc"] if name == "EgoIntent" else lb[("S", 0)]["acc"]
        rules.append(f"| {name} | R1 S >= 0.80 | {S:.3f} | | {'HOLDS' if r['R1'] else 'does not hold'} |")
        rules.append(f"| {name} | R2 (S+M p50) - S, CI lo > 0 | {r['R2'][0]:+.3f} | [{r['R2'][1]:+.3f}, {r['R2'][2]:+.3f}] "
                     f"| {'PASS' if r['R2'][1] > 0 else 'FAIL'} |")
        rules.append(f"| {name} | R3 M p50 CI lo > 1/K | {r['R3'][0]:.3f} | lo {r['R3'][1]:.3f} | "
                     f"{'PASS' if r['R3'][1] > 1 / K else 'FAIL'} |")
    rules += [f"| Own videos | R1 S >= 0.80 | {own_S:.3f} | | {'HOLDS' if own_S >= 0.8 else 'does not hold'} |", "",
              f"- EgoIntent: {ego_v}", f"- LIBERO-10: {lib_v}", f"- Own videos: {own_v}", "",
              "EgoIntent M p50 recall per class: " + ", ".join(f"{c} {v:.2f}" for c, v in
                                                               sorted(recM.items(), key=lambda t: -t[1]))]
    (RES / "rules.md").write_text("\n".join(rules) + "\n")
    # curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, rows_, m, ttl in ((ax[0], lib, "acc", "LIBERO-10 (accuracy)"), (ax[1], ego, "bal_acc",
                                                                             "EgoIntent (balanced accuracy)")):
        tab = {(r["feature"], r["p"]): r for r in rows_}
        ps = sorted({r["p"] for r in rows_ if r["p"]})
        for feat in ("S", "M", "S+M", "F+"):
            if feat == "S":
                ys = [tab[("S", 0)][m]] * len(ps)
            else:
                ys = [tab[(feat, p)][m] for p in ps]
            a.plot(ps, ys, marker="o", ls="--" if feat == "F+" else "-", label=feat)
        if ("S-vid", 0) in tab:
            a.axhline(tab[("S-vid", 0)][m], color="grey", ls=":", label="S-vid")
        a.axhline(0.1, color="k", lw=0.6, label="chance")
        a.set(title=ttl, xlabel="observed p (%)", ylim=(0, 1.02), xticks=ps)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "curve.png", dpi=120)
    # config
    ch_e = json.loads((WORK / "ego_choices.json").read_text())
    ch_l = json.loads((WORK / "lib_choices.json").read_text())
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (RES / "config.json").write_text(json.dumps({
        "git_head": git, "code_sources": {"stage15": f"{S15_BRANCH} {S15_COMMIT}", "stage16": f"claude/stage16-scene-motion {S16_COMMIT}"},
        "caches": {"ego": str(EGO_CACHE), "libero": str(LIB_FEAT)}, "labels": KEEP, "ego_p_frames": EGO_P,
        "lib_p": LIB_P, "C_grid": C_GRID, "bootstrap": B, "ego_folds": "run_scene_cv.assign_folds(scenes, 8)",
        "ego_inner": f"{N_INNER_EGO} folds of whole training videos", "lib_inner": "strat_folds(y, 4, 1 + fold)",
        "C_chosen": {"ego": {k: {f: v["C"] for f, v in d.items()} for k, d in ch_e.items()},
                     "libero": {k: {f: v["C"] for f, v in d.items()} for k, d in ch_l.items()}},
        "C_edge_extensions": {"ego": {k: {f: v["ext"] for f, v in d.items() if v["ext"]} for k, d in ch_e.items()},
                              "libero": {k: {f: v["ext"] for f, v in d.items() if v["ext"]} for k, d in ch_l.items()}},
        "fit_seconds": {"ego": sum(v["sec"] for d in ch_e.values() for v in d.values()),
                        "libero": sum(v["sec"] for d in ch_l.values() for v in d.values())},
    }, indent=1, default=list))
    state("table done: bang_3x3.md, rules.md, curve.png")
    print("\n".join(md))
    print("\n".join(rules))
    return 0


# ----------------------------------------------------------------------
# selftest (synthetic data)
# ----------------------------------------------------------------------
def cmd_selftest(_args) -> int:
    rng = np.random.default_rng(0)
    # features: M = 0 for constant frames, S independent of p
    z = np.tile(rng.normal(size=(1, 6)), (8, 1))
    for k in (2, 4, 8):
        f = feats(z, k)
        assert np.allclose(f["M"], 0) and np.allclose(f["S"], z[0])
    seq = rng.normal(size=(8, 6))
    assert all(np.allclose(feats(seq, k)["S"], seq[0]) for k in (1, 2, 4, 8))
    assert "M" not in feats(seq, 1)
    # nested CV: standardisation / C choice see training rows only, folds group whole videos
    n, K = 120, 3
    vids = np.repeat(np.arange(12), 10).astype(str)
    outer = np.repeat(np.arange(4), 30)
    y = rng.integers(0, K, n)
    X = rng.normal(size=(n, 5)) + y[:, None]
    seen = []
    global fit_predict
    orig = fit_predict

    def spy(Xtr, ytr, Xte, C, k):
        seen.append(len(Xtr))
        return orig(Xtr, ytr, Xte, C, k)

    fit_predict = spy
    try:
        tmp = WORK / "selftest_nested.json"
        tmp.unlink(missing_ok=True)
        X_poison = X.copy()
        probs, ch = nested(X_poison, y, outer, lambda tr: video_inner(vids[tr]), K,
                           lambda a, b: bal_acc(a, b, K), tmp)
        assert max(seen) <= 90, "a fit saw test rows"
        # changing test rows of fold 0 must not change fold 0's chosen C (selection uses train only)
        tmp.unlink()
        X2 = X.copy()
        X2[outer == 0] = rng.normal(size=((outer == 0).sum(), 5)) * 100
        _, ch2 = nested(X2, y, outer, lambda tr: video_inner(vids[tr]), K, lambda a, b: bal_acc(a, b, K), tmp)
        assert ch["0"]["C"] == ch2["0"]["C"] and ch["0"]["grid"] == ch2["0"]["grid"]
        tmp.unlink()
    finally:
        fit_predict = orig
    # inner folds: whole videos
    tr = np.flatnonzero(outer != 0)
    g = video_inner(vids[tr])
    for v in np.unique(vids[tr]):
        assert len(set(g[vids[tr] == v])) == 1
    assert np.mean(probs.argmax(1) == y) > 0.6
    # bootstrap helpers
    W = video_boot_weights(np.arange(4), seed=0, b=50)
    assert (W.sum(1) == 4).all()
    yy = np.array([0, 1, 0, 1])
    assert np.allclose(boot_bal(yy, yy, np.arange(4), W, 2), 1.0)
    # real data: no leakage, no locked test, cache order
    rows, Z, ye, vids_e, outer_e, _ = ego_data()
    for f in np.unique(outer_e):
        assert not (set(vids_e[outer_e == f]) & set(vids_e[outer_e != f]))
    assert not (set(vids_e) & locked_uids())
    assert Z.shape[1:] == (8, 1152) and np.allclose(np.linalg.norm(Z[:5, 0], axis=1), 1, atol=1e-2)
    log(f"selftest PASS (ego {len(ye)} steps, {len(np.unique(vids_e))} videos, folds {len(np.unique(outer_e))})")
    state("selftest PASS")
    return 0


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["labels", "selftest", "ego", "libero", "table"])
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    return {"labels": cmd_labels, "selftest": cmd_selftest, "ego": cmd_ego, "libero": cmd_libero,
            "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
