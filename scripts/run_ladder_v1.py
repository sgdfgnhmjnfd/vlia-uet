#!/usr/bin/env python3
"""Zero-parameter ladder (Stage 3), early decision (Stage 4), cache-kNN (Stage 5).

Implements `results/prereg_ladder_v1.md` with amendment 1
(`results/prereg_ladder_v1_amendment_1.md`) on the Stage 2 cache written by
`scripts/extract_ladder_embeddings.py`. CPU / numpy only; train/val only.

Subcommands
  selftest  numpy-only checks (aggregators, C1/C2 leakage, nested selection,
            early-decision tau, kNN neighbours)
  gate      amended baseline reproduction gate (|hits - E1 hits| <= 2 per k)
  stage3    nested pipeline ladder        -> results/ladder_v1/
  stage4    evidence-accumulation early decision -> results/ead_v1/
  stage5    cache-kNN text-space correction -> results/knn_cache_v1/

Implementation choices not fixed by the preregistration (recorded in config.json):
  * C1 mean frame embedding = mean over all 8 per-frame embeddings of the
    training-fold queries; mean text embedding = mean of the training-fold
    queries' target WHY embeddings (template-specific). Frames and texts are
    centred per vector, then re-normalised.
  * C2 subtracts the context from the frames only (texts stay raw); a query
    without an earlier step uses C1 for both frames and texts.
  * A2 = 0.02 * logsumexp(cos / 0.02) over the first k frames.
  * K-way hits are strict (positive > every distractor), as in E1; invalid
    K-way rows count as misses, as in E1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_prefix_length_ablation as P  # noqa: E402
import run_scene_cv as CV  # noqa: E402

LADDER_ROOT = Path("/media/dhqg/d3/vlia_outputs/ladder")
MANIFEST = ROOT / "results/hoi_detection_v1/manifest.jsonl"
MINILM_CACHE = Path("/media/dhqg/d3/vlia_outputs/e1/text_embeddings.pt")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODERS = (
    "siglip-base-patch16-224",
    "siglip-base-patch16-384",
    "siglip2-base-patch16-224",
    "siglip-so400m-patch14-384",
    "clip-vit-large-patch14",
)
BASELINE_ENCODER = ENCODERS[0]
AGGS = ("A0", "A1", "A2")
CENTS = ("C0", "C1", "C2")
TMPLS = ("T0", "T1")
PREFIXES = (2, 4, 6, 8)
LSE_TEMP = 0.02
N_FOLDS = 8
E1_SUMMARY = ROOT / "results/e1_zero_shot_v1/summary.csv"
PREREGS = (
    "results/prereg_ladder_v1.md",
    "results/prereg_ladder_v1_amendment_1.md",
    "results/prereg_daily_v1.md",
    "results/prereg_daily_v1_addendum.md",
)
BUDGETS = (3, 4, 5, 6)
KNN_M = 20
KNN_TEMP = 0.05
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)


# ======================================================================
# numpy-only logic
# ======================================================================

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def aggregate(frames: np.ndarray, texts: np.ndarray, agg: str, k: int) -> np.ndarray:
    """frames (n, F, D) per-frame unit vectors; texts (m, D) shared or (n, c, D)
    per query. Returns (n, m) or (n, c) scores over the first k frames."""
    fk = frames[:, :k]
    if agg == "A0":
        q = normalize(fk.mean(axis=1))
        return q @ texts.T if texts.ndim == 2 else np.einsum("nd,ncd->nc", q, texts)
    cos = (
        np.einsum("nkd,md->nkm", fk, texts)
        if texts.ndim == 2
        else np.einsum("nkd,ncd->nkc", fk, texts)
    )
    if agg == "A1":
        return cos.max(axis=1)
    if agg == "A2":
        z = cos / LSE_TEMP
        zmax = z.max(axis=1, keepdims=True)
        return LSE_TEMP * (np.log(np.exp(z - zmax).sum(axis=1)) + zmax[:, 0])
    raise ValueError(agg)


def strict_hits(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid & (scores[:, :1] > scores[:, 1:]).all(axis=1)


def c1_means(frames: np.ndarray, text_targets: np.ndarray, train_idx: np.ndarray):
    """Means fitted on training-fold queries only."""
    d = frames.shape[-1]
    return frames[train_idx].reshape(-1, d).mean(axis=0), text_targets[train_idx].mean(axis=0)


def earlier_context(frames: np.ndarray, video_uids, steps: np.ndarray):
    """Mean of all frames of the same video at steps strictly earlier than each
    query's step. Returns (context (N, D), has_context (N,))."""
    n, _, d = frames.shape
    ctx = np.zeros((n, d), dtype=np.float64)
    has = np.zeros(n, dtype=bool)
    by_video = defaultdict(list)
    for i, v in enumerate(video_uids):
        by_video[v].append(i)
    for members in by_video.values():
        members = sorted(members, key=lambda i: steps[i])
        for pos, i in enumerate(members):
            earlier = [j for j in members[:pos] if steps[j] < steps[i]]
            assert all(steps[j] < steps[i] for j in earlier)
            if earlier:
                ctx[i] = frames[earlier].reshape(-1, d).mean(axis=0)
                has[i] = True
    return ctx.astype(np.float32), has


def processed(frames, text_raw, cent, train_idx, ctx=None, has_ctx=None):
    """Returns (frames', text_table_a, text_table_b, use_b (N,)): query i
    scores candidate texts from table b if use_b[i] else from table a."""
    n = len(frames)
    if cent == "C0":
        return frames, text_raw, text_raw, np.zeros(n, dtype=bool)
    mu_f, mu_t = c1_means(frames, text_raw, train_idx)
    f1 = normalize(frames - mu_f)
    t1 = normalize(text_raw - mu_t)
    if cent == "C1":
        return f1, t1, t1, np.zeros(n, dtype=bool)
    if cent == "C2":
        f2 = normalize(frames - ctx[:, None, :])
        out = np.where(has_ctx[:, None, None], f2, f1)
        return out, text_raw, t1, ~has_ctx
    raise ValueError(cent)


def candidate_scores(fr, ta, tb, use_b, idx, cands, agg, k):
    safe = np.where(cands >= 0, cands, 0)
    texts = np.where(use_b[idx][:, None, None], tb[safe], ta[safe])
    return aggregate(fr[idx], texts, agg, k)


def gallery_scores(fr, ta, tb, use_b, idx, agg, k):
    sa = aggregate(fr[idx], ta[idx], agg, k)
    if not use_b[idx].any():
        return sa
    sb = aggregate(fr[idx], tb[idx], agg, k)
    return np.where(use_b[idx][:, None], sb, sa)


def select_argmax(train_scores: np.ndarray) -> int:
    """Lowest grid index among ties (np.argmax returns the first maximum)."""
    return int(np.argmax(train_scores))


def ed_mean_frames(cummax: np.ndarray, tau: float) -> float:
    """cummax (n, 7): running max of margin over k=2..8. Stop at the first k
    with margin >= tau, else 8. Mean frames used."""
    return float(2 + (cummax[:, :6] < tau).sum(axis=1).mean())


def ed_stop_k(margins: np.ndarray, tau: float) -> np.ndarray:
    """margins (n, 7) for k=2..8 -> stopping k."""
    reached = margins >= tau
    first = np.where(reached.any(axis=1), reached.argmax(axis=1), 6)
    return first + 2


def choose_tau(margins_train: np.ndarray, budget: float) -> float:
    """Candidate taus are observed training margins; closest mean frames to
    the budget, ties broken by the larger tau."""
    cummax = np.maximum.accumulate(margins_train, axis=1)
    cands = np.unique(margins_train)
    # mean frames = 2 + sum_k P(cummax_k < tau), evaluated with sorted columns
    cols = np.sort(cummax[:, :6], axis=0)
    n = len(cummax)
    frames = 2 + sum(np.searchsorted(cols[:, j], cands, side="left") for j in range(6)) / n
    dist = np.abs(frames - budget)
    best = np.flatnonzero(dist == dist.min())
    return float(cands[best.max()])


def knn_yhat(x_query, x_bank, t_bank, m=KNN_M, temp=KNN_TEMP):
    sims = x_query @ x_bank.T
    m = min(m, x_bank.shape[0])
    top = np.argpartition(-sims, m - 1, axis=1)[:, :m]
    s = np.take_along_axis(sims, top, axis=1) / temp
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return normalize(np.einsum("nm,nmd->nd", w, t_bank[top]))


def selftest() -> None:
    rng = np.random.default_rng(0)
    # aggregators on hand-made arrays
    fr = np.zeros((1, 3, 2))
    fr[0, 0] = [1, 0]
    fr[0, 1] = [0, 1]
    fr[0, 2] = [0, 1]
    tx = np.array([[1.0, 0.0], [0.0, 1.0]])
    a0 = aggregate(fr, tx, "A0", 2)
    assert np.allclose(a0, [[np.sqrt(0.5), np.sqrt(0.5)]])
    assert np.allclose(aggregate(fr, tx, "A1", 2), [[1.0, 1.0]])
    a2 = aggregate(fr, tx, "A2", 3)
    assert np.allclose(a2[0, 1], LSE_TEMP * np.log(2 * np.exp(1 / LSE_TEMP) + 1), atol=1e-9)
    assert a2[0, 1] > a2[0, 0]
    assert np.allclose(aggregate(fr, tx[None].repeat(1, 0), "A1", 1), [[1.0, 0.0]])
    # C1 means never use the held-out fold
    frames = normalize(rng.normal(size=(20, 8, 4)))
    texts = normalize(rng.normal(size=(20, 4)))
    train = np.arange(10)
    mu_f, mu_t = c1_means(frames, texts, train)
    poisoned_f, poisoned_t = frames.copy(), texts.copy()
    poisoned_f[10:] = 1e6
    poisoned_t[10:] = -1e6
    mu_f2, mu_t2 = c1_means(poisoned_f, poisoned_t, train)
    assert np.allclose(mu_f, mu_f2) and np.allclose(mu_t, mu_t2)
    # C2 never uses a same-or-later step
    vids = ["a"] * 4 + ["b"] * 2
    steps = np.array([3, 1, 10, 2, 5, 5])
    fr6 = rng.normal(size=(6, 8, 4))
    ctx, has = earlier_context(fr6, vids, steps)
    assert has.tolist() == [True, False, True, True, False, False]
    assert np.allclose(ctx[0], fr6[[1, 3]].reshape(-1, 4).mean(0))
    assert np.allclose(ctx[2], fr6[[0, 1, 3]].reshape(-1, 4).mean(0))
    fr6b = fr6.copy()
    fr6b[2] = 99.0  # the latest step must not affect anyone else
    ctx_b, _ = earlier_context(fr6b, vids, steps)
    assert np.allclose(np.delete(ctx_b, 2, 0), np.delete(ctx, 2, 0))
    # nested selection never reads held-out labels: grid scores are computed
    # from training rows only, so permuting held-out targets changes nothing
    n = 40
    folds = np.repeat(np.arange(4), 10)
    fr40 = normalize(rng.normal(size=(n, 8, 4)))
    tx40 = normalize(rng.normal(size=(n, 4)))
    cands = np.stack([np.r_[i, rng.choice(np.delete(np.arange(n), i), 4, replace=False)] for i in range(n)])
    for i in range(n):  # candidates stay within the query's fold, as same-video sets do
        same = np.flatnonzero(folds == folds[i])
        cands[i] = np.r_[i, rng.choice(same[same != i], 4, replace=False)]

    def grid_scores(tx_, held):
        train_idx = np.flatnonzero(folds != held)
        out = []
        for agg in AGGS:
            for cent in ("C0", "C1"):
                fr_, ta, tb, ub = processed(fr40, tx_, cent, train_idx)
                s = candidate_scores(fr_, ta, tb, ub, train_idx, cands[train_idx], agg, 4)
                out.append(strict_hits(s, np.ones(len(train_idx), bool)).sum())
        return np.array(out)

    tx_perm = tx40.copy()
    tx_perm[folds == 0] = tx40[folds == 0][::-1]
    assert np.array_equal(grid_scores(tx40, 0), grid_scores(tx_perm, 0))
    assert select_argmax(np.array([1, 3, 3, 2])) == 1
    # early decision tau search matches brute force
    mg = rng.random((50, 7))
    for b in BUDGETS:
        tau = choose_tau(mg, b)
        brute = [abs(ed_stop_k(mg, t).mean() - b) for t in np.unique(mg)]
        assert abs(abs(ed_stop_k(mg, tau).mean() - b) - min(brute)) < 1e-12
        assert abs(ed_mean_frames(np.maximum.accumulate(mg, 1), tau) - ed_stop_k(mg, tau).mean()) < 1e-12
    # kNN uses only the bank it is given
    xb = normalize(rng.normal(size=(30, 4)))
    tb_ = normalize(rng.normal(size=(30, 4)))
    xq = normalize(rng.normal(size=(5, 4)))
    y1 = knn_yhat(xq, xb, tb_)
    assert y1.shape == (5, 4) and np.allclose(np.linalg.norm(y1, axis=1), 1)
    point, lo, hi = CV.video_bootstrap(np.repeat(np.arange(3), 2), np.ones(6), np.ones(6), B=100)
    assert point == lo == hi == 1.0
    print("selftest: PASS (aggregators, C1 train-only means, C2 strictly-earlier context, "
          "nested selection ignores held-out labels, tau search, kNN bank)")


# ======================================================================
# data
# ======================================================================

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        fields += [f for f in row if f not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Data:
    def __init__(self, encoders=ENCODERS, minilm=True):
        rows = [json.loads(line) for line in MANIFEST.open(encoding="utf-8") if line.strip()]
        if any(r["split"] not in ("train", "val") for r in rows):
            raise ValueError("disallowed split in manifest")
        self.rows = rows
        self.n = len(rows)
        self.sample_ids = [r["sample_id"] for r in rows]
        self.vids = np.asarray([r["video_uid"] for r in rows])
        self.scenes = np.asarray([r["scene"] for r in rows])
        self.texts = [r["why"] for r in rows]
        vocab = {t: i for i, t in enumerate(sorted(set(self.texts)))}
        self.text_id = np.asarray([vocab[t] for t in self.texts])
        if not all(r["step_id"].isdigit() for r in rows):
            raise ValueError("step_id not numeric: C2 must be skipped")
        self.steps = np.asarray([int(r["step_id"]) for r in rows])
        fold_map = CV.assign_folds(self.scenes.tolist(), N_FOLDS)
        self.fold_of = np.asarray([fold_map[s] for s in self.scenes])
        self.folds = sorted(set(self.fold_of.tolist()))
        for v in np.unique(self.vids):
            assert len(set(self.fold_of[self.vids == v])) == 1, "video crosses folds"
        self.cands5 = CV.build_kway(self.vids.tolist(), self.texts, self.sample_ids, 5)
        self.cands10 = CV.build_kway(self.vids.tolist(), self.texts, self.sample_ids, 10)
        self.valid5 = (self.cands5 >= 0).all(axis=1)
        self.valid10 = (self.cands10 >= 0).all(axis=1)
        self.mrr_chance = CV.gallery_chance(self.text_id, self.fold_of)["MRR"]
        self.frames: dict[str, np.ndarray] = {}
        self.text: dict[str, dict[str, np.ndarray]] = {}
        self.meta: dict[str, dict] = {}
        for enc in encoders:
            path = LADDER_ROOT / enc
            if not (path / "frame_emb.npy").exists():
                continue
            ids = json.loads((path / "sample_ids.json").read_text())
            pos = {s: i for i, s in enumerate(ids)}
            order = np.asarray([pos[s] for s in self.sample_ids])
            self.frames[enc] = np.load(path / "frame_emb.npy")[order].astype(np.float32)
            utexts = json.loads((path / "texts.json").read_text())
            tpos = {t: i for i, t in enumerate(utexts)}
            tidx = np.asarray([tpos[t] for t in self.texts])
            self.text[enc] = {t: np.load(path / f"text_{t}.npy")[tidx].astype(np.float32) for t in TMPLS}
            self.meta[enc] = json.loads((path / "meta.json").read_text())
        self.encoders = [e for e in encoders if e in self.frames]
        self.ctx = {}
        for enc in self.encoders:
            self.ctx[enc] = earlier_context(self.frames[enc], self.vids.tolist(), self.steps)
        if minilm:
            import run_zero_shot_alignment as Z

            mini = CV.embed_texts(self.texts, TEXT_MODEL, "cpu", MINILM_CACHE).astype(np.float32)
            self.mini = normalize(mini)
            self.clean5 = Z.clean_kway(self.vids.tolist(), self.texts, self.sample_ids, self.mini)
            self.clean_valid = (self.clean5 >= 0).all(axis=1)
            self.para_pos = {}
            self.para_chance = np.zeros(self.n)
            for f in self.folds:
                idx = np.flatnonzero(self.fold_of == f)
                pos_ = Z.paraphrase_positives(idx, self.texts, self.mini)
                self.para_pos[f] = pos_
                for li, i in enumerate(idx):
                    self.para_chance[i] = P._chance_one(len(idx), int(pos_[li].sum()))[0]

    def fold_idx(self, f):
        return np.flatnonzero(self.fold_of == f), np.flatnonzero(self.fold_of != f)

    def prep(self, enc, cent, tmpl, train_idx):
        ctx, has = self.ctx[enc]
        return processed(self.frames[enc], self.text[enc][tmpl], cent, train_idx, ctx, has)


def e1_expected():
    with E1_SUMMARY.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["branch"] == "full"]
    return {int(r["prefix_frames"]): float(r["5way_acc"]) for r in rows}


def baseline_hits(data: Data, k: int) -> np.ndarray:
    fr = data.frames[BASELINE_ENCODER]
    t = data.text[BASELINE_ENCODER]["T0"]
    all_idx = np.arange(data.n)
    s = candidate_scores(fr, t, t, np.zeros(data.n, bool), all_idx, data.cands5, "A0", k)
    return strict_hits(s, data.valid5)


def cmd_gate(args) -> int:
    data = Data(encoders=(BASELINE_ENCODER,), minilm=False)
    meta = data.meta[BASELINE_ENCODER]
    assert meta["frame_dtype"] == "float32", meta["frame_dtype"]
    assert data.frames[BASELINE_ENCODER].dtype == np.float32
    expected = e1_expected()
    rows = []
    passed = True
    for k in PREFIXES:
        hits = baseline_hits(data, k)
        exp_hits = int(round(expected[k] * data.n))
        assert abs(exp_hits / data.n - expected[k]) < 1e-12
        diff = int(hits.sum()) - exp_hits
        ok = abs(diff) <= 2
        passed &= ok
        rows.append({
            "k": k, "e1_hits": exp_hits, "float32_hits": int(hits.sum()), "delta": diff,
            "n_queries": data.n, "e1_acc": expected[k], "float32_acc": float(hits.mean()),
            "within_2": ok,
        })
        print(rows[-1])
    out = ROOT / "results/ladder_v1"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "gate": "amendment 1: |float32 K=5 hits - E1 hits| <= 2 for every k in {2,4,6,8}",
        "pass": bool(passed),
        "rows": rows,
        "embeddings": str(LADDER_ROOT / BASELINE_ENCODER),
        "frame_dtype": meta["frame_dtype"],
        "preregistration_sha256": {p: sha256(ROOT / p) for p in PREREGS},
        "git_commit": git_commit(),
    }
    (out / "gate_amended.json").write_text(json.dumps(result, indent=2))
    write_csv(out / "gate_amended.csv", rows)
    print("GATE", "PASS" if passed else "FAIL")
    return 0 if passed else 3


# ======================================================================
# Stage 3
# ======================================================================

def grid_list(encoders):
    return [(e, a, c, t) for e in encoders for a in AGGS for c in CENTS for t in TMPLS]


def cfg_name(cfg):
    return "|".join(cfg)


def eval_held_out(data: Data, cfg, fold, prepared=None, full=True):
    """Per-query metrics for the held-out fold under one config; C1 means from
    the fold's training queries."""
    enc, agg, cent, tmpl = cfg
    idx, train_idx = data.fold_idx(fold)
    fr, ta, tb, ub = prepared if prepared is not None else data.prep(enc, cent, tmpl, train_idx)
    out = {}
    for k in PREFIXES:
        gs = gallery_scores(fr, ta, tb, ub, idx, agg, k)
        ids = data.text_id[idx]
        ranks = P.first_positive_rank(gs, ids[:, None] == ids[None, :])
        m = {"rr": 1.0 / ranks}
        s5 = candidate_scores(fr, ta, tb, ub, idx, data.cands5[idx], agg, k)
        m["kway5"] = strict_hits(s5, data.valid5[idx]).astype(float)
        s10 = candidate_scores(fr, ta, tb, ub, idx, data.cands10[idx], agg, k)
        m["kway10"] = strict_hits(s10, data.valid10[idx]).astype(float)
        if full:
            sc = candidate_scores(fr, ta, tb, ub, idx, data.clean5[idx], agg, k)
            m["clean5"] = strict_hits(sc, data.clean_valid[idx]).astype(float)
            pr = P.first_positive_rank(gs, data.para_pos[fold])
            m["para_rr"] = 1.0 / pr
        out[k] = m
    return idx, out


def pool(data: Data, parts):
    """parts: list of (idx, {k: {metric: values}}) -> {k: {metric: (N,)}}"""
    pooled = {}
    for idx, res in parts:
        for k, metrics in res.items():
            for name, vals in metrics.items():
                pooled.setdefault(k, {}).setdefault(name, np.full(data.n, np.nan))[idx] = vals
    for k in pooled:
        for name, vals in pooled[k].items():
            assert not np.isnan(vals).any(), (k, name)
    return pooled


def summarize(data: Data, label, per_q, with_boot=True, full=True):
    rows = []
    n = data.n
    for k in PREFIXES:
        m = per_q[k]
        row = {
            "pipeline": label, "k": k, "n_queries": n,
            "n_videos": int(len(np.unique(data.vids))),
            "MRR": float(m["rr"].mean()), "MRR_chance": float(data.mrr_chance.mean()),
            "MRR_lift": float(m["rr"].mean() / data.mrr_chance.mean()),
            "5way_acc": float(m["kway5"].mean()), "5way_lift": float(m["kway5"].mean() / 0.2),
            "10way_acc": float(m["kway10"].mean()), "10way_lift": float(m["kway10"].mean() / 0.1),
            "5way_hits": int(m["kway5"].sum()),
        }
        if full:
            cv = data.clean_valid
            row["clean5_acc"] = float(m["clean5"][cv].mean())
            row["clean5_n_valid"] = int(cv.sum())
            row["para_MRR"] = float(m["para_rr"].mean())
            row["para_MRR_lift"] = float(m["para_rr"].mean() / data.para_chance.mean())
        if with_boot:
            specs = [
                ("MRR", data.vids, m["rr"], data.mrr_chance),
                ("5way", data.vids, m["kway5"], np.full(n, 0.2)),
                ("10way", data.vids, m["kway10"], np.full(n, 0.1)),
            ]
            if full:
                specs += [
                    ("clean5", data.vids[cv], m["clean5"][cv], np.full(int(cv.sum()), 0.2)),
                    ("para_MRR", data.vids, m["para_rr"], data.para_chance),
                ]
            for name, cl, num, den in specs:
                point, lo, hi = CV.video_bootstrap(cl, num, den)
                row[f"{name}_lift_boot"] = point
                row[f"{name}_lift_ci_lo"] = lo
                row[f"{name}_lift_ci_hi"] = hi
        rows.append(row)
    return rows


def paired(data: Data, label_a, a, label_b, b):
    rows = []
    for k in PREFIXES:
        for metric in ("rr", "kway5", "kway10"):
            delta = a[k][metric] - b[k][metric]
            point, lo, hi = CV.video_bootstrap(data.vids, delta, np.ones(data.n))
            rows.append({
                "comparison": f"{label_a} - {label_b}", "k": k,
                "metric": "MRR" if metric == "rr" else metric,
                "delta": point, "ci95_lo": lo, "ci95_hi": hi,
                "n_queries": data.n, "n_videos": int(len(np.unique(data.vids))),
            })
    return rows


def control_rows(data: Data, configs, fold_cfg):
    order = sorted(range(data.n), key=lambda i: data.sample_ids[i])
    ctrl = np.asarray(order[::8])
    donor = np.empty(len(ctrl), dtype=np.int64)
    for j, i in enumerate(ctrl):
        pool_ = np.flatnonzero((data.scenes != data.scenes[i]) & (data.vids != data.vids[i]))
        rng = np.random.RandomState(zlib.crc32(f"{data.sample_ids[i]}|control".encode()) & 0xFFFFFFFF)
        donor[j] = pool_[rng.randint(len(pool_))]
    assert all(data.scenes[d] != data.scenes[i] for d, i in zip(donor, ctrl))
    rows = []
    for cfg in configs:
        enc, agg, cent, tmpl = cfg
        real = np.zeros(len(ctrl))
        fake = np.zeros(len(ctrl))
        for f in data.folds:
            sel = np.flatnonzero(data.fold_of[ctrl] == f)
            if not len(sel):
                continue
            _, train_idx = data.fold_idx(f)
            fr, ta, tb, ub = data.prep(enc, cent, tmpl, train_idx)
            qi = ctrl[sel]
            sr = candidate_scores(fr, ta, tb, ub, qi, data.cands5[qi], agg, 8)
            real[sel] = strict_hits(sr, data.valid5[qi])
            # donor frames (with the donor's own C2 context / C2 mode), query's candidates
            di = donor[sel]
            safe = np.where(data.cands5[qi] >= 0, data.cands5[qi], 0)
            texts = np.where(ub[di][:, None, None], tb[safe], ta[safe])
            sf = aggregate(fr[di], texts, agg, 8)
            fake[sel] = strict_hits(sf, data.valid5[qi])
        rows.append({
            "config": cfg_name(cfg),
            "selected_in_folds": ",".join(str(f) for f in data.folds if fold_cfg.get(f) == cfg) or "baseline",
            "k": 8, "n_control_queries": len(ctrl),
            "real_5way_acc": float(real.mean()), "control_5way_acc": float(fake.mean()),
            "chance": 0.2,
        })
    return rows


def lift_plot(path: Path, rows_by_label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "MRR", "Gallery MRR lift"), (axes[1], "5way", "5-way accuracy lift")):
        for label, rows in rows_by_label.items():
            rows = sorted(rows, key=lambda r: r["k"])
            y = np.array([r[f"{metric}_lift_boot"] for r in rows])
            lo = np.array([r[f"{metric}_lift_ci_lo"] for r in rows])
            hi = np.array([r[f"{metric}_lift_ci_hi"] for r in rows])
            ax.errorbar(PREFIXES, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=label)
        ax.axhline(1.0, color="gray", ls=":", label="chance")
        ax.set_xticks(PREFIXES)
        ax.set_xlabel("observed frames k")
        ax.set_ylabel("lift over chance (95% video-bootstrap CI)")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cmd_stage3(args) -> int:
    gate = json.loads((ROOT / "results/ladder_v1/gate_amended.json").read_text())
    if not gate["pass"]:
        raise SystemExit("amended gate did not pass")
    data = Data()
    encs = data.encoders
    grid = grid_list(encs)
    print(f"encoders {encs}; grid {len(grid)} configs; C2 fallback fraction "
          f"{1 - data.ctx[encs[0]][1].mean():.4f}", flush=True)
    out = ROOT / "results/ladder_v1"
    baseline_cfg = (BASELINE_ENCODER, "A0", "C0", "T0")

    # training-fold K=5 hits for every config and every outer fold
    train_hits = np.zeros((len(data.folds), len(grid)), dtype=np.int64)
    descriptive_parts = defaultdict(list)
    for gi, cfg in enumerate(grid):
        enc, agg, cent, tmpl = cfg
        prepared_c0 = data.prep(enc, cent, tmpl, None) if cent == "C0" else None
        for fi, f in enumerate(data.folds):
            idx, train_idx = data.fold_idx(f)
            prep = prepared_c0 if prepared_c0 is not None else data.prep(enc, cent, tmpl, train_idx)
            fr, ta, tb, ub = prep
            total = 0
            for k in PREFIXES:
                s = candidate_scores(fr, ta, tb, ub, train_idx, data.cands5[train_idx], agg, k)
                total += int(strict_hits(s, data.valid5[train_idx]).sum())
            train_hits[fi, gi] = total
            descriptive_parts[cfg].append(eval_held_out(data, cfg, f, prepared=prep, full=False))
        print(f"grid {gi + 1}/{len(grid)} {cfg_name(cfg)}", flush=True)

    fold_cfg = {}
    sel_rows = []
    nested_parts = []
    for fi, f in enumerate(data.folds):
        gi = select_argmax(train_hits[fi])
        cfg = grid[gi]
        fold_cfg[f] = cfg
        idx, train_idx = data.fold_idx(f)
        nested_parts.append(eval_held_out(data, cfg, f))
        sel_rows.append({
            "fold": f, "held_out_scenes": ",".join(sorted(set(data.scenes[idx]))),
            "n_held_out": len(idx), "n_train": len(train_idx), "grid_index": gi,
            "encoder": cfg[0], "aggregator": cfg[1], "centering": cfg[2], "template": cfg[3],
            "train_mean_5way_acc": train_hits[fi, gi] / (4 * len(train_idx)),
            "baseline_train_mean_5way_acc": train_hits[fi, grid.index(baseline_cfg)] / (4 * len(train_idx)),
            "n_tied_at_max": int((train_hits[fi] == train_hits[fi, gi]).sum()),
        })
    nested = pool(data, nested_parts)
    base = pool(data, [eval_held_out(data, baseline_cfg, f) for f in data.folds])
    sum_rows = summarize(data, "baseline", base) + summarize(data, "nested_selected", nested)
    pair_rows = paired(data, "nested_selected", nested, "baseline", base)
    counts = Counter(cfg_name(c) for c in fold_cfg.values())
    stability = {
        "n_folds": len(data.folds), "n_distinct_selected": len(counts),
        "most_common": counts.most_common(1)[0][0], "most_common_folds": counts.most_common(1)[0][1],
        "counts": dict(counts),
        "per_component": {
            name: dict(Counter(c[i] for c in fold_cfg.values()))
            for i, name in enumerate(("encoder", "aggregator", "centering", "template"))
        },
    }

    desc_rows = []
    for gi, cfg in enumerate(grid):
        pq = pool(data, descriptive_parts[cfg])
        for r in summarize(data, cfg_name(cfg), pq, with_boot=False, full=False):
            r = {"grid_index": gi, **r, "note": "not selection-corrected; no verdict"}
            desc_rows.append(r)

    ctrl_cfgs = list(dict.fromkeys([*fold_cfg.values(), baseline_cfg]))
    ctrl = control_rows(data, ctrl_cfgs, fold_cfg)

    # verdicts
    def passing(metric, cond):
        return [r["k"] for r in pair_rows if r["metric"] == metric and cond(r)]

    n1_mrr = passing("MRR", lambda r: r["ci95_lo"] > 0)
    n1_k5 = passing("kway5", lambda r: r["ci95_lo"] > 0)
    n1 = len(n1_mrr) >= 2 and len(n1_k5) >= 2
    n1_joint = sorted(set(n1_mrr) & set(n1_k5))
    nest = [r for r in sum_rows if r["pipeline"] == "nested_selected"]
    n2_acc = [r["k"] for r in nest if r["5way_acc"] >= 0.30]
    n2_mrr = [r["k"] for r in nest if r["MRR_lift_ci_lo"] > 1]
    n2_k5 = [r["k"] for r in nest if r["5way_lift_ci_lo"] > 1]
    n2 = len(n2_acc) >= 2 and len(n2_mrr) >= 2 and len(n2_k5) >= 2
    verdicts = {
        "N1": {
            "pass": n1, "label": "Selection helps" if n1 else "Selection does not help (rule not met)",
            "MRR_ci_lo_gt0_prefixes": n1_mrr, "5way_ci_lo_gt0_prefixes": n1_k5,
            "rule": "per metric: CI lower bound > 0 at >= 2 of 4 prefixes, for both MRR and 5-way",
            "both_metrics_same_prefix": n1_joint,
        },
        "N2": {
            "pass": n2, "5way_acc_ge_0.30_prefixes": n2_acc,
            "MRR_lift_ci_lo_gt1_prefixes": n2_mrr, "5way_lift_ci_lo_gt1_prefixes": n2_k5,
            "G1": "FAIL (unchanged)",
        },
    }

    write_csv(out / "summary.csv", sum_rows)
    write_csv(out / "selected_configs.csv", sel_rows)
    write_csv(out / "paired_vs_baseline.csv", pair_rows)
    write_csv(out / "grid_descriptive.csv", desc_rows)
    write_csv(out / "control.csv", ctrl)
    lift_plot(out / "lift_vs_k.png", {
        "baseline": [r for r in sum_rows if r["pipeline"] == "baseline"],
        "nested-selected": nest,
    })
    write_stage3_md(out / "summary.md", sum_rows, pair_rows, sel_rows, ctrl, verdicts, stability)
    config = {
        "stage": "ladder_v1 Stage 3",
        "encoders_evaluated": encs,
        "encoders_dropped": [e for e in ENCODERS if e not in encs],
        "grid_size": len(grid), "grid_order": "encoder x aggregator x centering x template (nested, listed order)",
        "folds": "run_scene_cv.assign_folds(scenes, 8)",
        "selection": "argmax of summed training-fold K=5 hits over k in {2,4,6,8}; ties -> lowest grid index",
        "baseline": cfg_name(baseline_cfg), "baseline_frame_dtype": data.meta[BASELINE_ENCODER]["frame_dtype"],
        "c2_fallback_fraction": float(1 - data.ctx[encs[0]][1].mean()),
        "lse_temperature": LSE_TEMP, "bootstrap": {"B": 10000, "seed": 0, "cluster": "video_uid"},
        "implementation_choices": __doc__.split("Implementation choices not fixed by the preregistration (recorded in config.json):")[1].strip(),
        "selection_stability": stability, "verdicts": verdicts,
        "preregistration_sha256": {p: sha256(ROOT / p) for p in PREREGS},
        "git_commit": git_commit(), "test_split_read": False,
        "encoder_meta": {e: {k: v for k, v in data.meta[e].items() if k != "throughput"} for e in encs},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(json.dumps(verdicts, indent=2))
    return 0


def fmt_ci(r, metric):
    return f"{r[f'{metric}_lift_boot']:.3f} [{r[f'{metric}_lift_ci_lo']:.3f}, {r[f'{metric}_lift_ci_hi']:.3f}]"


def write_stage3_md(path, sum_rows, pair_rows, sel_rows, ctrl, verdicts, stability):
    lines = [
        "# Ladder v1 Stage 3: nested pipeline ladder",
        "",
        "Out-of-fold, scene-grouped 8-fold; train/val only. Lifts: video bootstrap B=10000, seed 0.",
        "",
        "| Pipeline | k | MRR | MRR lift [95% CI] | 5-way acc | 5-way lift [95% CI] | 10-way acc | 10-way lift [95% CI] |",
        "|---|---:|---:|---|---:|---|---:|---|",
    ]
    for r in sum_rows:
        lines.append(
            f"| {r['pipeline']} | {r['k']} | {r['MRR']:.4f} | {fmt_ci(r, 'MRR')} | {r['5way_acc']:.4f} | "
            f"{fmt_ci(r, '5way')} | {r['10way_acc']:.4f} | {fmt_ci(r, '10way')} |"
        )
    lines += ["", "## Secondary (no verdict)", "",
              "| Pipeline | k | clean 5-way acc | clean 5-way lift [95% CI] | para MRR | para MRR lift [95% CI] |",
              "|---|---:|---:|---|---:|---|"]
    for r in sum_rows:
        lines.append(f"| {r['pipeline']} | {r['k']} | {r['clean5_acc']:.4f} | {fmt_ci(r, 'clean5')} | "
                     f"{r['para_MRR']:.4f} | {fmt_ci(r, 'para_MRR')} |")
    lines += ["", "## Paired: nested-selected minus baseline", "",
              "| k | metric | delta [95% CI] |", "|---:|---|---|"]
    for r in pair_rows:
        lines.append(f"| {r['k']} | {r['metric']} | {r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |")
    lines += ["", "## Selected configuration per outer fold", "",
              "| fold | held-out scenes | encoder | agg | centering | template | train mean 5-way | baseline train mean 5-way | ties |",
              "|---:|---|---|---|---|---|---:|---:|---:|"]
    for r in sel_rows:
        lines.append(f"| {r['fold']} | {r['held_out_scenes']} | {r['encoder']} | {r['aggregator']} | {r['centering']} | "
                     f"{r['template']} | {r['train_mean_5way_acc']:.4f} | {r['baseline_train_mean_5way_acc']:.4f} | {r['n_tied_at_max']} |")
    lines += ["", f"Selection stability: {stability['n_distinct_selected']} distinct configs over "
              f"{stability['n_folds']} folds; most common `{stability['most_common']}` in {stability['most_common_folds']} folds.",
              "", "## Mismatched-video control (every 8th query by sample_id, k=8)", "",
              "| config | selected in folds | n | real 5-way | control 5-way |", "|---|---|---:|---:|---:|"]
    for r in ctrl:
        lines.append(f"| {r['config']} | {r['selected_in_folds']} | {r['n_control_queries']} | "
                     f"{r['real_5way_acc']:.4f} | {r['control_5way_acc']:.4f} |")
    lines += ["", "## Verdicts", "", "```", json.dumps(verdicts, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# Stage 4
# ======================================================================

def baseline_margins(data: Data):
    fr = data.frames[BASELINE_ENCODER]
    t = data.text[BASELINE_ENCODER]["T0"]
    all_idx = np.arange(data.n)
    margins = np.zeros((data.n, 8))
    hits = np.zeros((data.n, 8), dtype=bool)
    for k in range(1, 9):
        s = candidate_scores(fr, t, t, np.zeros(data.n, bool), all_idx, data.cands5, "A0", k)
        srt = np.sort(s, axis=1)
        margins[:, k - 1] = srt[:, -1] - srt[:, -2]
        hits[:, k - 1] = strict_hits(s, data.valid5)
    return margins, hits


def cmd_stage4(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = json.loads((ROOT / "results/ladder_v1/gate_amended.json").read_text())
    if not gate["pass"]:
        raise SystemExit("amended gate did not pass")
    data = Data(encoders=(BASELINE_ENCODER,), minilm=False)
    out = ROOT / "results/ead_v1"
    out.mkdir(parents=True, exist_ok=True)
    margins, hits = baseline_margins(data)
    m28 = margins[:, 1:]  # k = 2..8
    fixed_acc = {k: float(hits[:, k - 1].mean()) for k in range(2, 9)}
    rows, tau_rows, paired_rows = [], [], []
    plot_pts = []
    for b in BUDGETS:
        stop = np.zeros(data.n, dtype=np.int64)
        for f in data.folds:
            idx, train_idx = data.fold_idx(f)
            tau = choose_tau(m28[train_idx], b)
            stop[idx] = ed_stop_k(m28[idx], tau)
            tau_rows.append({"budget": b, "fold": f, "tau": tau,
                             "train_mean_frames": float(ed_stop_k(m28[train_idx], tau).mean()),
                             "held_out_mean_frames": float(stop[idx].mean())})
        ed_correct = hits[np.arange(data.n), stop - 1].astype(float)
        mf = float(stop.mean())
        lo_k = int(np.clip(np.floor(mf), 2, 8))
        hi_k = int(np.clip(lo_k + 1, 2, 8))
        w = mf - lo_k if hi_k != lo_k else 0.0
        interp = (1 - w) * hits[:, lo_k - 1] + w * hits[:, hi_k - 1]
        point, lo, hi = CV.video_bootstrap(data.vids, ed_correct - interp, np.ones(data.n))
        acc_pt, acc_lo, acc_hi = CV.video_bootstrap(data.vids, ed_correct, np.ones(data.n))
        rows.append({
            "budget": b, "realized_mean_frames": mf, "early_decision_acc": float(ed_correct.mean()),
            "early_decision_acc_ci_lo": acc_lo, "early_decision_acc_ci_hi": acc_hi,
            "fixed_k_interpolated_acc": float(interp.mean()),
            "delta": point, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "stop_k_distribution": json.dumps({int(k): int(c) for k, c in sorted(Counter(stop.tolist()).items())}),
            "n_queries": data.n,
        })
        plot_pts.append((mf, float(ed_correct.mean()), acc_lo, acc_hi))
    passing = [r["budget"] for r in rows if r["delta_ci_lo"] > 0]
    s2 = len(passing) >= 2
    fixed_rows = []
    for k in range(1, 9):
        pt, lo, hi = CV.video_bootstrap(data.vids, hits[:, k - 1].astype(float), np.ones(data.n))
        fixed_rows.append({"k": k, "fixed_k_acc": pt, "ci_lo": lo, "ci_hi": hi,
                           "mean_margin": float(margins[:, k - 1].mean())})
    write_csv(out / "summary.csv", rows)
    write_csv(out / "fixed_k.csv", fixed_rows)
    write_csv(out / "thresholds.csv", tau_rows)
    np.savez_compressed(out / "per_query_margins.npz", sample_ids=np.asarray(data.sample_ids),
                        margins=margins, hits=hits)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ks = list(range(2, 9))
    ax.plot(ks, [fixed_acc[k] for k in ks], marker="o", label="fixed-k (baseline)")
    for mf, acc, lo, hi in plot_pts:
        ax.errorbar([mf], [acc], yerr=[[acc - lo], [hi - acc]], marker="s", color="C3", capsize=3)
    ax.plot([], [], "s", color="C3", label="early decision (budgets 3-6)")
    ax.axhline(0.2, color="gray", ls=":", label="chance (0.20)")
    ax.set_xlabel("mean frames used")
    ax.set_ylabel("5-way accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_frames.png", dpi=160)
    plt.close(fig)
    verdict = {"S2": {"pass": s2, "budgets_ci_lo_gt0": passing,
                      "rule": "early-decision minus interpolated fixed-k: CI lower bound > 0 at >= 2 of 4 budgets"}}
    config = {
        "stage": "ladder_v1 Stage 4 (early decision)", "pipeline": f"{BASELINE_ENCODER}|A0|C0|T0 (float32)",
        "budgets": BUDGETS, "tau_candidates": "observed training-fold margins at k=2..8",
        "tau_tie_break": "closest training mean frames to budget; ties -> larger tau",
        "comparator": "per-query linear interpolation of fixed-k hits at the pooled realized mean frames",
        "bootstrap": {"B": 10000, "seed": 0, "cluster": "video_uid"}, "verdicts": verdict,
        "preregistration_sha256": {p: sha256(ROOT / p) for p in PREREGS},
        "git_commit": git_commit(), "test_split_read": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    lines = ["# Early decision (Stage 4)", "", "| budget | mean frames | ED acc [95% CI] | fixed-k interp. acc | delta [95% CI] | stop-k counts |",
             "|---:|---:|---|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['budget']} | {r['realized_mean_frames']:.3f} | {r['early_decision_acc']:.4f} "
                     f"[{r['early_decision_acc_ci_lo']:.4f}, {r['early_decision_acc_ci_hi']:.4f}] | "
                     f"{r['fixed_k_interpolated_acc']:.4f} | {r['delta']:+.4f} [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | {r['stop_k_distribution']} |")
    lines += ["", "| k | fixed-k acc [95% CI] | mean margin |", "|---:|---|---:|"]
    for r in fixed_rows:
        lines.append(f"| {r['k']} | {r['fixed_k_acc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['mean_margin']:.4f} |")
    lines += ["", f"S2: {'PASS' if s2 else 'FAIL'} (budgets with CI lower bound > 0: {passing})", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps(verdict, indent=2))
    return 0


# ======================================================================
# Stage 5
# ======================================================================

def prefix_emb(frames, k):
    return normalize(frames[:, :k].mean(axis=1))


def knn_scores(x_q, t_cands, yhat, beta):
    """x_q (n,D); t_cands (n,c,D) or (m,D); yhat (n,D)."""
    if t_cands.ndim == 2:
        return x_q @ t_cands.T + beta * (yhat @ t_cands.T)
    return np.einsum("nd,ncd->nc", x_q, t_cands) + beta * np.einsum("nd,ncd->nc", yhat, t_cands)


def cmd_stage5(args) -> int:
    if not (ROOT / "results/ladder_v1/summary.csv").exists():
        raise SystemExit("Stage 5 runs only after Stage 3")
    data = Data(encoders=(BASELINE_ENCODER,), minilm=False)
    out = ROOT / "results/knn_cache_v1"
    out.mkdir(parents=True, exist_ok=True)
    fr = data.frames[BASELINE_ENCODER]
    t0 = data.text[BASELINE_ENCODER]["T0"]
    X = {k: prefix_emb(fr, k) for k in PREFIXES}
    knn_parts, base_parts, beta_rows = [], [], []
    for f in data.folds:
        idx, train_idx = data.fold_idx(f)
        assert not np.isin(train_idx, idx).any()
        # beta selection: leave-one-scene-out inside the training scenes
        hits_by_beta = np.zeros(len(BETAS), dtype=np.int64)
        for s in sorted(set(data.scenes[train_idx])):
            q = train_idx[data.scenes[train_idx] == s]
            bank = train_idx[data.scenes[train_idx] != s]
            for k in PREFIXES:
                yhat = knn_yhat(X[k][q], X[k][bank], t0[bank])
                safe = np.where(data.cands5[q] >= 0, data.cands5[q], 0)
                for bi, beta in enumerate(BETAS):
                    sc = knn_scores(X[k][q], t0[safe], yhat, beta)
                    hits_by_beta[bi] += int(strict_hits(sc, data.valid5[q]).sum())
        bi = int(np.argmax(hits_by_beta))
        beta = BETAS[bi]
        beta_rows.append({"fold": f, "beta": beta, **{f"train_loso_mean_5way_beta_{b}": h / (4 * len(train_idx))
                                                      for b, h in zip(BETAS, hits_by_beta)}})
        res_k, base_k = {}, {}
        ids = data.text_id[idx]
        pos = ids[:, None] == ids[None, :]
        for k in PREFIXES:
            yhat = knn_yhat(X[k][idx], X[k][train_idx], t0[train_idx])
            gal = knn_scores(X[k][idx], t0[idx], yhat, beta)
            m = {"rr": 1.0 / P.first_positive_rank(gal, pos)}
            gb = X[k][idx] @ t0[idx].T
            mb = {"rr": 1.0 / P.first_positive_rank(gb, pos)}
            for K, cands, valid in ((5, data.cands5, data.valid5), (10, data.cands10, data.valid10)):
                safe = np.where(cands[idx] >= 0, cands[idx], 0)
                m[f"kway{K}"] = strict_hits(knn_scores(X[k][idx], t0[safe], yhat, beta), valid[idx]).astype(float)
                mb[f"kway{K}"] = strict_hits(knn_scores(X[k][idx], t0[safe], yhat, 0.0), valid[idx]).astype(float)
            res_k[k], base_k[k] = m, mb
        knn_parts.append((idx, res_k))
        base_parts.append((idx, base_k))
        print(f"fold {f}: beta={beta}", flush=True)
    knn = pool(data, knn_parts)
    base = pool(data, base_parts)
    sum_rows = summarize(data, "baseline", base, full=False) + summarize(data, "knn_beta_selected", knn, full=False)
    pair_rows = paired(data, "knn_beta_selected", knn, "baseline", base)
    k1_mrr = [r["k"] for r in pair_rows if r["metric"] == "MRR" and r["ci95_lo"] > 0]
    k1_k5 = [r["k"] for r in pair_rows if r["metric"] == "kway5" and r["ci95_lo"] > 0]
    k1 = len(k1_mrr) >= 2 and len(k1_k5) >= 2
    verdict = {"K1": {"pass": k1, "MRR_ci_lo_gt0_prefixes": k1_mrr, "5way_ci_lo_gt0_prefixes": k1_k5,
                      "rule": "per metric: CI lower bound > 0 at >= 2 of 4 prefixes, for both MRR and 5-way"}}
    write_csv(out / "summary.csv", sum_rows)
    write_csv(out / "paired_vs_baseline.csv", pair_rows)
    write_csv(out / "selected_beta.csv", beta_rows)
    lines = ["# Cache-kNN text-space correction (Stage 5)", "",
             "| Pipeline | k | MRR | MRR lift [95% CI] | 5-way acc | 5-way lift [95% CI] | 10-way acc |", "|---|---:|---:|---|---:|---|---:|"]
    for r in sum_rows:
        lines.append(f"| {r['pipeline']} | {r['k']} | {r['MRR']:.4f} | {fmt_ci(r, 'MRR')} | {r['5way_acc']:.4f} | {fmt_ci(r, '5way')} | {r['10way_acc']:.4f} |")
    lines += ["", "| k | metric | delta [95% CI] |", "|---:|---|---|"]
    for r in pair_rows:
        lines.append(f"| {r['k']} | {r['metric']} | {r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |")
    lines += ["", "Selected beta per fold: " + ", ".join(f"fold {r['fold']}: {r['beta']}" for r in beta_rows),
              "", f"K1: {'PASS' if k1 else 'FAIL'}", ""]
    (out / "summary.md").write_text("\n".join(lines))
    config = {
        "stage": "ladder_v1 Stage 5 (cache-kNN)", "pipeline": f"{BASELINE_ENCODER}|A0|C0|T0 (float32)",
        "M": KNN_M, "temperature": KNN_TEMP, "betas": BETAS,
        "beta_selection": "leave-one-scene-out within training scenes; max summed 5-way hits over k; ties -> lowest beta",
        "neighbours": "training-fold queries at the same k; never the held-out fold",
        "bootstrap": {"B": 10000, "seed": 0, "cluster": "video_uid"}, "verdicts": verdict,
        "note": "uses training-scene labels: a non-parametric trained method",
        "preregistration_sha256": {p: sha256(ROOT / p) for p in PREREGS},
        "git_commit": git_commit(), "test_split_read": False,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("selftest", "gate", "stage3", "stage4", "stage5"))
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        return 0
    return {"gate": cmd_gate, "stage3": cmd_stage3, "stage4": cmd_stage4, "stage5": cmd_stage5}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
