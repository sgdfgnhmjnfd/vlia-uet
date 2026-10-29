#!/usr/bin/env python3
"""Supplementary v3 experiments on the EgoIntent train/val cache (CPU only).

Rules are fixed in `results/prereg_v3_supplementary.md` (written before any
number). Reference pipeline F = siglip-so400m-patch14-384 | A0 | C2 | T1, the
frozen config. All numbers are scene-grouped 8-fold out-of-fold on the 2503
train/val queries; the locked test split is never read.

Subcommands
  selftest  numpy/torch checks on synthetic data (weights, fusion limits,
            adapter identity at init, NEXT label join)
  gate      F must reproduce 853/874/907/969 out-of-fold 5-way hits (k=2,4,6,8)
  stage1    aggregation x C0/C2 score fusion, nested selection -> results/v3_stage1_aggfusion/
  stage2    hard-negative residual adapter, nested selection   -> results/v3_stage2_hn_adapter/
  next      predict the observed NEXT step (EgoIntent label)   -> results/v3_next_step/

Every subcommand except selftest runs the gate first and stops if it fails.
Typical run on the lab machine (GPU busy with training, so everything on CPU):
  export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
  $PY scripts/run_v3_supplementary.py selftest
  $PY scripts/run_v3_supplementary.py stage1
  $PY scripts/run_v3_supplementary.py next
  $PY scripts/run_v3_supplementary.py stage2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ladder_v1 as L  # noqa: E402
import run_prefix_length_ablation as P  # noqa: E402
import run_scene_cv as CV  # noqa: E402

ENC = "siglip-so400m-patch14-384"
ENCODER_ID = "google/siglip-so400m-patch14-384"
PREFIXES = (2, 4, 6, 8)
F_EXPECTED_HITS = {2: 853, 4: 874, 6: 907, 8: 969}
PREREG = ROOT / "results/prereg_v3_supplementary.md"
B = 10000

# Stage 1 grid. F (A0, lambda = inf) is index 0 so that ties go to F.
AGGS1 = ("A0", "A3", "A4h1", "A4h2", "A4h4")
LAMBDAS = (np.inf, 0.0, 0.25, 0.5, 1.0, 2.0)
GRID1 = [("A0", np.inf)] + [(a, l) for a in AGGS1 for l in LAMBDAS if (a, l) != ("A0", np.inf)]

# Stage 2 grid (weight decay, epochs); index 0 first among ties.
GRID2 = [(wd, ep) for ep in (5, 20) for wd in (1e-2, 1e-1, 1.0)]
SEEDS2 = (0, 1, 2)
TAU2 = 0.05
N_RANDOM_NEG = 64
BATCH2 = 64


# ======================================================================
# scoring (numpy)
# ======================================================================

def agg_weights(agg: str, k: int) -> np.ndarray:
    if agg == "A0":
        w = np.ones(k)
    elif agg == "A3":
        w = np.zeros(k)
        w[-1] = 1.0
    elif agg.startswith("A4h"):
        h = float(agg[3:])
        w = np.exp(-(k - 1 - np.arange(k)) / h)
    else:
        raise ValueError(agg)
    return w / w.sum()


def query_vec(frames: np.ndarray, agg: str, k: int) -> np.ndarray:
    """frames (n, 8, D) per-frame unit vectors -> (n, D) unit query."""
    return L.normalize(np.einsum("nkd,k->nd", frames[:, :k], agg_weights(agg, k)))


def cand_scores(q, ta, tb, ub, idx, cands):
    """q (len(idx), D); per-query text table b if ub else a."""
    safe = np.where(cands >= 0, cands, 0)
    texts = np.where(ub[idx][:, None, None], tb[safe], ta[safe])
    return np.einsum("nd,ncd->nc", q, texts)


def gal_scores(q, ta, tb, ub, idx):
    sa = q @ ta[idx].T
    sb = q @ tb[idx].T
    return np.where(ub[idx][:, None], sb, sa)


def fuse(s0, s2, lam):
    return s2 if np.isinf(lam) else s0 + lam * s2


def metrics_for(data, idx, s5, s10, gs):
    ids = data.text_id[idx]
    ranks = P.first_positive_rank(gs, ids[:, None] == ids[None, :])
    return {
        "rr": 1.0 / ranks,
        "kway5": L.strict_hits(s5, data.valid5[idx]).astype(float),
        "kway10": L.strict_hits(s10, data.valid10[idx]).astype(float),
    }


class FoldPrep:
    """C0 and C2 inputs for one outer fold (C1 means from its training queries)."""

    def __init__(self, data, fold):
        self.idx, self.train_idx = data.fold_idx(fold)
        self.c0 = data.prep(ENC, "C0", "T1", self.train_idx)
        self.c2 = data.prep(ENC, "C2", "T1", self.train_idx)

    def scores(self, data, qidx, agg, lam, k, gallery_idx=None):
        out = {}
        parts = {}
        for name, (fr, ta, tb, ub) in (("c0", self.c0), ("c2", self.c2)):
            if name == "c0" and np.isinf(lam):
                continue
            q = query_vec(fr[qidx], agg, k)
            parts[name] = (
                cand_scores(q, ta, tb, ub, qidx, data.cands5[qidx]),
                cand_scores(q, ta, tb, ub, qidx, data.cands10[qidx]),
                gal_scores(q, ta, tb, ub, qidx) if gallery_idx is not None else None,
            )
        for j, key in enumerate(("s5", "s10", "gs")):
            s2 = parts["c2"][j]
            if s2 is None:
                out[key] = None
                continue
            out[key] = fuse(parts["c0"][j], s2, lam) if "c0" in parts else s2
        return out


def eval_config(data, prep, agg, lam, qidx, with_gallery=True):
    res = {}
    for k in PREFIXES:
        s = prep.scores(data, qidx, agg, lam, k, gallery_idx=qidx if with_gallery else None)
        if with_gallery:
            res[k] = metrics_for(data, qidx, s["s5"], s["s10"], s["gs"])
        else:
            res[k] = {"kway5": L.strict_hits(s["s5"], data.valid5[qidx]).astype(float)}
    return res


def pool_parts(n, parts):
    pooled = {}
    for idx, res in parts:
        for k, m in res.items():
            for name, vals in m.items():
                pooled.setdefault(k, {}).setdefault(name, np.full(n, np.nan))[idx] = vals
    return pooled


# ======================================================================
# reporting
# ======================================================================

def summary_rows(data, label, per_q):
    rows = []
    for k in PREFIXES:
        m = per_q[k]
        row = {"pipeline": label, "k": k, "n_queries": data.n,
               "5way_acc": float(np.mean(m["kway5"])), "10way_acc": float(np.mean(m["kway10"])),
               "MRR": float(np.mean(m["rr"]))}
        for name, num, den in (("5way", m["kway5"], np.full(data.n, 0.2)),
                               ("10way", m["kway10"], np.full(data.n, 0.1)),
                               ("MRR", m["rr"], data.mrr_chance)):
            p, lo, hi = CV.video_bootstrap(data.vids, num, den, B=B)
            row[f"{name}_lift"], row[f"{name}_lift_lo"], row[f"{name}_lift_hi"] = p, lo, hi
        rows.append(row)
    return rows


def paired_rows(data, label, a, f):
    rows = []
    for k in PREFIXES:
        for metric in ("kway5", "kway10", "rr"):
            delta = np.asarray(a[k][metric]) - np.asarray(f[k][metric])
            p, lo, hi = CV.video_bootstrap(data.vids, delta, np.ones(data.n), B=B)
            rows.append({"comparison": f"{label} - F", "k": k,
                         "metric": "MRR" if metric == "rr" else metric,
                         "delta": p, "ci95_lo": lo, "ci95_hi": hi})
    return rows


def improves_on_f(pair_rows):
    five = [r for r in pair_rows if r["metric"] == "kway5"]
    mrr = [r for r in pair_rows if r["metric"] == "MRR"]
    n5 = sum(r["ci95_lo"] > 0 for r in five)
    mrr_ok = all(r["ci95_hi"] > 0 for r in mrr)
    return {"5way_ci_lo_gt0_count": n5, "mrr_not_clearly_worse_all_k": mrr_ok,
            "PASS": bool(n5 >= 2 and mrr_ok)}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def out_dir(name: str, overwrite: bool) -> Path:
    out = ROOT / "results" / name
    if (out / "summary.csv").exists() and not overwrite:
        raise FileExistsError(f"{out}/summary.csv exists (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_data():
    data = L.Data(encoders=(ENC,), minilm=False)
    if ENC not in data.frames:
        raise FileNotFoundError(f"ladder cache for {ENC} not found under {L.LADDER_ROOT}")
    return data


def f_out_of_fold(data):
    parts = []
    for fold in data.folds:
        prep = FoldPrep(data, fold)
        parts.append((prep.idx, eval_config(data, prep, "A0", np.inf, prep.idx)))
    return pool_parts(data.n, parts)


def gate(data, f_per_q=None) -> dict:
    f_per_q = f_per_q or f_out_of_fold(data)
    hits = {k: int(f_per_q[k]["kway5"].sum()) for k in PREFIXES}
    ok = hits == F_EXPECTED_HITS
    print(f"GATE F hits {hits} expected {F_EXPECTED_HITS}: {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("gate failed: F does not reproduce; stopping")
    return f_per_q


# ======================================================================
# Stage 1
# ======================================================================

def cmd_stage1(args) -> int:
    data = load_data()
    f_per_q = gate(data)
    out = out_dir("v3_stage1_aggfusion", args.overwrite)
    parts, selections = [], []
    for fold in data.folds:
        prep = FoldPrep(data, fold)
        t0 = time.time()
        train_scores = []
        for agg, lam in GRID1:
            r = eval_config(data, prep, agg, lam, prep.train_idx, with_gallery=False)
            train_scores.append(np.mean([r[k]["kway5"].mean() for k in PREFIXES]))
        best = L.select_argmax(np.asarray(train_scores))
        agg, lam = GRID1[best]
        selections.append({"fold": fold, "aggregator": agg, "lambda": str(lam),
                           "train_mean_5way": float(train_scores[best]),
                           "train_mean_5way_F": float(train_scores[0])})
        parts.append((prep.idx, eval_config(data, prep, agg, lam, prep.idx)))
        print(f"fold {fold}: selected {agg} lambda={lam} ({time.time() - t0:.0f}s)", flush=True)
    sel = pool_parts(data.n, parts)
    rows = summary_rows(data, "F", f_per_q) + summary_rows(data, "stage1_selected", sel)
    pairs = paired_rows(data, "stage1_selected", sel, f_per_q)
    verdict = improves_on_f(pairs)
    write_csv(out / "summary.csv", rows)
    write_csv(out / "paired_vs_F.csv", pairs)
    write_csv(out / "selections.csv", selections)
    (out / "verdict.json").write_text(json.dumps({"V1_improves_on_F": verdict,
                                                  "grid": [[a, str(l)] for a, l in GRID1]}, indent=2))
    print_table(rows, pairs, verdict, "V1")
    return 0


def print_table(rows, pairs, verdict, name):
    for r in rows:
        print(f"{r['pipeline']:<22} k={r['k']}  5way {100 * r['5way_acc']:.1f}  10way {100 * r['10way_acc']:.1f}  "
              f"MRR {r['MRR']:.4f}")
    for r in pairs:
        print(f"{r['comparison']:<26} k={r['k']} {r['metric']:<7} {r['delta']:+.4f} "
              f"[{r['ci95_lo']:+.4f}; {r['ci95_hi']:+.4f}]")
    print(f"{name}: {'PASS' if verdict['PASS'] else 'FAIL'} {verdict}", flush=True)


# ======================================================================
# Stage 2: residual adapter z' = norm(z + W z), W zero-initialised
# ======================================================================

def stage2_inputs(data, prep, qidx):
    """Per (query, k) rows: query vector (C2, A0), own text table choice."""
    fr, ta, tb, ub = prep.c2
    xs, owner, kk = [], [], []
    for k in PREFIXES:
        xs.append(query_vec(fr[qidx], "A0", k))
        owner.append(qidx)
        kk.append(np.full(len(qidx), k))
    return np.concatenate(xs), np.concatenate(owner), np.concatenate(kk)


def train_adapter(data, prep, train_q, wd, epochs, seed, same_video_neg=True, torch_threads=4):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fr, ta, tb, ub = prep.c2
    tables = torch.from_numpy(np.stack([ta, tb]).astype(np.float32))  # (2, N, D)
    x, owner, _ = stage2_inputs(data, prep, train_q)
    x = torch.from_numpy(x.astype(np.float32))
    d = x.shape[1]
    train_set = set(train_q.tolist())
    # same-video negatives: one representative query per distinct other WHY text
    by_video = defaultdict(list)
    for i in train_q:
        by_video[data.vids[i]].append(int(i))
    neg_of = {}
    for i in train_q:
        seen, pool = {data.text_id[i]}, []
        for j in by_video[data.vids[i]]:
            if data.text_id[j] not in seen:
                seen.add(data.text_id[j])
                pool.append(j)
        neg_of[int(i)] = pool
    max_neg = max(len(v) for v in neg_of.values()) if same_video_neg else 0
    w = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=1e-3, weight_decay=wd)
    train_list = np.asarray(sorted(train_set))
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH2):
            b = order[start : start + BATCH2]
            own = owner[b]
            sel_table = torch.from_numpy(ub[own].astype(np.int64))
            pos = tables[sel_table, torch.from_numpy(own)]  # (b, D)
            cols = [pos[:, None, :]]
            masks = [torch.ones(len(b), 1, dtype=torch.bool)]
            if same_video_neg:
                negs = np.zeros((len(b), max_neg), dtype=np.int64)
                nm = np.zeros((len(b), max_neg), dtype=bool)
                for r, i in enumerate(own):
                    p = neg_of[int(i)]
                    negs[r, : len(p)] = p
                    nm[r, : len(p)] = True
                cols.append(tables[sel_table[:, None], torch.from_numpy(negs)])
                masks.append(torch.from_numpy(nm))
            n_rand = min(N_RANDOM_NEG, len(train_list))
            rnd = rng.choice(train_list, size=n_rand, replace=False)
            rv = tables[sel_table[:, None], torch.from_numpy(np.broadcast_to(rnd, (len(b), n_rand)).copy())]
            other = torch.from_numpy(data.vids[rnd][None, :] != data.vids[own][:, None])
            other &= torch.from_numpy(data.text_id[rnd][None, :] != data.text_id[own][:, None])
            cols.append(rv)
            masks.append(other)
            cand = torch.cat(cols, dim=1)
            mask = torch.cat(masks, dim=1)
            z = F.normalize(x[b] + x[b] @ w.T, dim=-1)
            logits = torch.einsum("bd,bcd->bc", z, cand) / TAU2
            logits = logits.masked_fill(~mask, -1e9)
            loss = F.cross_entropy(logits, torch.zeros(len(b), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return w.detach().numpy()


def adapt(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return L.normalize(q + q @ w.T)


def eval_adapter(data, prep, qidx, w, with_gallery=True):
    fr, ta, tb, ub = prep.c2
    res = {}
    for k in PREFIXES:
        q = adapt(query_vec(fr[qidx], "A0", k), w)
        s5 = cand_scores(q, ta, tb, ub, qidx, data.cands5[qidx])
        if not with_gallery:
            res[k] = {"kway5": L.strict_hits(s5, data.valid5[qidx]).astype(float)}
            continue
        s10 = cand_scores(q, ta, tb, ub, qidx, data.cands10[qidx])
        res[k] = metrics_for(data, qidx, s5, s10, gal_scores(q, ta, tb, ub, qidx))
    return res


def cmd_stage2(args) -> int:
    data = load_data()
    f_per_q = gate(data)
    out = out_dir("v3_stage2_hn_adapter", args.overwrite)
    all_rows, all_pairs, verdicts, selections = [], [], {}, []
    rows_f = summary_rows(data, "F", f_per_q)
    for variant, same_neg in (("hn_adapter", True), ("rand_neg_adapter", False)):
        parts = []
        for fold in data.folds:
            prep = FoldPrep(data, fold)
            inner_tr, inner_va = CV.inner_split(prep.train_idx, data.scenes, fold)
            assert not set(inner_va) & set(prep.idx) and not set(inner_tr) & set(prep.idx)
            t0 = time.time()
            scores = []
            for wd, ep in GRID2:
                w = train_adapter(data, prep, inner_tr, wd, ep, seed=0, same_video_neg=same_neg,
                                  torch_threads=args.threads)
                r = eval_adapter(data, prep, inner_va, w, with_gallery=False)
                scores.append(np.mean([r[k]["kway5"].mean() for k in PREFIXES]))
            best = L.select_argmax(np.asarray(scores))
            wd, ep = GRID2[best]
            seed_res = []
            for seed in SEEDS2:
                w = train_adapter(data, prep, prep.train_idx, wd, ep, seed, same_video_neg=same_neg,
                                  torch_threads=args.threads)
                seed_res.append(eval_adapter(data, prep, prep.idx, w))
            avg = {k: {m: np.mean([sr[k][m] for sr in seed_res], axis=0) for m in seed_res[0][k]}
                   for k in PREFIXES}
            parts.append((prep.idx, avg))
            selections.append({"variant": variant, "fold": fold, "weight_decay": wd, "epochs": ep,
                               "inner_val_mean_5way": float(scores[best]),
                               "inner_scores": json.dumps([round(float(s), 4) for s in scores])})
            print(f"[{variant}] fold {fold}: wd={wd} epochs={ep} ({time.time() - t0:.0f}s)", flush=True)
        per_q = pool_parts(data.n, parts)
        all_rows += summary_rows(data, variant, per_q)
        pairs = paired_rows(data, variant, per_q, f_per_q)
        all_pairs += pairs
        verdicts[variant] = improves_on_f(pairs)
        print_table(all_rows[-4:], pairs, verdicts[variant], f"V2[{variant}]")
    write_csv(out / "summary.csv", rows_f + all_rows)
    write_csv(out / "paired_vs_F.csv", all_pairs)
    write_csv(out / "selections.csv", selections)
    (out / "verdict.json").write_text(json.dumps({
        "V2_improves_on_F": verdicts["hn_adapter"], "control_rand_neg": verdicts["rand_neg_adapter"],
        "grid": GRID2, "seeds": SEEDS2, "tau": TAU2, "inner_selection": "CV.inner_split (held-out training scenes)",
    }, indent=2))
    return 0


# ======================================================================
# NEXT: predict the observed next step
# ======================================================================

def load_next_labels(rows):
    """sample_id -> (observed_next_step, local_intent) from step_label.json."""
    cache, out = {}, {}
    for r in rows:
        label = Path(r["video_path"]).parent / "step_label.json"
        if label not in cache:
            payload = json.loads(label.read_text(encoding="utf-8"))
            cache[label] = {int(s["step_id"]): s for s in payload["steps"]}
        step = cache[label].get(int(r["step_id"]))
        if step is None:
            raise KeyError(f"{r['sample_id']}: step not in {label}")
        if str(step["local_intent"]) != r["what"]:
            raise ValueError(f"{r['sample_id']}: WHAT mismatch with {label}")
        out[r["sample_id"]] = str(step["observed_next_step"])
    return out


def cmd_next(args) -> int:
    import run_anticipation_horizon as AH

    data = load_data()
    f_per_q = gate(data)
    out = out_dir("v3_next_step", args.overwrite)
    nxt = load_next_labels(data.rows)
    texts = [nxt[s] for s in data.sample_ids]
    whats = [r["what"] for r in data.rows]
    vocab = sorted(set(texts))
    emb = AH.encode_texts(ENCODER_ID, vocab, device=args.device)  # T1 template
    tix = {t: i for i, t in enumerate(vocab)}
    text_raw = emb[[tix[t] for t in texts]]
    text_id = np.asarray([tix[t] for t in texts])
    c5 = CV.build_kway(data.vids.tolist(), texts, data.sample_ids, 5)
    c10 = CV.build_kway(data.vids.tolist(), texts, data.sample_ids, 10)
    v5, v10 = (c5 >= 0).all(axis=1), (c10 >= 0).all(axis=1)
    chance = CV.gallery_chance(text_id, data.fold_of)["MRR"]
    # the step that follows each query in the same video (for the upper bound)
    nxt_row = np.full(data.n, -1)
    for v in np.unique(data.vids):
        members = sorted(np.flatnonzero(data.vids == v), key=lambda i: data.steps[i])
        for a, b in zip(members, members[1:]):
            nxt_row[a] = b
    # diagnostics
    same_as_next_what = np.mean([texts[i] == whats[nxt_row[i]] for i in range(data.n) if nxt_row[i] >= 0])
    frames = data.frames[ENC]
    ctx, has = data.ctx[ENC]
    results = {}
    for cent in ("C2", "C0"):
        parts = []
        for fold in data.folds:
            idx, train_idx = data.fold_idx(fold)
            fr, ta, tb, ub = L.processed(frames, text_raw, cent, train_idx, ctx, has)
            res = {}
            for k in PREFIXES:
                q = query_vec(fr[idx], "A0", k)
                s5 = cand_scores(q, ta, tb, ub, idx, c5[idx])
                s10 = cand_scores(q, ta, tb, ub, idx, c10[idx])
                gs = gal_scores(q, ta, tb, ub, idx)
                ids = text_id[idx]
                ranks = P.first_positive_rank(gs, ids[:, None] == ids[None, :])
                res[k] = {"rr": 1.0 / ranks, "kway5": L.strict_hits(s5, v5[idx]).astype(float),
                          "kway10": L.strict_hits(s10, v10[idx]).astype(float)}
            if cent == "C2":
                # upper bound: frames of the step that actually follows, same candidates
                has_next = nxt_row[idx] >= 0
                qi = np.where(has_next, nxt_row[idx], idx)
                q = query_vec(fr[qi], "A0", 8)
                s5 = cand_scores(q, ta, tb, ub, idx, c5[idx])
                res["upper8"] = {"kway5": np.where(has_next, L.strict_hits(s5, v5[idx]), np.nan)}
            parts.append((idx, res))
        results[cent] = pool_parts(data.n, parts)
    rows = []
    for cent, per_q in results.items():
        for k in PREFIXES:
            m = per_q[k]
            row = {"label": "NEXT", "centering": cent, "k": k, "n_queries": data.n,
                   "n_valid5": int(v5.sum()), "n_valid10": int(v10.sum()),
                   "5way_acc": float(m["kway5"][v5].mean()) if v5.any() else np.nan,
                   "10way_acc": float(m["kway10"][v10].mean()) if v10.any() else np.nan,
                   "MRR": float(m["rr"].mean())}
            for name, num, den, ok in (("5way", m["kway5"], np.full(data.n, 0.2), v5),
                                       ("10way", m["kway10"], np.full(data.n, 0.1), v10),
                                       ("MRR", m["rr"], chance, np.ones(data.n, bool))):
                if not ok.any():
                    continue
                p, lo, hi = CV.video_bootstrap(data.vids[ok], num[ok], den[ok], B=B)
                row[f"{name}_lift"], row[f"{name}_lift_lo"], row[f"{name}_lift_hi"] = p, lo, hi
            rows.append(row)
    up = results["C2"]["upper8"]["kway5"]
    okup = ~np.isnan(up) & v5
    p, lo, hi = CV.video_bootstrap(data.vids[okup], up[okup], np.full(int(okup.sum()), 0.2), B=B)
    rows.append({"label": "NEXT_upper_bound_sees_next_step", "centering": "C2", "k": 8,
                 "n_queries": int(okup.sum()), "5way_acc": float(up[okup].mean()),
                 "5way_lift": p, "5way_lift_lo": lo, "5way_lift_hi": hi})
    for k in PREFIXES:
        rows.append({"label": "WHY_F_reference", "centering": "C2", "k": k, "n_queries": data.n,
                     "5way_acc": float(f_per_q[k]["kway5"].mean())})
    write_csv(out / "summary.csv", rows)
    (out / "diagnostics.json").write_text(json.dumps({
        "n_distinct_next": len(vocab), "n_queries": data.n, "n_valid5": int(v5.sum()),
        "frac_next_equals_following_step_WHAT": float(same_as_next_what),
        "n_last_steps": int((nxt_row < 0).sum()), "template": "T1", "encoder": ENCODER_ID,
    }, indent=2))
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("label", "centering", "k", "n_queries", "5way_acc", "5way_lift_lo", "5way_lift_hi",
                        "10way_acc", "MRR")}, flush=True)
    print(f"NEXT(i) == WHAT(i+1) for {100 * same_as_next_what:.1f}% of steps", flush=True)
    return 0


# ======================================================================
# selftest (synthetic)
# ======================================================================

def cmd_selftest(args) -> int:
    for k in PREFIXES:
        for a in AGGS1:
            w = agg_weights(a, k)
            assert abs(w.sum() - 1) < 1e-12 and len(w) == k
        assert agg_weights("A3", k)[-1] == 1.0
        assert np.all(np.diff(agg_weights("A4h2", k)) > 0)
    rng = np.random.default_rng(0)
    s0, s2 = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    assert np.allclose(fuse(s0, s2, np.inf), s2) and np.allclose(fuse(s0, s2, 0.0), s0)
    q = L.normalize(rng.normal(size=(4, 16)))
    assert np.allclose(adapt(q, np.zeros((16, 16))), q), "adapter must be identity at init"
    assert GRID1[0] == ("A0", np.inf) and len(GRID1) == 30
    assert len(GRID2) == 6
    print("selftest passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["selftest", "gate", "stage1", "stage2", "next"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads for stage2")
    p.add_argument("--device", default="cpu", help="device for the NEXT text encoder")
    args = p.parse_args()
    if args.cmd != "selftest" and not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}")
    if args.cmd == "gate":
        gate(load_data())
        return 0
    return {"selftest": cmd_selftest, "stage1": cmd_stage1, "stage2": cmd_stage2, "next": cmd_next}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
