#!/usr/bin/env python3
"""Scene-grouped cross-validation for early goal ranking (plan steps 0 + 1).

Why this script exists
  The fixed 4-video validation split + best-epoch-on-val selection made every
  number noisy and optimistic, and the locked candidate test came out at chance.
  Here every train+val sample is evaluated exactly once, out-of-fold, on
  scenes the model never saw, with the epoch chosen on an INNER split of the
  training scenes only.  The locked test split is never read.

Protocol
  * data: frame-level cache (8 x 960 frozen SmolVLM features) of the full
    dataset, splits "train" and "val" ONLY (the loader refuses "test").
  * folds: whole scenes are assigned to folds (greedy size balancing), so a
    held-out video never shares a scene with training data.
  * model/loss/hyper-parameters: same MeanPoolBaseline + multi-positive InfoNCE
    + cosine loss as the earlier pilot runs; trained on frames[:, :k] and
    evaluated on frames[:, :k] (train@k / eval@k), k in {2,4,6,8}.
  * epoch selection: a fixed inner split of the training scenes picks E*
    (smoothed best inner-val MRR); the model is then refit on all training
    scenes for E* epochs.  The held-out fold is never used for selection.
  * targets compared:
      sentence    frozen sentence encoder over the WHY text (no fitting)
      smolvlm_pca the earlier SmolVLM-feature + pre-fit PCA target (optional;
                  needs --stage-a-cache and --why-pca; PCA was fit on the pilot
                  train videos, not fold-wise: flagged as a mild caveat)
  * metrics (all out-of-fold):
      gallery retrieval inside the held-out fold (exact-WHY-text positives) ->
        MRR / Top-1 / Top-3, plus exact random-ranking chance and lift
      K-way candidate choice (K = 5, 10): GT + (K-1) distractors drawn from the
        SAME video (different WHY text), fixed per query -> accuracy, chance 1/K
  * uncertainty: bootstrap over source videos (resample whole videos).

`--selftest` runs the numpy-only checks and needs no data / torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_prefix_length_ablation as P  # numpy-only helpers (ranks, chance, json/csv io)

ALLOWED_SPLITS = ("train", "val")  # the locked test split is never loaded
KWAYS = (5, 10)


# ======================================================================
# numpy-only logic (unit-tested by --selftest)
# ======================================================================

def assign_folds(scenes, n_folds):
    """Whole scenes -> folds. n_folds<=0 means leave-one-scene-out.
    Greedy balancing by sample count; deterministic."""
    counts = defaultdict(int)
    for s in scenes:
        counts[s] += 1
    names = sorted(counts, key=lambda s: (-counts[s], s))
    if n_folds <= 0 or n_folds >= len(names):
        return {s: i for i, s in enumerate(names)}
    load = [0] * n_folds
    out = {}
    for s in names:
        f = min(range(n_folds), key=lambda i: (load[i], i))
        out[s] = f
        load[f] += counts[s]
    return out


def inner_split(train_idx, scenes, fold, frac=0.15):
    """Hold out whole training scenes (>= frac of training samples) as the
    inner validation set. Deterministic in (scene, fold); same for every
    seed / prefix / target so comparisons stay paired."""
    train_idx = np.asarray(train_idx)
    sc = np.asarray(scenes)[train_idx]
    names = sorted(set(sc.tolist()), key=lambda s: zlib.crc32(f"{s}|fold{fold}".encode()))
    target = max(1, int(round(frac * len(train_idx))))
    chosen, total = [], 0
    for s in names[:-1]:  # always leave at least one scene for inner-train
        if total >= target:
            break
        chosen.append(s)
        total += int((sc == s).sum())
    mask = np.isin(sc, chosen)
    return train_idx[~mask], train_idx[mask]


def build_kway(video_uids, texts, sample_ids, K):
    """[N,K] global indices: col 0 = the query itself, cols 1.. = distractors
    from the same video with a different WHY text (distinct among themselves).
    Rows with too few distinct texts are all -1 (excluded)."""
    N = len(texts)
    by_video = defaultdict(list)
    for i, v in enumerate(video_uids):
        by_video[v].append(i)
    out = np.full((N, K), -1, dtype=np.int64)
    for i in range(N):
        seen, pool = {texts[i]}, []
        for j in by_video[video_uids[i]]:
            if texts[j] not in seen:
                seen.add(texts[j])
                pool.append(j)
        if len(pool) < K - 1:
            continue
        rng = np.random.RandomState(zlib.crc32(f"{sample_ids[i]}|{K}".encode()) & 0xFFFFFFFF)
        pick = rng.choice(len(pool), K - 1, replace=False)
        out[i] = [i] + [pool[p] for p in pick]
    return out


def eval_fold(Z, target, idx_f, text_id, cands):
    """Z: predictions for held-out samples idx_f; target: [N,D] all targets."""
    G = target[idx_f]
    sim = Z @ G.T
    ids = text_id[idx_f]
    ranks = P.first_positive_rank(sim, ids[:, None] == ids[None, :])
    out = {"ranks": ranks}
    for K, c in cands.items():
        cc = c[idx_f]
        valid = (cc >= 0).all(axis=1)
        Tc = target[np.where(cc >= 0, cc, 0)]
        sc = np.einsum("nd,nkd->nk", Z, Tc)
        out[f"kway{K}"] = np.where(valid, (sc[:, :1] > sc[:, 1:]).all(axis=1), False)
        out[f"kway{K}_valid"] = valid
    return out


def gallery_chance(text_id, fold_of_sample):
    """Exact per-example random-ranking chance (MRR, Top1, Top3) given the
    held-out-fold gallery each sample is ranked in."""
    N = len(text_id)
    mrr, t1, t3 = np.zeros(N), np.zeros(N), np.zeros(N)
    cache = {}
    for f in np.unique(fold_of_sample):
        idx = np.where(fold_of_sample == f)[0]
        n = len(idx)
        cnt = defaultdict(int)
        for t in text_id[idx]:
            cnt[int(t)] += 1
        for i in idx:
            m = cnt[int(text_id[i])]
            if (n, m) not in cache:
                cache[(n, m)] = P._chance_one(n, m)
            mrr[i], t1[i], t3[i] = cache[(n, m)]
    return {"MRR": mrr, "Top1": t1, "Top3": t3}


def video_bootstrap(clusters, num, den, B=10000, seed=0):
    """Ratio estimate sum(num)/sum(den) with a bootstrap over whole clusters
    (source videos). Returns (point, lo95, hi95)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(uniq))
    sd = np.bincount(inv, weights=den, minlength=len(uniq))
    point = float(sn.sum() / sd.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paper_numbers(R, valid=None):
    """R: [S,N] ranks -> seed-averaged per-example RR / Top1 / Top3 hits."""
    R = np.asarray(R, dtype=np.float64)
    return {"RR": (1.0 / R).mean(0), "Top1": (R <= 1).mean(0), "Top3": (R <= 3).mean(0)}


def selftest():
    rng = np.random.default_rng(0)
    # folds: scene-pure, complete, roughly balanced, deterministic
    scenes = np.array(sum([[f"s{i}"] * c for i, c in enumerate([50, 40, 30, 30, 20, 20, 10, 10])], []))
    fm = assign_folds(scenes.tolist(), 3)
    assert set(fm) == set(scenes.tolist()) and set(fm.values()) == {0, 1, 2}
    loads = [sum((scenes == s).sum() for s in fm if fm[s] == f) for f in range(3)]
    assert max(loads) - min(loads) <= 30, loads
    assert assign_folds(scenes.tolist(), 3) == fm
    assert len(set(assign_folds(scenes.tolist(), 0).values())) == 8  # leave-one-scene-out

    # inner split: disjoint, scene-pure, non-empty both sides
    tr = np.where(np.array([fm[s] for s in scenes]) != 0)[0]
    a, b = inner_split(tr, scenes, 0)
    assert len(a) and len(b) and not set(a) & set(b) and set(a) | set(b) == set(tr)
    assert not (set(scenes[a].tolist()) & set(scenes[b].tolist()))

    # K-way: GT first, same video, distinct different texts
    vids = np.array([f"v{i // 30}" for i in range(120)])
    texts = [f"t{rng.integers(0, 40)}_{i // 30}" for i in range(120)]
    sids = [f"id{i}" for i in range(120)]
    c = build_kway(vids.tolist(), texts, sids, 5)
    for i in range(120):
        row = c[i]
        assert (row >= 0).all() and row[0] == i
        assert len({vids[j] for j in row}) == 1
        assert len({texts[j] for j in row}) == 5
    assert (build_kway(vids.tolist(), texts, sids, 5) == c).all()  # deterministic

    # eval_fold with perfect predictions -> rank 1, K-way all correct
    text_id = np.array([hash(t) % 10**6 for t in texts])
    D = 16
    uniq = {t: rng.normal(size=D) for t in set(texts)}
    target = np.stack([uniq[t] for t in texts])
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    idx = np.arange(0, 60)
    ev = eval_fold(target[idx], target, idx, text_id, {5: c})
    assert (ev["ranks"] == 1).all() and ev["kway5"].all() and ev["kway5_valid"].all()

    # chance: unique texts, one fold of n -> H_n / n
    ch = gallery_chance(np.arange(50), np.zeros(50, dtype=int))
    assert abs(ch["MRR"][0] - sum(1 / r for r in range(1, 51)) / 50) < 1e-9

    # bootstrap: constant ratio 2 -> CI collapses on 2
    clusters = np.repeat(np.arange(10), 7)
    pt, lo, hi = video_bootstrap(clusters, 2 * np.ones(70), np.ones(70), B=500)
    assert pt == lo == hi == 2.0
    pt, lo, hi = video_bootstrap(clusters, rng.normal(1.0, 1.0, 70), np.ones(70), B=2000)
    assert lo < pt < hi
    print("selftest: PASS (folds, inner split, K-way, eval_fold, chance, video bootstrap)")


# ======================================================================
# data / targets (torch lazily imported)
# ======================================================================

def load_records(temporal_root: Path):
    import torch
    recs = []
    for split in ALLOWED_SPLITS:
        root = temporal_root / split
        if not root.is_dir():
            raise FileNotFoundError(root)
        for p in sorted(root.glob("*.pt")):
            x = torch.load(p, map_location="cpu", weights_only=False)
            assert str(x.get("split", split)) in ALLOWED_SPLITS, f"{p}: unexpected split"
            recs.append({
                "sample_id": str(x["sample_id"]), "split": split,
                "video_uid": str(x["video_uid"]), "scene": str(x["scene"]),
                "event": str(x.get("event", "")), "step_id": str(x.get("step_id", "")),
                "why": str(x["why"]), "frames": x["frame_features"].float(),
            })
    return recs


def embed_texts(texts, model_name, device, cache_path: Path):
    import torch
    import torch.nn.functional as F
    uniq = sorted(set(texts))
    if cache_path.exists():
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        if d.get("model") == model_name and d.get("texts") == uniq:
            table = {t: d["emb"][i] for i, t in enumerate(uniq)}
            return torch.stack([table[t] for t in texts]).numpy()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    vecs = []
    for i in range(0, len(uniq), 64):
        enc = tok(uniq[i:i + 64], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs.append(F.normalize((h * m).sum(1) / m.sum(1).clamp_min(1e-9), dim=-1).cpu())
    emb = torch.cat(vecs)
    torch.save({"model": model_name, "texts": uniq, "emb": emb}, cache_path)
    table = {t: emb[i] for i, t in enumerate(uniq)}
    return torch.stack([table[t] for t in texts]).numpy()


def smolvlm_pca_targets(T, torch, recs, stage_a_root, why_pca):
    tf = T.load_pca_transform(why_pca)
    meta = {}
    for split in ALLOWED_SPLITS:
        meta.update(T.load_stage_a_metadata(stage_a_root, split))
    feats, mism = [], 0
    for r in recs:
        m = meta[r["sample_id"]]
        mism += int(m["why"] != r["why"])
        feats.append(m["why_feature"])
    z = torch.nn.functional.normalize(tf(torch.stack(feats)), dim=-1)
    return z.cpu().numpy(), mism


# ======================================================================
# training (torch)
# ======================================================================

def fit(T, torch, args, dev, frames, target_t, text_id, train_idx, k, seed, epochs, val_idx=None):
    T.set_seed(seed)
    model = T.MeanPoolBaseline(hidden_dim=args.hidden_dim, output_dim=int(target_t.shape[1]),
                               dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr = torch.as_tensor(np.asarray(train_idx), device=dev)
    x_tr, y_tr = frames[tr][:, :k], target_t[tr]          # prefix = first k frames only
    ids = text_id[np.asarray(train_idx)]
    pos_tr = torch.as_tensor(ids[:, None] == ids[None, :], device=dev)
    if val_idx is not None:
        vi = torch.as_tensor(np.asarray(val_idx), device=dev)
        x_va, y_va = frames[vi][:, :k], target_t[vi]
        vids = text_id[np.asarray(val_idx)]
        pos_va = vids[:, None] == vids[None, :]
    hist = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x_tr)
        loss = (T.multi_positive_infonce(pred, y_tr, pos_tr, args.temperature)
                + args.cosine_weight * T.cosine_target_loss(pred, y_tr))
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if val_idx is not None:
            model.eval()
            with torch.no_grad():
                sim = (model(x_va) @ y_va.T).cpu().numpy()
            hist.append(float((1.0 / P.first_positive_rank(sim, pos_va)).mean()))
    return model, hist


def pick_epochs(hist, window=5):
    h = np.asarray(hist, dtype=np.float64)
    w = max(1, min(window, len(h)))
    sm = np.convolve(h, np.ones(w) / w, mode="valid")
    return int(np.clip(int(sm.argmax()) + 1 + w // 2, 1, len(h)))


# ======================================================================
# main run
# ======================================================================

def run(args):
    import torch
    import train_temporal_progression_intention as T

    out = args.output_dir
    (out / "runs").mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{out}/summary.json exists: use a new --output-dir or --overwrite "
                 "(cached runs/ are reused unless --rerun).")
    dev = torch.device(args.device)

    # cached runs are only reusable if everything that changes a result is unchanged
    sig = {k: str(getattr(args, k)) for k in ("n_folds", "epochs", "lr", "weight_decay", "temperature",
                                              "cosine_weight", "hidden_dim", "dropout", "grad_clip",
                                              "sentence_model", "stage_a_cache", "why_pca", "temporal_cache")}
    sigp = out / "runs" / "_signature.json"
    if sigp.exists() and json.loads(sigp.read_text()) != sig and not args.rerun:
        sys.exit("runs/ was produced with different settings; use a new --output-dir or --rerun.")
    P.write_json(sigp, sig)

    recs = load_records(args.temporal_cache)
    N = len(recs)
    sids = [r["sample_id"] for r in recs]
    assert len(set(sids)) == N, "duplicate sample_id"
    assert all(r["split"] in ALLOWED_SPLITS for r in recs)
    frames = torch.stack([r["frames"] for r in recs]).to(dev)
    assert frames.shape[1] >= max(args.prefixes) and frames.shape[2] == 960
    assert torch.isfinite(frames).all()
    texts = [r["why"] for r in recs]
    vids = np.array([r["video_uid"] for r in recs])
    scenes = np.array([r["scene"] for r in recs])
    text_vocab = {t: i for i, t in enumerate(sorted(set(texts)))}
    text_id = np.array([text_vocab[t] for t in texts])

    fold_of_scene = assign_folds(scenes.tolist(), args.n_folds)
    n_folds = len(set(fold_of_scene.values()))
    fold_of = np.array([fold_of_scene[s] for s in scenes])
    # a video belongs to exactly one scene -> one fold, so same-video candidates never leave the fold
    assert all(len(set(fold_of[vids == v])) == 1 for v in np.unique(vids))
    print("=" * 78)
    print(f"samples={N} videos={len(np.unique(vids))} scenes={len(set(scenes))} folds={n_folds} "
          f"| splits used={ALLOWED_SPLITS} (test never loaded)")

    # targets
    targets = {}
    emb = embed_texts(texts, args.sentence_model, dev, out / "text_embeddings.pt")
    targets["sentence"] = emb.astype(np.float32)
    para = {}
    sim_same = []
    for v in np.unique(vids):
        ix = np.where(vids == v)[0]
        S = emb[ix] @ emb[ix].T
        diff = np.array([[texts[a] != texts[b] for b in ix] for a in ix])
        S = np.where(diff, S, -1.0)
        sim_same.append(S.max(axis=1))
    sim_same = np.concatenate(sim_same)
    for thr in (0.8, 0.9):
        para[f"frac_queries_with_other_WHY_in_same_video_cos>={thr}"] = float((sim_same >= thr).mean())
    if args.stage_a_cache and args.why_pca:
        z, mism = smolvlm_pca_targets(T, torch, recs, args.stage_a_cache, args.why_pca)
        targets["smolvlm_pca"] = z.astype(np.float32)
        print(f"smolvlm_pca baseline target loaded (WHY text mismatches vs cache: {mism})")
    print("paraphrase diagnostic:", para)

    cands = {K: build_kway(vids.tolist(), texts, sids, K) for K in KWAYS}
    chance = gallery_chance(text_id, fold_of)
    chance_k = {K: float(1.0 / K) for K in KWAYS}
    print("exact gallery chance (mean over examples): "
          + " ".join(f"{k}={v.mean():.4f}" for k, v in chance.items())
          + " | K-way chance " + str(chance_k)
          + " | K-way valid " + str({K: int((cands[K][:, 0] >= 0).sum()) for K in KWAYS}))
    print("=" * 78)

    target_t = {n: torch.as_tensor(a, device=dev) for n, a in targets.items()}
    seeds, prefixes = args.seeds, sorted(set(args.prefixes))
    res = {}  # (target,k) -> dict arrays
    for tname in targets:
        for k in prefixes:
            res[(tname, k)] = {"ranks": np.zeros((len(seeds), N), dtype=np.int64),
                               "kway5": np.zeros((len(seeds), N), bool), "kway10": np.zeros((len(seeds), N), bool),
                               "estar": np.zeros((len(seeds), n_folds), dtype=int)}
    t0 = time.time()
    for f in range(n_folds):
        idx_f = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        in_tr, in_va = inner_split(train_idx, scenes, f)
        assert not (set(vids[idx_f]) & set(vids[train_idx])) and not (set(scenes[idx_f]) & set(scenes[train_idx]))
        for tname in targets:
            for k in prefixes:
                for si, seed in enumerate(seeds):
                    p = out / "runs" / f"fold{f}__{tname}__k{k}__seed{seed}.json"
                    if p.exists() and not args.rerun:
                        d = json.loads(p.read_text())
                    else:
                        _, hist = fit(T, torch, args, dev, frames, target_t[tname], text_id,
                                      in_tr, k, seed, args.epochs, val_idx=in_va)
                        estar = pick_epochs(hist)
                        model, _ = fit(T, torch, args, dev, frames, target_t[tname], text_id,
                                       train_idx, k, seed, estar)
                        model.eval()
                        with torch.no_grad():
                            fi = torch.as_tensor(idx_f, device=dev)
                            Z = model(frames[fi][:, :k]).cpu().numpy()
                            # future frames must not matter for a k-frame prefix
                            noisy = frames[fi].clone()
                            noisy[:, k:] = torch.randn_like(noisy[:, k:]) * 1e3
                            assert torch.allclose(model(frames[fi][:, :k]), model(noisy[:, :k]))
                        ev = eval_fold(Z, targets[tname], idx_f, text_id, cands)
                        d = {"idx": idx_f.tolist(), "estar": estar,
                             "ranks": ev["ranks"].tolist(),
                             "kway5": ev["kway5"].tolist(), "kway10": ev["kway10"].tolist()}
                        P.write_json(p, d)
                    r = res[(tname, k)]
                    r["ranks"][si, idx_f] = d["ranks"]
                    r["kway5"][si, idx_f] = d["kway5"]
                    r["kway10"][si, idx_f] = d["kway10"]
                    r["estar"][si, f] = d["estar"]
        print(f"fold {f + 1}/{n_folds} done | held-out videos={len(np.unique(vids[idx_f]))} "
              f"samples={len(idx_f)} | {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------ aggregate
    valid = {K: (cands[K][:, 0] >= 0) for K in KWAYS}
    summary_rows, per_example = [], {}
    boot_cache = {}
    for (tname, k), r in sorted(res.items()):
        nums = paper_numbers(r["ranks"])
        row = {"target": tname, "prefix_frames": k, "n_examples": N, "n_videos": int(len(np.unique(vids))),
               "n_seeds": len(seeds), "mean_E_star": float(r["estar"].mean())}
        for m, key in (("MRR", "RR"), ("Top1", "Top1"), ("Top3", "Top3")):
            per_seed = [float(np.mean({"RR": 1.0 / r["ranks"][s], "Top1": (r["ranks"][s] <= 1) * 1.0,
                                       "Top3": (r["ranks"][s] <= 3) * 1.0}[key])) for s in range(len(seeds))]
            mu, sd = P.mean_std(per_seed)
            pt, lo, hi = video_bootstrap(vids, nums[key], chance[m])
            row.update({f"{m}_mean": mu, f"{m}_sd_seeds": sd, f"{m}_chance": float(chance[m].mean()),
                        f"{m}_lift": pt, f"{m}_lift_ci_lo": lo, f"{m}_lift_ci_hi": hi})
        for K in KWAYS:
            acc_ex = r[f"kway{K}"].mean(0)
            per_seed = [float(r[f"kway{K}"][s][valid[K]].mean()) for s in range(len(seeds))]
            mu, sd = P.mean_std(per_seed)
            pt, lo, hi = video_bootstrap(vids[valid[K]], acc_ex[valid[K]], np.ones(valid[K].sum()))
            row.update({f"kway{K}_acc_mean": mu, f"kway{K}_acc_sd_seeds": sd, f"kway{K}_chance": chance_k[K],
                        f"kway{K}_lift": pt / chance_k[K], f"kway{K}_lift_ci_lo": lo / chance_k[K],
                        f"kway{K}_lift_ci_hi": hi / chance_k[K], f"kway{K}_n_valid": int(valid[K].sum())})
        summary_rows.append(row)
        per_example[(tname, k)] = {"RR": nums["RR"], "kway5": r["kway5"].mean(0), "kway10": r["kway10"].mean(0)}

    # paired differences (same examples), bootstrap over videos
    paired = []

    def add_pair(label, A, B, ka, kb, ta, tb):
        for metric, getter, mask in (("MRR", lambda d: d["RR"], np.ones(N, bool)),
                                     ("kway5", lambda d: d["kway5"], valid[5]),
                                     ("kway10", lambda d: d["kway10"], valid[10])):
            diff = getter(per_example[(tb, kb)]) - getter(per_example[(ta, ka)])
            pt, lo, hi = video_bootstrap(vids[mask], diff[mask], np.ones(mask.sum()))
            paired.append({"comparison": label, "metric": metric, "from": f"{ta}@k{ka}", "to": f"{tb}@k{kb}",
                           "delta": pt, "ci95_lo": lo, "ci95_hi": hi, "n_videos": int(len(np.unique(vids[mask])))})

    for tname in targets:
        ks = prefixes
        pairs = list(zip(ks[:-1], ks[1:])) + ([(ks[0], ks[-1])] if len(ks) > 2 else [])
        for a, b in pairs:
            add_pair("prefix", None, None, a, b, tname, tname)
    if "smolvlm_pca" in targets:
        for k in prefixes:
            add_pair("target (sentence minus smolvlm_pca)", None, None, k, k, "smolvlm_pca", "sentence")

    # per-video breakdown
    pv_rows = []
    for (tname, k), r in sorted(res.items()):
        rr = per_example[(tname, k)]["RR"]
        for v in np.unique(vids):
            m = vids == v
            pv_rows.append({"target": tname, "prefix_frames": k, "video_uid": v, "scene": scenes[m][0],
                            "event": recs[int(np.where(m)[0][0])]["event"], "n": int(m.sum()),
                            "MRR": float(rr[m].mean()), "MRR_chance": float(chance["MRR"][m].mean()),
                            "MRR_lift": float(rr[m].mean() / chance["MRR"][m].mean()),
                            "kway5_acc": (float(per_example[(tname, k)]["kway5"][m & valid[5]].mean())
                                          if (m & valid[5]).any() else None)})

    # ------------------------------------------------------------------ write
    f_sum = ["target", "prefix_frames", "n_examples", "n_videos", "n_seeds", "mean_E_star"]
    for m in ("MRR", "Top1", "Top3"):
        f_sum += [f"{m}_mean", f"{m}_sd_seeds", f"{m}_chance", f"{m}_lift", f"{m}_lift_ci_lo", f"{m}_lift_ci_hi"]
    for K in KWAYS:
        f_sum += [f"kway{K}_acc_mean", f"kway{K}_acc_sd_seeds", f"kway{K}_chance", f"kway{K}_lift",
                  f"kway{K}_lift_ci_lo", f"kway{K}_lift_ci_hi", f"kway{K}_n_valid"]
    P.write_csv(out / "summary.csv", summary_rows, f_sum)
    P.write_csv(out / "paired_differences.csv", paired,
                ["comparison", "metric", "from", "to", "delta", "ci95_lo", "ci95_hi", "n_videos"])
    P.write_csv(out / "per_video_metrics.csv", pv_rows,
                ["target", "prefix_frames", "video_uid", "scene", "event", "n", "MRR", "MRR_chance",
                 "MRR_lift", "kway5_acc"])
    ex_fields = ["sample_id", "video_uid", "scene", "fold"]
    ex_rows = [{"sample_id": sids[i], "video_uid": vids[i], "scene": scenes[i], "fold": int(fold_of[i])}
               for i in range(N)]
    for (tname, k), d in sorted(per_example.items()):
        for key in ("RR", "kway5"):
            col = f"{key}|{tname}|k{k}"
            ex_fields.append(col)
            for i in range(N):
                ex_rows[i][col] = round(float(d[key][i]), 6)
    P.write_csv(out / "per_example.csv", ex_rows, ex_fields)
    cfg = {"script": "scripts/run_scene_cv.py", "git_commit": P.git_commit(),
           "args": {k: str(v) for k, v in vars(args).items()}, "torch": torch.__version__,
           "n_samples": N, "videos": sorted(np.unique(vids).tolist()), "scenes": sorted(set(scenes.tolist())),
           "fold_of_scene": fold_of_scene, "splits_used": list(ALLOWED_SPLITS),
           "gallery_chance_mean": {k: float(v.mean()) for k, v in chance.items()}, "kway_chance": chance_k,
           "paraphrase_diagnostic": para, "targets": list(targets)}
    P.write_json(out / "config.json", cfg)
    P.write_json(out / "summary.json", {"summary": summary_rows, "paired_differences": paired})
    write_markdown(out / "summary.md", summary_rows, paired, args, len(np.unique(vids)), n_folds, para)
    plot(out, summary_rows, prefixes)
    print(f"\nDONE -> {out}")


def write_markdown(path, rows, paired, args, n_videos, n_folds, para):
    L = ["# Scene-grouped CV: early goal ranking (out-of-fold)", ""]
    L.append(f"{n_folds} scene-grouped folds over {n_videos} videos (train+val only; locked test never read). "
             f"Seeds {args.seeds}; epoch chosen on an inner split of training scenes. "
             "Lift = metric / exact random-ranking chance (1.0 = chance). "
             "95% intervals come from a bootstrap over source videos (resampling whole videos); "
             "they are descriptive and do not cover scene/domain shift beyond these videos.")
    L.append("")
    for tname in sorted({r["target"] for r in rows}):
        L += [f"## target = {tname}", "",
              "| k | MRR | MRR lift [95% CI] | Top-1 lift | Top-3 lift | 5-way acc (chance 0.20) | 5-way lift [CI] | 10-way acc (chance 0.10) |",
              "|---:|---|---|---|---|---|---|---|"]
        for r in sorted([x for x in rows if x["target"] == tname], key=lambda x: x["prefix_frames"]):
            L.append(
                f"| {r['prefix_frames']} | {P.fmt(r['MRR_mean'], r['MRR_sd_seeds'])} | "
                f"{r['MRR_lift']:.2f} [{r['MRR_lift_ci_lo']:.2f}, {r['MRR_lift_ci_hi']:.2f}] | "
                f"{r['Top1_lift']:.2f} | {r['Top3_lift']:.2f} | "
                f"{P.fmt(r['kway5_acc_mean'], r['kway5_acc_sd_seeds'])} | "
                f"{r['kway5_lift']:.2f} [{r['kway5_lift_ci_lo']:.2f}, {r['kway5_lift_ci_hi']:.2f}] | "
                f"{P.fmt(r['kway10_acc_mean'], r['kway10_acc_sd_seeds'])} |")
        L.append("")
    L += ["## Paired differences (to minus from; same examples; bootstrap over videos)", "",
          "| comparison | metric | from -> to | delta [95% CI] |", "|---|---|---|---|"]
    for p in paired:
        L.append(f"| {p['comparison']} | {p['metric']} | {p['from']} -> {p['to']} | "
                 f"{p['delta']:+.4f} [{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] |")
    L += ["", "## Notes", "",
          f"- Paraphrase diagnostic (sentence encoder): {para}. Exact-text positives treat paraphrases as wrong, "
          "so absolute scores are capped.",
          "- `smolvlm_pca` (if present) uses a PCA fit on the pilot training videos, not fold-wise.",
          "- K-way distractors are drawn once per query from the same video (different WHY text), independent of "
          "model, prefix and seed."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(out, rows, prefixes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plot skipped:", repr(e))
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "MRR_lift", "MRR lift over chance"), (axes[1], "kway5_lift", "5-way accuracy lift")):
        for tname in sorted({r["target"] for r in rows}):
            rr = sorted([x for x in rows if x["target"] == tname], key=lambda x: x["prefix_frames"])
            y = np.array([x[key] for x in rr])
            lo = np.array([x[key.replace("lift", "lift_ci_lo")] for x in rr])
            hi = np.array([x[key.replace("lift", "lift_ci_hi")] for x in rr])
            ax.errorbar([x["prefix_frames"] for x in rr], y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=tname)
        ax.axhline(1.0, color="gray", linestyle=":", label="chance")
        ax.set_xticks(prefixes)
        ax.set_xlabel("observed frames (first k of 8)")
        ax.set_title(title + " (out-of-fold, CI over videos)", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lift_vs_frames.png", dpi=160)
    fig.savefig(out / "lift_vs_frames.svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temporal-cache", type=Path, help=".../egointent_full/cache/temporal_intention_v1 (train/ val/ [test/ is never read])")
    p.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--stage-a-cache", type=Path, default=None, help="optional: baseline target from SmolVLM why_feature")
    p.add_argument("--why-pca", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "scene_cv_v1")
    p.add_argument("--n-folds", type=int, default=8, help="<=0: leave-one-scene-out")
    p.add_argument("--prefixes", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=100, help="max epochs of the inner selection run")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="1 seed, 8 epochs, 3 folds, prefixes 2/8, tmp output")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if args.temporal_cache is None:
        p.error("--temporal-cache is required (path is deliberately not guessed)")
    if (args.stage_a_cache is None) != (args.why_pca is None):
        p.error("--stage-a-cache and --why-pca go together")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds, args.epochs, args.n_folds, args.prefixes = args.seeds[:1], 8, 3, [2, 8]
        if args.output_dir == REPO_ROOT / "results" / "scene_cv_v1":
            args.output_dir = Path("/tmp/scene_cv_smoke")
        args.overwrite = args.rerun = True
    run(args)


if __name__ == "__main__":
    main()