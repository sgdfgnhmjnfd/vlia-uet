#!/usr/bin/env python3
"""Prefix-length goal-disambiguation ablation (2/4/6/8 observed frames).

Question: how does the amount of observed egocentric video (first k of the 8
cached frames) affect early goal (WHY) ranking?

Protocol (reuses train_temporal_progression_intention.py: same loaders, same
MeanPoolBaseline, same loss, same hyper-parameter defaults, same retrieval
gallery = all validation WHY targets, positives = rows with identical WHY text):

  setting "train_k_eval_k"    (main): model trained on frames[:, :k], evaluated
                                      on frames[:, :k].
  setting "train_full_eval_k" (diagnostic, training/evaluation length mismatch): model
                                      trained on all frames, evaluated on
                                      frames[:, :k].  This is what the previously
                                      saved mean_pool checkpoints did.

Two checkpoint-selection rules are reported side by side because
"best_val_epoch" picks the best epoch on the *same* validation set it reports on:
  "best_val_epoch" : pilot protocol (optimistic),
  "last_epoch"     : no validation-based selection.

This script loads train and validation caches only; it does not load or score a
test split. Validation results are descriptive, and best-epoch metrics are
optimistic because checkpoint selection uses the same validation set.

Usage (see --help).  Cache/PCA paths are required on purpose: they are not
guessed.  `--selftest` runs the numpy-only checks and needs no data/torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SETTINGS = ("train_k_eval_k", "train_full_eval_k")
SELECTIONS = ("best_val_epoch", "last_epoch")


# ======================================================================
# numpy-only utilities (unit-tested by --selftest)
# ======================================================================

def first_positive_rank(sim: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """1-based rank of the best-scoring positive for every query row.

    Equals the argsort-based definition used by the project's
    retrieval_metrics (first positive in descending-similarity order) when
    there are no exact ties between a positive and a negative.
    """
    max_pos = np.where(pos, sim, -np.inf).max(axis=1)
    return 1 + ((sim > max_pos[:, None]) & ~pos).sum(axis=1)


def rank_metrics(ranks) -> dict:
    r = np.asarray(ranks, dtype=np.float64)
    return {
        "n_examples": int(r.size),
        "MRR": float((1.0 / r).mean()),
        "Top1": float((r <= 1).mean()),
        "Top3": float((r <= 3).mean()),
        "Top5": float((r <= 5).mean()),
        "mean_rank": float(r.mean()),
        "median_rank": float(np.median(r)),
    }


def error_breakdown(top1_idx, pos: np.ndarray, tasks) -> dict:
    """Top-1 outcome split into correct / same-task wrong / cross-task wrong.

    tasks=None means task metadata is unavailable -> split reported as None.
    """
    top1_idx = np.asarray(top1_idx)
    n = top1_idx.size
    correct = pos[np.arange(n), top1_idx]
    out = {"top1_correct": int(correct.sum()), "top1_wrong": int(n - correct.sum())}
    if tasks is None:
        out.update(same_task_wrong=None, cross_task_wrong=None,
                   same_task_wrong_fraction=None)
        return out
    tasks = np.asarray(tasks)
    same = (~correct) & (tasks[top1_idx] == tasks)
    cross = (~correct) & (tasks[top1_idx] != tasks)
    wrong = n - int(correct.sum())
    out.update(
        same_task_wrong=int(same.sum()),
        cross_task_wrong=int(cross.sum()),
        same_task_wrong_fraction=(float(same.sum()) / wrong if wrong else None),
    )
    return out


def _chance_one(n: int, m: int):
    """Exact (E[1/rank], P(top1), P(top3)) for m positives among n items under
    a uniformly random ordering."""
    q = 1.0  # P(first r-1 items are all negatives)
    e_rr = 0.0
    top3 = 0.0
    for r in range(1, n - m + 2):
        p = q * m / (n - r + 1)
        e_rr += p / r
        if r <= 3:
            top3 += p
        q *= (n - m - r + 1) / (n - r + 1)
    return e_rr, m / n, top3


def chance_levels(why_texts) -> dict:
    """Random-ranking expectation for this gallery (multi-positive)."""
    n = len(why_texts)
    counts = defaultdict(int)
    for t in why_texts:
        counts[t] += 1
    cache = {}
    mrr = top1 = top3 = 0.0
    for t in why_texts:
        m = counts[t]
        if m not in cache:
            cache[m] = _chance_one(n, m)
        a, b, c = cache[m]
        mrr += a
        top1 += b
        top3 += c
    return {"MRR": mrr / n, "Top1": top1 / n, "Top3": top3 / n}


def mean_std(values) -> tuple:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def paired_difference(ranks_a, ranks_b, n_boot: int = 10000, rng_seed: int = 0) -> dict:
    """Paired per-example comparison of two prefix lengths (b minus a).

    ranks_*: [n_seeds, n_examples] first-positive ranks on the SAME examples.
    The CI is an example-level bootstrap on the seed-averaged reciprocal rank.
    It ignores clustering by source video (only 2 validation videos), so it
    understates uncertainty: treat it as descriptive.
    """
    ra = np.asarray(ranks_a, dtype=np.float64)
    rb = np.asarray(ranks_b, dtype=np.float64)
    assert ra.shape == rb.shape, (ra.shape, rb.shape)
    diff = 1.0 / rb - 1.0 / ra                      # [seeds, examples]
    per_seed = diff.mean(axis=1)
    d = diff.mean(axis=0)                           # seed-averaged, per example
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    hit_a, hit_b = ra <= 1, rb <= 1
    return {
        "delta_MRR_mean": float(d.mean()),
        "delta_MRR_ci95_lo": float(lo),
        "delta_MRR_ci95_hi": float(hi),
        "delta_MRR_per_seed": [float(x) for x in per_seed],
        "seeds_with_positive_delta": int((per_seed > 0).sum()),
        "n_seeds": int(per_seed.size),
        "top1_gains": int((hit_b & ~hit_a).sum()),     # pooled over seeds x examples
        "top1_losses": int((hit_a & ~hit_b).sum()),
        "n_examples": int(d.size),
    }


def selftest() -> None:
    rng = np.random.default_rng(0)

    # 1. vectorised rank == argsort-based definition (project's retrieval_metrics)
    n = 60
    texts = [f"why{rng.integers(0, 15)}" for _ in range(n)]
    pos = np.array([[a == b for b in texts] for a in texts])
    sim = rng.normal(size=(n, n))
    fast = first_positive_rank(sim, pos)
    for i in range(n):
        order = np.argsort(-sim[i])
        brute = next(r for r, j in enumerate(order, 1) if pos[i, j])
        assert fast[i] == brute, (i, fast[i], brute)

    # 2. exact chance expectation == Monte Carlo
    n, groups = 40, [1] * 20 + [2] * 10 + [5] * 4
    texts = [f"g{g}" for g, m in enumerate(groups) for _ in range(m)]
    assert len(texts) == 20 + 20 + 20
    exact = chance_levels(texts)
    n = len(texts)
    pos = np.array([[a == b for b in texts] for a in texts])
    mc = []
    for _ in range(300):
        mc.append(rank_metrics(first_positive_rank(rng.normal(size=(n, n)), pos)))
    for key in ("MRR", "Top1", "Top3"):
        est = np.mean([x[key] for x in mc])
        assert abs(est - exact[key]) < 0.01, (key, est, exact[key])

    # 3. rank metrics / breakdown on a hand-made case
    m = rank_metrics([1, 2, 4])
    assert abs(m["MRR"] - (1 + 0.5 + 0.25) / 3) < 1e-12 and abs(m["Top3"] - 2 / 3) < 1e-12
    pos3 = np.eye(3, dtype=bool)
    bd = error_breakdown([0, 0, 1], pos3, ["a", "a", "b"])  # q0 ok; q1 wrong same-task; q2 wrong cross-task
    assert (bd["top1_correct"], bd["same_task_wrong"], bd["cross_task_wrong"]) == (1, 1, 1)
    assert error_breakdown([0, 0, 1], pos3, None)["same_task_wrong"] is None

    # 4. paired difference: identical -> 0; strictly better -> positive CI
    r = rng.integers(1, 50, size=(3, 100))
    same = paired_difference(r, r)
    assert same["delta_MRR_mean"] == 0.0 and same["top1_gains"] == same["top1_losses"] == 0
    better = paired_difference(r, np.maximum(1, r // 2))
    assert better["delta_MRR_ci95_lo"] > 0 and better["seeds_with_positive_delta"] == 3
    print("selftest: PASS (rank==argsort, chance==Monte Carlo, breakdown, paired diff)")


# ======================================================================
# helpers: IO
# ======================================================================

def _clean(o):
    """JSON-safe copy (NaN/inf -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list, fields: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def fmt(m, s=None, nd=4):
    if m is None:
        return "NA"
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


# ======================================================================
# training / evaluation (torch is imported lazily so --selftest works anywhere)
# ======================================================================

def predict_prefix(model, frames, k):
    """The ONLY place a prefix is formed: first k frames, nothing else."""
    return model(frames[:, :k])


def run_one(T, torch, seed, train_len, data, args):
    """Train one MeanPoolBaseline on frames[:, :train_len] and evaluate it."""
    dev = data["device"]
    T.set_seed(seed)  # same seed -> same init for every prefix length
    model = T.MeanPoolBaseline(
        hidden_dim=args.hidden_dim, output_dim=256, dropout=args.dropout
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def val_mrr(k):
        model.eval()
        with torch.no_grad():
            z = predict_prefix(model, data["val_frames"], k)
            sim = (z @ data["val_why_target"].T).cpu().numpy()
        return float((1.0 / first_positive_rank(sim, data["val_pos_np"])).mean())

    best_mrr, best_epoch, best_state, hist = -1.0, -1, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = predict_prefix(model, data["train_frames"], train_len)
        loss = (
            T.multi_positive_infonce(pred, data["train_why_target"], data["train_pos"], args.temperature)
            + args.cosine_weight * T.cosine_target_loss(pred, data["train_why_target"])
        )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        mrr = val_mrr(train_len)          # selection metric: val MRR at the model's own length
        hist.append(mrr)
        if mrr > best_mrr:                # strict '>' as in the pilot protocol
            best_mrr, best_epoch, best_state = mrr, epoch, T.clone_state_dict(model)
    last_state = T.clone_state_dict(model)

    eval_lens = (sorted(set(args.prefixes) | {train_len})
                 if train_len == data["full_len"] else [train_len])
    evals, crosscheck = {}, {}
    for sel, state in (("best_val_epoch", best_state), ("last_epoch", last_state)):
        model.load_state_dict(state)
        model.eval()
        evals[sel] = {}
        with torch.no_grad():
            for k in eval_lens:
                z = predict_prefix(model, data["val_frames"], k)
                sim = (z @ data["val_why_target"].T).cpu().numpy()
                evals[sel][k] = {
                    "ranks": first_positive_rank(sim, data["val_pos_np"]).tolist(),
                    "top1_idx": sim.argmax(axis=1).tolist(),
                }
            # cross-check my vectorised ranks against the project's own metric
            z = predict_prefix(model, data["val_frames"], train_len)
            ref = T.retrieval_metrics(z, data["val_why_target"], data["val_why_texts"])
        mine = rank_metrics(evals[sel][train_len]["ranks"]) if train_len in evals[sel] else None
        if mine is not None:
            crosscheck[sel] = {
                "MRR_abs_diff": abs(mine["MRR"] - ref["MRR"]),
                "Top1_abs_diff": abs(mine["Top1"] - ref["Top1"]),
                "Top3_abs_diff": abs(mine["Top3"] - ref["Top3"]),
            }
    return {
        "seed": seed, "train_len": train_len, "best_epoch": best_epoch,
        "best_val_mrr_at_train_len": best_mrr, "val_mrr_history": hist,
        "evals": {s: {str(k): v for k, v in d.items()} for s, d in evals.items()},
        "crosscheck_vs_project_retrieval_metrics": crosscheck,
    }, model, best_state


def run(args):
    import torch  # noqa: lazy
    sys.path.insert(0, str(SCRIPT_DIR))
    import train_temporal_progression_intention as T

    out = args.output_dir
    runs_dir = out / "runs"
    if (out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{out}/summary.json already exists. Use a new --output-dir, or --overwrite "
                 "to rewrite this script's own files (cached runs/ are reused either way "
                 "unless --rerun).")
    runs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    why_transform = T.load_pca_transform(args.why_pca)
    train_rows = T.load_temporal_split(args.temporal_cache, args.stage_a_cache, "train")
    val_rows = T.load_temporal_split(args.temporal_cache, args.stage_a_cache, "val")

    # ---------------- leakage / consistency guards ----------------
    tr_vid = {r["video_uid"] for r in train_rows}
    va_vid = {r["video_uid"] for r in val_rows}
    assert not (tr_vid & va_vid), f"source videos shared across splits: {sorted(tr_vid & va_vid)}"
    tr_ids = [r["sample_id"] for r in train_rows]
    va_ids = [r["sample_id"] for r in val_rows]
    assert len(set(tr_ids)) == len(tr_ids) and len(set(va_ids)) == len(va_ids), "duplicate sample_id"
    assert not (set(tr_ids) & set(va_ids)), "sample_id shared across splits"

    train_frames, train_why_target, train_why_texts = T.stack_split(train_rows, why_transform, device)
    val_frames, val_why_target, val_why_texts = T.stack_split(val_rows, why_transform, device)
    full_len = int(train_frames.shape[1])
    assert int(val_frames.shape[1]) == full_len, "train/val frame count differ"
    assert train_frames.shape[2] == 960 and val_frames.shape[2] == 960, "expected 960-d features"
    assert torch.isfinite(train_frames).all() and torch.isfinite(val_frames).all(), "non-finite features"
    args.prefixes = sorted(set(args.prefixes))
    assert args.prefixes[0] >= 1 and args.prefixes[-1] <= full_len, (
        f"prefixes {args.prefixes} must lie in [1, {full_len}]")
    assert torch.allclose(val_why_target.norm(dim=-1), torch.ones(len(val_rows), device=device), atol=1e-4)

    val_pos = T.build_positive_mask(val_why_texts, device)
    data = {
        "device": device, "full_len": full_len,
        "train_frames": train_frames, "train_why_target": train_why_target,
        "train_pos": T.build_positive_mask(train_why_texts, device),
        "val_frames": val_frames, "val_why_target": val_why_target,
        "val_why_texts": val_why_texts, "val_pos_np": val_pos.cpu().numpy(),
    }
    tasks = [r["task"] for r in val_rows]
    task_ok = len({t for t in tasks if t}) >= 2
    tasks_arr = np.asarray(tasks) if task_ok else None
    if not task_ok:
        print("WARNING: task metadata missing/degenerate in val rows -> same/cross-task split = NA")
    chance = chance_levels(val_why_texts)

    print("=" * 78)
    print(f"train: {len(train_rows)} samples / {len(tr_vid)} videos | "
          f"val: {len(val_rows)} samples / {len(va_vid)} videos | frames={full_len}")
    print(f"prefixes={args.prefixes} seeds={args.seeds} epochs={args.epochs} device={device}")
    print("chance (exact, multi-positive): "
          + " ".join(f"{k}={v:.4f}" for k, v in chance.items())
          + "   (computed from this validation gallery)")
    print("=" * 78)

    # ---------------- train/evaluate (cached per run) ----------------
    train_lens = sorted(set(args.prefixes) | {full_len})
    run_results = {}
    for L in train_lens:
        for seed in args.seeds:
            p = runs_dir / f"train{L}_seed{seed}.json"
            if p.exists() and not args.rerun:
                run_results[(L, seed)] = json.loads(p.read_text())
                print(f"[cached] train_len={L} seed={seed}")
                continue
            t0 = time.time()
            res, model, _ = run_one(T, torch, seed, L, data, args)
            write_json(p, res)
            run_results[(L, seed)] = res
            cc = res["crosscheck_vs_project_retrieval_metrics"]
            worst = max((v for d in cc.values() for v in d.values()), default=0.0)
            flag = "" if worst < 1e-5 else "   <-- CROSSCHECK MISMATCH"
            e = res["evals"]["best_val_epoch"][str(L)]
            m = rank_metrics(e["ranks"])
            print(f"train_len={L} seed={seed} best_epoch={res['best_epoch']:3d} "
                  f"val MRR={m['MRR']:.4f} Top1={m['Top1']:.4f} ({time.time()-t0:.1f}s){flag}")
            if args.smoke and worst >= 1e-5:
                raise RuntimeError(f"vectorised metrics disagree with project metrics: {cc}")
            # causality: changing frames after position k must not change the prefix prediction
            for k in args.prefixes:
                noisy = data["val_frames"].clone()
                noisy[:, k:] = torch.randn_like(noisy[:, k:]) * 1e3
                model.eval()
                with torch.no_grad():
                    a = predict_prefix(model, data["val_frames"], k)
                    b = predict_prefix(model, noisy, k)
                assert torch.allclose(a, b), f"future frames influence prefix k={k}"

    # ---------------- assemble ----------------
    ranks, top1 = {}, {}
    best_epochs = defaultdict(dict)
    for setting in SETTINGS:
        for sel in SELECTIONS:
            for k in args.prefixes:
                L = k if setting == "train_k_eval_k" else full_len
                rs, ts = [], []
                for seed in args.seeds:
                    ev = run_results[(L, seed)]["evals"][sel][str(k)]
                    rs.append(ev["ranks"])
                    ts.append(ev["top1_idx"])
                    best_epochs[(setting, k)][seed] = run_results[(L, seed)]["best_epoch"]
                ranks[(setting, sel, k)] = np.asarray(rs)
                top1[(setting, sel, k)] = np.asarray(ts)

    per_seed_rows, summary_rows = [], []
    for (setting, sel, k), R in sorted(ranks.items()):
        metr = []
        for si, seed in enumerate(args.seeds):
            m = rank_metrics(R[si])
            m.update(error_breakdown(top1[(setting, sel, k)][si], data["val_pos_np"], tasks_arr))
            row = {"setting": setting, "selection": sel, "prefix_frames": k, "seed": seed,
                   "best_epoch": best_epochs[(setting, k)][seed] if sel == "best_val_epoch" else args.epochs}
            row.update(m)
            per_seed_rows.append(row)
            metr.append(m)
        srow = {"setting": setting, "selection": sel, "prefix_frames": k,
                "n_seeds": len(args.seeds), "n_examples_per_seed": int(R.shape[1])}
        for key in ("MRR", "Top1", "Top3", "Top5", "mean_rank", "same_task_wrong_fraction"):
            mu, sd = mean_std([m.get(key) for m in metr])
            srow[f"{key}_mean"], srow[f"{key}_std"] = mu, sd
        for key in ("top1_correct", "same_task_wrong", "cross_task_wrong"):
            vals = [m.get(key) for m in metr]
            srow[f"{key}_total"] = None if any(v is None for v in vals) else int(sum(vals))
        summary_rows.append(srow)

    paired_rows = []
    for setting in SETTINGS:
        for sel in SELECTIONS:
            ks = args.prefixes
            pairs = [(a, b) for a, b in zip(ks[:-1], ks[1:])] + ([(ks[0], ks[-1])] if len(ks) > 2 else [])
            for a, b in pairs:
                d = paired_difference(ranks[(setting, sel, a)], ranks[(setting, sel, b)])
                paired_rows.append({"setting": setting, "selection": sel, "from_k": a, "to_k": b, **d})

    # ---------------- write ----------------
    fields_seed = ["setting", "selection", "prefix_frames", "seed", "n_examples", "MRR", "Top1", "Top3",
                   "Top5", "mean_rank", "median_rank", "top1_correct", "top1_wrong", "same_task_wrong",
                   "cross_task_wrong", "same_task_wrong_fraction", "best_epoch"]
    write_csv(out / "per_seed_metrics.csv", per_seed_rows, fields_seed)
    fields_sum = ["setting", "selection", "prefix_frames", "n_seeds", "n_examples_per_seed",
                  "MRR_mean", "MRR_std", "Top1_mean", "Top1_std", "Top3_mean", "Top3_std", "Top5_mean",
                  "Top5_std", "mean_rank_mean", "same_task_wrong_fraction_mean", "top1_correct_total",
                  "same_task_wrong_total", "cross_task_wrong_total"]
    write_csv(out / "summary.csv", summary_rows, fields_sum)
    pf = ["setting", "selection", "from_k", "to_k", "delta_MRR_mean", "delta_MRR_ci95_lo",
          "delta_MRR_ci95_hi", "seeds_with_positive_delta", "n_seeds", "top1_gains", "top1_losses",
          "n_examples"]
    write_csv(out / "paired_differences.csv", paired_rows, pf)

    ex_fields = ["sample_id", "video_uid", "task", "why"]
    ex_rows = [{"sample_id": r["sample_id"], "video_uid": r["video_uid"], "task": r["task"], "why": r["why"]}
               for r in val_rows]
    for (setting, sel, k), R in sorted(ranks.items()):
        for si, seed in enumerate(args.seeds):
            col = f"rank|{setting}|{sel}|k{k}|seed{seed}"
            ex_fields.append(col)
            for i, row in enumerate(ex_rows):
                row[col] = int(R[si, i])
    write_csv(out / "per_example_ranks.csv", ex_rows, ex_fields)

    config = {
        "script": "scripts/run_prefix_length_ablation.py", "git_commit": git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "torch": torch.__version__, "numpy": np.__version__,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "train_videos": sorted(tr_vid), "val_videos": sorted(va_vid), "frames_per_sample": full_len,
        "task_metadata_available": task_ok, "chance": chance,
        "crosscheck_max_abs_diff": max(
            (v for r in run_results.values() for d in r["crosscheck_vs_project_retrieval_metrics"].values()
             for v in d.values()), default=None),
        "frame_sampling_note": ("cached frames are np.linspace over the whole clip; prefix k = first k of "
                                f"{full_len} frames (~{[round(100*k/full_len) for k in args.prefixes]}% of the "
                                "sampled grid), not an absolute time"),
    }
    write_json(out / "config.json", config)
    summary = {"config_ref": "config.json", "chance": chance, "summary": summary_rows,
               "paired_differences": paired_rows}
    write_json(out / "summary.json", summary)
    write_markdown(
        out / "summary.md",
        summary_rows,
        paired_rows,
        chance,
        args,
        task_ok,
        n_val_examples=len(val_rows),
        n_val_videos=len(va_vid),
    )
    plot(out, summary_rows, chance, args.prefixes)
    print(f"\nDONE. Outputs in {out}")


def write_markdown(
    path,
    summary_rows,
    paired_rows,
    chance,
    args,
    task_ok,
    n_val_examples,
    n_val_videos,
):
    L = ["# Prefix-length goal-disambiguation ablation (validation only)", ""]
    L.append(f"Seeds: {args.seeds}; epochs: {args.epochs}; gallery = all validation WHY targets; "
             "positives = rows with identical WHY text. Values are mean ± sample std over seeds. "
             f"Chance (exact, multi-positive): MRR {chance['MRR']:.4f}, Top-1 {chance['Top1']:.4f}, "
             f"Top-3 {chance['Top3']:.4f}.")
    L.append(
        f"Validation contains {n_val_examples} examples from {n_val_videos} source-video UIDs. "
        "No test features were loaded and no test score is reported."
    )
    L.append("")
    for setting in SETTINGS:
        for sel in SELECTIONS:
            L.append(f"## {setting} / {sel}")
            L.append("")
            L.append("| k frames | MRR | Top-1 | Top-3 | n/seed | same-task wrong (frac of errors) | correct / same-task wrong / cross-task wrong (summed over seeds) |")
            L.append("|---:|---|---|---|---:|---:|---|")
            for r in summary_rows:
                if r["setting"] != setting or r["selection"] != sel:
                    continue
                split = (f"{r['top1_correct_total']} / {r['same_task_wrong_total']} / {r['cross_task_wrong_total']}"
                         if task_ok else f"{r['top1_correct_total']} / NA / NA")
                L.append(f"| {r['prefix_frames']} | {fmt(r['MRR_mean'], r['MRR_std'])} | "
                         f"{fmt(r['Top1_mean'], r['Top1_std'])} | {fmt(r['Top3_mean'], r['Top3_std'])} | "
                         f"{r['n_examples_per_seed']} | {fmt(r['same_task_wrong_fraction_mean'], None, 3)} | {split} |")
            L.append("")
    L.append("## Paired differences (to_k minus from_k, same examples)")
    L.append("")
    L.append("Example-level bootstrap CI on the seed-averaged reciprocal rank. It ignores clustering by source "
             "video (the validation set has few videos), so it understates uncertainty: descriptive only.")
    L.append("")
    L.append("| setting / selection | from→to | ΔMRR [95% CI] | seeds with Δ>0 | Top-1 gains / losses (pooled) |")
    L.append("|---|---|---|---:|---|")
    for r in paired_rows:
        L.append(f"| {r['setting']} / {r['selection']} | {r['from_k']}→{r['to_k']} | "
                 f"{r['delta_MRR_mean']:+.4f} [{r['delta_MRR_ci95_lo']:+.4f}, {r['delta_MRR_ci95_hi']:+.4f}] | "
                 f"{r['seeds_with_positive_delta']}/{r['n_seeds']} | {r['top1_gains']} / {r['top1_losses']} |")
    L += ["", "## Limitations", "",
          "- Validation-only; no significance or generalisation claim. Best-epoch selection and reporting use the same validation split.",
          "- `best_val_epoch` selects on the same validation set it reports on (optimistic); `last_epoch` does not.",
          "- Prefix k = first k of the 8 cached frames, sampled uniformly over the whole clip; its share of "
          "wall-clock time depends on clip length.",
          "- `train_full_eval_k` is a training/evaluation length-mismatch diagnostic, not the main comparison.",
          "- Confidence/margin/entropy decision analysis is a separate step."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(out, summary_rows, chance, prefixes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print("plot skipped:", repr(e))
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"train_k_eval_k": "tab:blue", "train_full_eval_k": "tab:orange"}
    styles = {"best_val_epoch": "-", "last_epoch": "--"}
    for ax, key, title in ((axes[0], "MRR", "MRR"), (axes[1], "Top1", "Top-1")):
        for setting in SETTINGS:
            for sel in SELECTIONS:
                rows = sorted([r for r in summary_rows if r["setting"] == setting and r["selection"] == sel],
                              key=lambda r: r["prefix_frames"])
                ax.errorbar([r["prefix_frames"] for r in rows], [r[f"{key}_mean"] for r in rows],
                            yerr=[r[f"{key}_std"] or 0 for r in rows], color=colors[setting],
                            linestyle=styles[sel], marker="o", capsize=3, label=f"{setting} / {sel}")
        ax.axhline(chance[key], color="gray", linestyle=":", label="chance (exact)")
        ax.set_xticks(prefixes)
        ax.set_xlabel("observed frames (first k of 8)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs observed frames (val, mean ± std over seeds)", fontsize=9)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "prefix_length_vs_ranking.png", dpi=160)
    fig.savefig(out / "prefix_length_vs_ranking.svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temporal-cache", type=Path, help="dir with train/ and val/ frame-level .pt files")
    p.add_argument("--stage-a-cache", type=Path, help="dir with train/ and val/ Stage-A .pt (why/video_uid/task)")
    p.add_argument("--why-pca", type=Path, help="why_pca_960_to_256.pt")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "prefix_length_ablation_v2")
    p.add_argument("--prefixes", nargs="+", type=int, default=[2, 4, 6, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="1 seed, 3 epochs, tmp output dir, strict cross-checks")
    p.add_argument("--selftest", action="store_true", help="numpy-only unit checks; needs no data/torch")
    p.add_argument("--overwrite", action="store_true", help="rewrite this script's own summary files")
    p.add_argument("--rerun", action="store_true", help="ignore cached runs/ and retrain")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    missing = [n for n in ("temporal_cache", "stage_a_cache", "why_pca") if getattr(args, n) is None]
    if missing:
        p.error("required (paths are deliberately not guessed): " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds, args.epochs = args.seeds[:1], 3
        if args.output_dir == REPO_ROOT / "results" / "prefix_length_ablation_v2":
            args.output_dir = Path("/tmp/prefix_ablation_smoke")
        args.overwrite = args.rerun = True
    run(args)


if __name__ == "__main__":
    main()