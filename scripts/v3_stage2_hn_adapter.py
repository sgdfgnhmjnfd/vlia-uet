#!/usr/bin/env python3
"""prereg v3 Stage 2: hard-negative residual adapter on frozen so400m embeddings.

z' = norm(z + W z), W (1152x1152) zero-initialised; text side frozen. z = F's
aggregated query (A0 mean of the first k C2 frames; V1 FAILED so F's
aggregation is used). InfoNCE (tau 0.05) over the positive WHY, all other
distinct same-video WHY texts (hard) and the other-video WHY texts of the other
queries in the batch (random, <= 63 for a 64-query batch). Control: no hard
negatives, 64 sampled other-video WHY texts. AdamW lr 1e-3; wd x epochs grid
chosen by inner leave-one-scene-out (seed 0); final seeds 0,1,2.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ladder_v1 as L  # noqa: E402
import v3_common as V  # noqa: E402

TAU = 0.05
LR = 1e-3
WDS = (1e-2, 1e-1, 1.0)
EPOCHS = (5, 20)
CONFIGS = [(wd, ep) for wd in WDS for ep in EPOCHS]
SEEDS = (0, 1, 2)
BATCH = 64
OUT = ROOT / "results/v3_stage2_hn_adapter"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Fold:
    """F-processed query embeddings and per-query text tables for one training set."""

    def __init__(self, data, train_idx):
        fr, ta, tb, ub = V.prep_f(data, train_idx)
        self.z = np.stack([L.normalize(fr[:, :k].mean(axis=1)) for k in V.PREFIXES], axis=1)  # (N,4,D)
        self.ta, self.tb, self.ub = ta, tb, ub
        self.ta_t = torch.tensor(ta, device=DEV)
        self.tb_t = torch.tensor(tb, device=DEV)
        self.ub_t = torch.tensor(ub, device=DEV)

    def text_for(self, rows, cols):
        """text vectors (rows' mode) for candidate sample indices cols (same shape as rows broadcast)."""
        return np.where(self.ub[rows][..., None], self.tb[cols], self.ta[cols])


def hard_lists(data):
    """For each query: representative sample index of every other distinct WHY in its video."""
    reps = defaultdict(dict)
    for i in range(data.n):
        reps[data.vids[i]].setdefault(data.texts[i], i)
    out = []
    for i in range(data.n):
        out.append([j for t, j in reps[data.vids[i]].items() if t != data.texts[i]])
    width = max(len(h) for h in out)
    arr = np.full((data.n, width), -1, dtype=np.int64)
    for i, h in enumerate(out):
        arr[i, : len(h)] = h
    return arr


def train_adapter(data, fold: Fold, train_idx, wd, epochs, seed, hard, control=False, forbid=None):
    if forbid is not None:
        assert not np.isin(train_idx, forbid).any(), "held-out queries in training set"
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    D = fold.z.shape[-1]
    W = torch.zeros(D, D, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([W], lr=LR, weight_decay=wd)
    z_all = torch.tensor(fold.z, device=DEV)
    tid = data.text_id
    for _ in range(epochs):
        perm = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
        for s in range(0, len(perm), BATCH):
            b = perm[s:s + BATCH]
            if forbid is not None:
                assert not np.isin(b, forbid).any()
            if control:
                pool = train_idx[rng.integers(0, len(train_idx), size=(len(b), 64))]
                neg = np.where(data.vids[pool] != data.vids[b][:, None], pool, -1)
            else:
                other = np.broadcast_to(b[None, :], (len(b), len(b)))
                rnd = np.where((data.vids[other] != data.vids[b][:, None]), other, -1)
                neg = np.concatenate([hard[b], rnd], axis=1)
            # drop negatives whose text equals the positive's text
            neg = np.where((neg >= 0) & (tid[np.maximum(neg, 0)] != tid[b][:, None]), neg, -1)
            cand = np.concatenate([b[:, None], neg], axis=1)  # (B, 1+M)
            mask = torch.tensor(cand >= 0, device=DEV)
            cand_t = torch.tensor(np.maximum(cand, 0), device=DEV)
            use_b = fold.ub_t[torch.tensor(b, device=DEV)][:, None, None]
            tv = torch.where(use_b, fold.tb_t[cand_t], fold.ta_t[cand_t])  # (B, C, D)
            z = z_all[torch.tensor(b, device=DEV)]  # (B, 4, D)
            zp = torch.nn.functional.normalize(z + z @ W.T, dim=-1)
            logits = torch.einsum("bkd,bcd->bkc", zp, tv) / TAU
            logits = logits.masked_fill(~mask[:, None, :], -1e9)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                     torch.zeros(logits.shape[0] * 4, dtype=torch.long, device=DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return W.detach()


def adapted(fold: Fold, W, idx):
    z = torch.tensor(fold.z[idx], device=DEV)
    return torch.nn.functional.normalize(z + z @ W.T, dim=-1).cpu().numpy()  # (n,4,D)


def eval_hits(data, fold, W, idx):
    za = adapted(fold, W, idx)
    total = 0
    for ki in range(4):
        safe = np.where(data.cands5[idx] >= 0, data.cands5[idx], 0)
        s = np.einsum("nd,ncd->nc", za[:, ki], fold.text_for(idx[:, None], safe))
        total += int(L.strict_hits(s, data.valid5[idx]).sum())
    return total


def held_metrics(data, fold, W, idx):
    za = adapted(fold, W, idx)
    res = {}
    for ki, k in enumerate(V.PREFIXES):
        q = za[:, ki]
        res[k] = V.scorer_metrics(
            lambda i, q=q: np.where(fold.ub[i][:, None], q @ fold.tb[i].T, q @ fold.ta[i].T),
            lambda i, c, q=q: np.einsum("nd,ncd->nc", q, fold.text_for(i[:, None], np.where(c >= 0, c, 0))),
            data, idx)
    return res


def mean_over_seeds(list_of_res):
    return {k: {m: np.mean([r[k][m] for r in list_of_res], axis=0) for m in list_of_res[0][k]} for k in list_of_res[0]}


def main() -> int:
    data = V.load_data()
    mf = V.f_gate(data)
    hard = hard_lists(data)
    OUT.mkdir(parents=True, exist_ok=True)
    sel_rows, parts, ctrl_parts = [], [], []
    t0 = time.time()
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        # inner leave-one-scene-out selection (seed 0); C1 means re-fitted on the inner training set
        scores = np.zeros(len(CONFIGS), dtype=np.int64)
        for s in sorted(set(data.scenes[tr])):
            inner_ho = tr[data.scenes[tr] == s]
            inner_tr = tr[data.scenes[tr] != s]
            fold_in = Fold(data, inner_tr)
            for ci, (wd, ep) in enumerate(CONFIGS):
                W = train_adapter(data, fold_in, inner_tr, wd, ep, 0, hard, forbid=np.concatenate([idx, inner_ho]))
                scores[ci] += eval_hits(data, fold_in, W, inner_ho)
        ci = int(np.argmax(scores))
        wd, ep = CONFIGS[ci]
        fold = Fold(data, tr)
        res = [held_metrics(data, fold, train_adapter(data, fold, tr, wd, ep, sd, hard, forbid=idx), idx) for sd in SEEDS]
        ctrl = [held_metrics(data, fold, train_adapter(data, fold, tr, wd, ep, sd, hard, control=True, forbid=idx), idx)
                for sd in SEEDS]
        parts.append((idx, mean_over_seeds(res)))
        ctrl_parts.append((idx, mean_over_seeds(ctrl)))
        sel_rows.append({"fold": f, "weight_decay": wd, "epochs": ep, "config_index": ci,
                         **{f"inner_hits_wd{w}_ep{e}": int(x) for (w, e), x in zip(CONFIGS, scores)}})
        print(f"fold {f}: wd={wd} epochs={ep} ({time.time() - t0:.0f}s)", flush=True)
    hn = L.pool(data, parts)
    ctl = L.pool(data, ctrl_parts)
    pair = V.paired_vs_f(data, "hn_adapter", hn, mf)
    pair_c = V.paired_vs_f(data, "random_neg_adapter (control)", ctl, mf)
    v2 = V.improves_on_f(pair)
    summary = []
    for label, m in (("F", mf), ("hn_adapter", hn), ("random_neg_adapter_control", ctl)):
        for k in V.PREFIXES:
            summary.append({"pipeline": label, "k": k, "MRR": float(m[k]["rr"].mean()),
                            "5way_acc": float(m[k]["kway5"].mean()), "10way_acc": float(m[k]["kway10"].mean())})
    V.write_csv(OUT / "summary.csv", summary)
    V.write_csv(OUT / "selected_by_fold.csv", sel_rows)
    V.write_csv(OUT / "paired_vs_F.csv", pair + pair_c)
    verdict = {"V2": {**v2, "rule": "improves on F (hard-negative adapter, mean over 3 seeds)"},
               "control": {**V.improves_on_f(pair_c), "note": "descriptive only"}}
    lines = ["# v3 Stage 2: hard-negative residual adapter (mean over 3 seeds)", "",
             "| pipeline | k | MRR | 5-way | 10-way |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['pipeline']} | {r['k']} | {r['MRR']:.4f} | {r['5way_acc']:.4f} | {r['10way_acc']:.4f} |" for r in summary]
    lines += ["", "| comparison | k | metric | delta [95% CI] |", "|---|---:|---|---|"]
    lines += [f"| {r['comparison']} | {r['k']} | {r['metric']} | {r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |"
              for r in pair + pair_c]
    lines += ["", "Selected (weight decay, epochs) per fold: " +
              ", ".join(f"{r['fold']}:({r['weight_decay']}, {r['epochs']})" for r in sel_rows),
              "", "```", json.dumps(verdict, indent=2), "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    (OUT / "config.json").write_text(json.dumps({
        "input": "F aggregation (A0 | C2 | T1, so400m); V1 FAIL", "tau": TAU, "lr": LR, "batch_queries": BATCH,
        "grid": CONFIGS, "seeds": SEEDS, "device": str(DEV), "verdicts": verdict, "selected_by_fold": sel_rows,
        **V.meta()}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
