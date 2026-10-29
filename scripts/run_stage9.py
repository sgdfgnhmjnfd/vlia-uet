#!/usr/bin/env python3
"""Stage 9: improving the prediction part (rules: `results/prereg_stage9_prediction.md`).

Subcommands
  selftest     synthetic checks of the helpers (no data, no model)
  recognition  9b: F's current-step WHAT recognition within the video at
               k = 2/4/6/8 (row match, raw / normalised text match, recall@5/10
               over distinct texts) -> results/stage9b_recognition/   (CPU)
  qwen         9c: Qwen2.5-VL-3B re-ranks F's top-5 distinct WHAT texts of every
               step from its frames; one jsonl line per (sample_id, k); k=8
               first, then k=2; --probe N times N queries (GPU, resumable)
  antgpt       9c: recognised labels F / rec_qwen / rec_fuse (nested gamma),
               then the unchanged text-only AntGPT scorer with the new
               histories qwen_k2/k8, fuse_k2/k8 (GPU, resumable)
  analyze      9c: recognition, AntGPT and fusion summaries, R1/A1 (CPU)
               -> results/stage9c_qwen_recognition/

  export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1
  PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
  $PY scripts/test_stage9.py
  CUDA_VISIBLE_DEVICES="" $PY scripts/run_stage9.py recognition
  $PY scripts/run_stage9.py qwen --probe 30 && $PY scripts/run_stage9.py qwen
  $PY scripts/run_stage9.py antgpt
  CUDA_VISIBLE_DEVICES="" $PY scripts/run_stage9.py analyze
The locked test split is never read (train/val manifest only).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402
import run_ladder_v1 as L  # noqa: E402
import run_literature_baselines as LB  # noqa: E402
import run_v3_supplementary as V  # noqa: E402

PREREG = ROOT / "results/prereg_stage9_prediction.md"
OUT_B = ROOT / "results/stage9b_recognition"
OUT_C = ROOT / "results/stage9c_qwen_recognition"
REC_PREFIXES = (2, 4, 6, 8)
QWEN_PREFIXES = (8, 2)  # run order
SHORTLIST = 5
GAMMAS = LB.GAMMAS
STORED_REC = ROOT / "results/lit_baselines_next/config.json"
OLD_REC_K8 = 0.29165
QUESTION = ("These frames are from a first-person video of one step of a task. "
            "Which action is the person doing in this step? Options: {options} Answer with one letter.")
NEW_VARIANTS = ("qwen_k2", "qwen_k8", "fuse_k2", "fuse_k8")


# ======================================================================
# pure helpers (tested in scripts/test_stage9.py)
# ======================================================================

def norm_text(t: str) -> str:
    return " ".join(t.lower().split())


def rank_texts(scores, rows_g, texts):
    """Distinct normalised texts of a gallery, best first.

    scores (G,) F scores of the gallery rows rows_g; a text scores the max over
    its rows and is shown in the wording of that row. Ties keep the order of
    first appearance in the gallery. Returns [(norm, display, score), ...]."""
    best = {}
    for s, r in zip(scores, rows_g):
        key = norm_text(texts[r])
        if key not in best or s > best[key][1]:
            best[key] = (texts[r], float(s))
    items = [(k, d, s) for k, (d, s) in best.items()]
    order = np.argsort([-s for _, _, s in items], kind="stable")
    return [items[j] for j in order]


def recall_at(ranked, true_norm, k):
    return float(true_norm in [t for t, _, _ in ranked[:k]])


def zscore_rows(x):
    """Row-wise z-score ignoring NaN (missing options stay NaN)."""
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def pick(scores):
    """Index of the best option; NaN = missing; ties -> earliest (F's ranking)."""
    return np.argmax(np.where(np.isnan(scores), -np.inf, scores), axis=1)


def nested_gamma_rec(zf, zq, correct, fold_of, folds):
    """Per outer fold choose gamma on the other folds (max recognition accuracy, ties -> smaller gamma)."""
    chosen_pick, chosen = np.zeros(len(zf), int), {}
    rows = np.arange(len(zf))
    for f in folds:
        test, train = fold_of == f, fold_of != f
        acc = [correct[rows[train], pick(zf[train] + g * zq[train])].mean() for g in GAMMAS]
        g = GAMMAS[int(np.argmax(acc))]
        chosen[int(f)] = g
        chosen_pick[test] = pick(zf[test] + g * zq[test])
    return chosen_pick, chosen


def history_from_labels(members, i, label8, labelk):
    """Recognised WHAT texts of up to HISTORY previous steps (k=8) plus the current step (k)."""
    m = members[i]
    p = m.index(i)
    steps = m[max(0, p - LB.HISTORY): p + 1]
    return [label8[j] for j in steps[:-1]] + [labelk[i]]


def options_text(texts):
    return " ".join(f"{a}. {t}" for a, t in zip(LB.LETTERS, texts))


# ======================================================================
# F gallery scores (same code path as run_literature_baselines.cmd_next)
# ======================================================================

def f_gallery(data, ks, device):
    nxt_texts, _ = LB.next_setup(data)
    whats = [r["what"] for r in data.rows]
    vocab = sorted(set(nxt_texts) | set(whats))
    emb = AH.encode_texts(V.ENCODER_ID, vocab, device=device)
    tix = {t: i for i, t in enumerate(vocab)}
    what_raw = emb[[tix[t] for t in whats]]
    members = LB.ordered_members(data)
    frames = data.frames[V.ENC]
    ctx, has = data.ctx[V.ENC]
    rec_row = {k: np.full(data.n, -1) for k in ks}
    ranked = {k: [None] * data.n for k in ks}
    for fold in data.folds:
        idx, train_idx = data.fold_idx(fold)
        fr, _, _, _ = L.processed(frames, emb[[tix[t] for t in nxt_texts]], "C2", train_idx, ctx, has)
        _, wa, wb, ub = L.processed(frames, what_raw, "C2", train_idx, ctx, has)
        for k in ks:
            q = V.query_vec(fr[idx], "A0", k)
            for li, i in enumerate(idx):
                g = np.asarray(members[data.vids[i]])
                s = (wb[g] if ub[i] else wa[g]) @ q[li]
                rec_row[k][i] = g[int(np.argmax(s))]
                ranked[k][i] = rank_texts(s, g, whats)
    return rec_row, ranked, members


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def acc_row(name, k, values, vids, **extra):
    p, lo, hi = LB.boot(values, vids)
    return {"measure": name, "k": k, "n": int(np.sum(~np.isnan(values))), "value": p, "lo": lo, "hi": hi, **extra}


# ======================================================================
# 9b
# ======================================================================

def cmd_recognition(args) -> int:
    if (OUT_B / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{OUT_B}/summary.csv exists (use --overwrite)")
    data = V.load_data()
    V.gate(data)
    rec_row, ranked, members = f_gallery(data, REC_PREFIXES, args.device)
    whats = [r["what"] for r in data.rows]
    norms = [norm_text(t) for t in whats]
    stored = json.loads(STORED_REC.read_text())["what_recognition_top1_in_video"]
    summary, bad = [], {}
    for k in REC_PREFIXES:
        rr = rec_row[k]
        row = np.array([float(rr[i] == i) for i in range(data.n)])
        raw = np.array([float(whats[rr[i]] == whats[i]) for i in range(data.n)])
        nrm = np.array([float(norms[rr[i]] == norms[i]) for i in range(data.n)])
        if abs(raw.mean() - stored[str(k)]) > 1e-9:
            bad[k] = (raw.mean(), stored[str(k)])
        summary.append(acc_row("row_match", k, row, data.vids))
        summary.append(acc_row("text_match_raw", k, raw, data.vids))
        summary.append(acc_row("text_match_norm", k, nrm, data.vids))
        for top in (1, 5, 10):
            r = np.array([recall_at(ranked[k][i], norms[i], top) for i in range(data.n)])
            summary.append(acc_row(f"recall@{top}_distinct_text", k, r, data.vids))
    print(f"GATE raw text match reproduces {STORED_REC.name}: {'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
    if bad:
        raise SystemExit("gate failed: recognition does not reproduce the stored numbers; stopping")
    n_distinct = np.array([len({norms[i] for i in m}) for m in members.values()])
    n_steps = np.array([len(m) for m in members.values()])
    OUT_B.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_B / "summary.csv", summary)
    (OUT_B / "config.json").write_text(json.dumps({
        "encoder": V.ENCODER_ID, "aggregation": "A0", "centering": "C2", "gallery": "all steps of the same video",
        "B": LB.B, "stored_reference": str(STORED_REC.relative_to(ROOT)), "gate": "PASS",
        "distinct_what_per_video": {"median": float(np.median(n_distinct)), "min": int(n_distinct.min()),
                                    "max": int(n_distinct.max()), "per_video": n_distinct.tolist()},
        "steps_per_video": {"median": float(np.median(n_steps)), "min": int(n_steps.min()), "max": int(n_steps.max())},
        "test_split_read": False}, indent=2))
    for r in summary:
        print(f"k={r['k']} {r['measure']:<28} {100 * r['value']:5.1f} [{100 * r['lo']:.1f}; {100 * r['hi']:.1f}]")
    print(f"distinct WHAT texts per video: median {np.median(n_distinct)}, min {n_distinct.min()}, "
          f"max {n_distinct.max()} (steps per video median {np.median(n_steps)})")
    print(f"saved {OUT_B}", flush=True)
    return 0


# ======================================================================
# 9c: shortlists and Qwen vision re-ranking
# ======================================================================

def shortlists(data, device):
    """{sample_id: {k: [{"norm", "text", "f_score"}, ...]}} for k in (2, 8), cached on disk."""
    path = OUT_C / "shortlists.json"
    if path.exists():
        return json.loads(path.read_text())
    _, ranked, _ = f_gallery(data, (2, 8), device)
    out = {data.sample_ids[i]: {str(k): [{"norm": n, "text": d, "f_score": s} for n, d, s in ranked[k][i][:SHORTLIST]]
                                for k in (2, 8)} for i in range(data.n)}
    OUT_C.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def resize448(frame):
    from PIL import Image

    img = Image.fromarray(frame)
    w, h = img.size
    s = 448 / max(w, h)
    return img.resize((max(28, round(w * s)), max(28, round(h * s))), Image.BICUBIC)


class VisionScorer:
    """Qwen2.5-VL-3B letter log-probs, as v3_stage3_mllm.MLLM with the 9c question."""

    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(LB.LLM_ID, token=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            LB.LLM_ID, token=False, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        tok = self.processor.tokenizer
        self.letter_ids = []
        for a in LB.LETTERS:
            ids = tok.encode(a, add_special_tokens=False)
            assert len(ids) == 1, (a, ids)
            self.letter_ids.append(ids[0])

    def letter_logprobs(self, images, texts):
        torch = self.torch
        content = [{"type": "image"} for _ in images] + \
                  [{"type": "text", "text": QUESTION.format(options=options_text(texts))}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return [float(lp[i]) for i in self.letter_ids[: len(texts)]]

    def score(self, images, texts):
        fwd = self.letter_logprobs(images, texts)
        rev = self.letter_logprobs(images, texts[::-1])[::-1]
        return {"lp_order1": fwd, "lp_order2_canonical": rev, "score": [(a + b) / 2 for a, b in zip(fwd, rev)]}


def cmd_qwen(args) -> int:
    from hoi_utils import sample_exact_video_frames

    data = V.load_data()
    sl = shortlists(data, "cpu")
    order = sorted(range(data.n), key=lambda i: data.sample_ids[i])
    if args.probe:
        order = order[: args.probe]
    base = OUT_C / ("probe" if args.probe else "")
    base.mkdir(parents=True, exist_ok=True)
    scorer = VisionScorer()

    def load(i):
        return [resize448(f) for f in sample_exact_video_frames(data.rows[i]["video_path"])]

    t_all, n_all = time.time(), 0
    for k in (QWEN_PREFIXES[:1] if args.probe else QWEN_PREFIXES):
        path = base / f"qwen_scores_k{k}.jsonl"
        seen = set()
        if path.exists():
            seen = {json.loads(line)["sample_id"] for line in path.open() if line.strip()}
        todo = [i for i in order if data.sample_ids[i] not in seen]
        print(f"[k={k}] {len(order)} steps, {len(todo)} to score", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as pool, path.open("a", encoding="utf-8") as handle:
            futs = {}
            for n_done, i in enumerate(todo, 1):
                for j in todo[n_done - 1: n_done + 5]:  # bounded prefetch (memory: Stage 3 was OOM-killed)
                    if j not in futs:
                        futs[j] = pool.submit(load, j)
                frames = futs.pop(i).result()
                opts = sl[data.sample_ids[i]][str(k)]
                rec = {"sample_id": data.sample_ids[i], "video_uid": str(data.vids[i]), "k": k,
                       "options": [o["text"] for o in opts], "options_norm": [o["norm"] for o in opts],
                       "f_score": [o["f_score"] for o in opts]}
                rec.update(scorer.score(frames[:k], rec["options"]))
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                n_all += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"  k={k} {n_done}/{len(todo)} elapsed {el:.0f}s s/query {el / n_done:.2f} "
                          f"eta {el / n_done * (len(todo) - n_done) / 60:.0f} min", flush=True)
    el = time.time() - t_all
    info = {"model": LB.LLM_ID, "question": QUESTION, "probe": args.probe, "n_scored": n_all,
            "seconds_per_query": el / max(1, n_all)}
    if args.probe:
        info["eta_full_hours_k8_upper_bound_for_both_k"] = el / max(1, n_all) * data.n * len(QWEN_PREFIXES) / 3600
    (base / ("probe_info.json" if args.probe else "qwen_run_info.json")).write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return 0


# ======================================================================
# 9c: recognised labels, AntGPT, analysis
# ======================================================================

def load_qwen(data):
    out = {}
    for k in (2, 8):
        recs = [json.loads(line) for line in (OUT_C / f"qwen_scores_k{k}.jsonl").open() if line.strip()]
        out[k] = {r["sample_id"]: r for r in recs}
        missing = set(data.sample_ids) - set(out[k])
        if missing:
            raise SystemExit(f"qwen scores k={k} incomplete: {len(missing)} steps missing")
    return out


def recognised_labels(data, qwen):
    """F / rec_qwen / rec_fuse labels (display text) and correctness per k; gamma by fold."""
    norms = [norm_text(r["what"]) for r in data.rows]
    labels, correct, gammas, ceiling = {}, {}, {}, {}
    for k in (2, 8):
        n = data.n
        zf, zq = np.full((n, SHORTLIST), np.nan), np.full((n, SHORTLIST), np.nan)
        ok = np.zeros((n, SHORTLIST), bool)
        opts = []
        for i, s in enumerate(data.sample_ids):
            r = qwen[k][s]
            m = len(r["options"])
            zf[i, :m], zq[i, :m] = r["f_score"], r["score"]
            ok[i, :m] = [o == norms[i] for o in r["options_norm"]]
            opts.append(r["options"])
        zf, zq = zscore_rows(zf), zscore_rows(zq)
        p_f = np.zeros(n, int)
        p_q = pick(zq)
        p_u, chosen = nested_gamma_rec(zf, zq, ok, data.fold_of, data.folds)
        gammas[k] = chosen
        rows = np.arange(n)
        for name, p in (("F", p_f), ("rec_qwen", p_q), ("rec_fuse", p_u)):
            labels[(name, k)] = [opts[i][p[i]] for i in range(n)]
            correct[(name, k)] = ok[rows, p].astype(float)
        ceiling[k] = ok.any(axis=1).astype(float)
    return labels, correct, gammas, ceiling


def check_old_antgpt():
    recs = [json.loads(line) for line in (ROOT / "results/lit_baselines_next/antgpt_scores/rec_k8.jsonl").open()
            if line.strip()]
    acc = float(np.mean([LB.fractional_credit(np.asarray(r["score"])) for r in recs]))
    ok = abs(acc - OLD_REC_K8) < 5e-5
    print(f"GATE old AntGPT rec_k8 = {acc:.5f} (expected {OLD_REC_K8}): {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("gate failed: old AntGPT rec_k8 does not reproduce; stopping")
    return acc


def cmd_antgpt(args) -> int:
    data = V.load_data()
    V.gate(data)
    check_old_antgpt()
    qwen = load_qwen(data)
    labels, correct, gammas, _ = recognised_labels(data, qwen)
    members_v = LB.ordered_members(data)
    members = {i: list(members_v[data.vids[i]]) for i in range(data.n)}
    texts, c5 = LB.next_setup(data)
    order = [i for i in sorted(range(data.n), key=lambda i: data.sample_ids[i]) if (c5[i] >= 0).all()]
    (OUT_C / "recognized_labels.json").write_text(json.dumps(
        {f"{name}_k{k}": dict(zip(data.sample_ids, lab)) for (name, k), lab in labels.items()}))
    score_dir = OUT_C / "antgpt_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    llm = LB.LLM(args.device)
    t0, done = time.time(), 0
    for variant in NEW_VARIANTS:
        kind, k = variant.split("_k")
        name = "rec_qwen" if kind == "qwen" else "rec_fuse"
        lab8, labk = labels[(name, 8)], labels[(name, int(k))]
        path = score_dir / f"{variant}.jsonl"
        seen = {json.loads(line)["sample_id"] for line in path.open() if line.strip()} if path.exists() else set()
        todo = [i for i in order if data.sample_ids[i] not in seen]
        print(f"[{variant}] {len(order)} queries, {len(todo)} to score", flush=True)
        with path.open("a", encoding="utf-8") as handle:
            for n_done, i in enumerate(todo, 1):
                hist = history_from_labels(members, i, lab8, labk)
                opts = [texts[j] for j in c5[i]]
                handle.write(json.dumps({"sample_id": data.sample_ids[i], "video_uid": str(data.vids[i]),
                                         "variant": variant, "history": hist,
                                         "candidate_sample_ids": [data.sample_ids[j] for j in c5[i]],
                                         "score": llm.score(hist, opts)}) + "\n")
                handle.flush()
                done += 1
                if n_done % 250 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"  {n_done}/{len(todo)} elapsed {el:.0f}s s/query {el / done:.2f}", flush=True)
    (OUT_C / "antgpt_run_info.json").write_text(json.dumps({
        "model": LB.LLM_ID, "variants": list(NEW_VARIANTS), "n_queries": len(order),
        "seconds_per_query": (time.time() - t0) / max(1, done), "gamma_by_fold": {str(k): v for k, v in gammas.items()}},
        indent=2))
    return 0


def f_next_scores(data, device):
    """F's NEXT candidate scores out of fold at k = 2, 8 (as run_literature_baselines.cmd_fusion)."""
    texts, c5 = LB.next_setup(data)
    vocab = sorted(set(texts))
    emb = AH.encode_texts(V.ENCODER_ID, vocab, device=device)
    tix = {t: i for i, t in enumerate(vocab)}
    next_raw = emb[[tix[t] for t in texts]]
    frames = data.frames[V.ENC]
    ctx, has = data.ctx[V.ENC]
    fs = {k: np.full((data.n, 5), np.nan) for k in (2, 8)}
    for fold in data.folds:
        idx, train_idx = data.fold_idx(fold)
        fr, ta, tb, ub = L.processed(frames, next_raw, "C2", train_idx, ctx, has)
        for k in (2, 8):
            fs[k][idx] = V.cand_scores(V.query_vec(fr[idx], "A0", k), ta, tb, ub, idx, c5[idx])
    return fs, c5


def cmd_analyze(args) -> int:
    data = V.load_data()
    V.gate(data)
    old_k8 = check_old_antgpt()
    qwen = load_qwen(data)
    labels, correct, gammas, ceiling = recognised_labels(data, qwen)
    # 9b gate: F (shortlist top) reproduces the stored raw text match at k = 2, 8 (normalised == raw here)
    stored = json.loads(STORED_REC.read_text())["what_recognition_top1_in_video"]
    b = {int(r["k"]): float(r["value"]) for r in csv.DictReader((OUT_B / "summary.csv").open())
         if r["measure"] == "text_match_norm"}
    bad = {k: (correct[("F", k)].mean(), b[k]) for k in (2, 8) if abs(correct[("F", k)].mean() - b[k]) > 1e-9}
    print(f"GATE F shortlist top-1 reproduces 9b text match (stored raw {stored['2']:.4f}/{stored['8']:.4f}): "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
    if bad:
        raise SystemExit("gate failed: shortlist top-1 does not reproduce 9b; stopping")
    vids = data.vids
    rec_rows = []
    for k in (2, 8):
        rec_rows.append(acc_row("ceiling_recall@5", k, ceiling[k], vids))
        for name in ("F", "rec_qwen", "rec_fuse"):
            row = acc_row(name, k, correct[(name, k)], vids)
            if name != "F":
                d = LB.boot(correct[(name, k)] - correct[("F", k)], vids)
                row.update({"minus_F": d[0], "minus_F_lo": d[1], "minus_F_hi": d[2]})
            rec_rows.append(row)
    r1 = next(r for r in rec_rows if r["measure"] == "rec_fuse" and r["k"] == 8)
    r1_pass = bool(r1["minus_F_lo"] > 0)
    q8 = next(r for r in rec_rows if r["measure"] == "rec_qwen" and r["k"] == 8)["value"]
    a1_variant = "qwen_k8" if q8 > r1["value"] else "fuse_k8"
    # AntGPT
    ours = {}
    with gzip.open(ROOT / "results/lit_baselines_next/per_query.csv.gz", "rt", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "ours" and int(r["k"]) in (2, 8):
                ours[(r["sample_id"], int(r["k"]))] = float(r["hit5"])
    texts, c5 = LB.next_setup(data)
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    variants = [("old", "rec_k2"), ("old", "rec_k8"), ("old", "true")] + [("new", v) for v in NEW_VARIANTS]
    scores, ant_rows = {}, []
    for src, variant in variants:
        d = ROOT / "results/lit_baselines_next/antgpt_scores" if src == "old" else OUT_C / "antgpt_scores"
        recs = [json.loads(line) for line in (d / f"{variant}.jsonl").open() if line.strip()]
        for r in recs:
            assert r["candidate_sample_ids"] == [data.sample_ids[j] for j in c5[pos[r["sample_id"]]]], "candidates"
        scores[variant] = {r["sample_id"]: np.asarray(r["score"]) for r in recs}
        ks = (2, 8) if variant == "true" else (int(variant.split("_k")[1]),)
        for k in ks:
            v = np.array([r["video_uid"] for r in recs])
            hits = {"ours": np.array([ours[(r["sample_id"], k)] for r in recs]),
                    f"antgpt_{variant}": np.array([LB.fractional_credit(np.asarray(r["score"])) for r in recs])}
            ant_rows += LB.compare_rows("next_antgpt", {"k": k, "variant": variant, "history": src}, hits, v,
                                        np.ones(len(recs), bool))
    a1 = next(r for r in ant_rows if r["method"] == f"antgpt_{a1_variant}")
    a1_pass = bool(a1["ours_minus_hi"] < 0)  # AntGPT - F clearly > 0 <=> ours - AntGPT CI entirely < 0
    # fusion z(F) + gamma z(LLM), as cmd_fusion
    fs, c5f = f_next_scores(data, args.device)
    v5 = (c5f >= 0).all(axis=1)
    fus_rows, fus_gamma = [], {}
    for variant in NEW_VARIANTS + ("rec_k2", "rec_k8"):
        k = int(variant.split("_k")[1])
        ms = np.full((data.n, 5), np.nan)
        for s, sc in scores[variant].items():
            ms[pos[s]] = sc
        ok = v5 & ~np.isnan(ms).any(axis=1)
        fused, chosen = LB.nested_gamma(LB.zscore(fs[k][ok]), LB.zscore(ms[ok]), data.fold_of[ok], data.folds)
        fus_gamma[f"{variant}|k{k}"] = chosen
        hits = {"ours": LB.strict_top1(fs[k][ok]), f"F+antgpt_{variant}": fused}
        fus_rows += LB.compare_rows("next_fusion", {"k": k, "variant": variant}, hits, data.vids[ok],
                                    np.ones(int(ok.sum()), bool))
    write_csv(OUT_C / "recognition_summary.csv", rec_rows)
    write_csv(OUT_C / "antgpt_summary.csv", ant_rows)
    write_csv(OUT_C / "fusion_summary.csv", fus_rows)
    verdicts = {
        "R1": {"pass": r1_pass, "rule": "rec_fuse - F text-match recognition at k=8, CI lower bound > 0",
               "delta": r1["minus_F"], "ci95": [r1["minus_F_lo"], r1["minus_F_hi"]]},
        "A1": {"pass": a1_pass, "variant": a1_variant,
               "rule": "AntGPT(variant) - F on NEXT at k=8, CI lower bound > 0",
               "delta": -a1["ours_minus"], "ci95": [-a1["ours_minus_hi"], -a1["ours_minus_lo"]]},
    }
    (OUT_C / "config.json").write_text(json.dumps({
        "prereg": str(PREREG.relative_to(ROOT)), "model": LB.LLM_ID, "question": QUESTION, "shortlist": SHORTLIST,
        "gammas": list(GAMMAS), "recognition_gamma_by_fold": {str(k): v for k, v in gammas.items()},
        "fusion_gamma_by_fold": fus_gamma, "old_antgpt_rec_k8": old_k8,
        "antgpt_accuracy": "fractional tie credit (as antgpt-analyze)", "F_next": "strict ours hits, per_query.csv.gz",
        "B": LB.B, "verdicts": verdicts, "test_split_read": False}, indent=2))
    for r in rec_rows:
        extra = f"  minus F {100 * r['minus_F']:+.1f} [{100 * r['minus_F_lo']:+.1f}; {100 * r['minus_F_hi']:+.1f}]" \
            if "minus_F" in r else ""
        print(f"recognition k={r['k']} {r['measure']:<17} {100 * r['value']:.1f} [{100 * r['lo']:.1f}; "
              f"{100 * r['hi']:.1f}]{extra}")
    LB.print_rows(ant_rows)
    LB.print_rows(fus_rows)
    print(json.dumps(verdicts, indent=2), flush=True)
    return 0


# ======================================================================

def cmd_selftest(args) -> int:
    import test_stage9

    return test_stage9.main()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["selftest", "recognition", "qwen", "antgpt", "analyze"])
    p.add_argument("--device", default="cpu", help="text encoder (recognition/analyze) or text LLM (antgpt)")
    p.add_argument("--probe", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.cmd != "selftest" and not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}")
    if args.cmd == "antgpt" and args.device == "cpu":
        args.device = "cuda"
    return {"selftest": cmd_selftest, "recognition": cmd_recognition, "qwen": cmd_qwen, "antgpt": cmd_antgpt,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
