#!/usr/bin/env python3
"""Baselines from the anticipation literature, on the EgoIntent train/val cache.

Rules are fixed in `results/prereg_literature_baselines.md` (written before any
number). "Ours" is the frozen pipeline F (siglip-so400m | mean | C2 | T1); the
locked test split is never read.

Subcommands
  selftest   checks on synthetic numbers
  horizon    Stage 7 queries (WHAT/WHY of the step containing t+H) scored by
             ours vs three baselines common in action anticipation papers
             (EPIC-KITCHENS / RULSTM style):
               blind      no video: candidate closest to the mean label of the
                          OTHER videos (a text-only prior)
               copy_rec   recognise the current step (context frames vs the
                          video's labels), predict the candidate whose text is
                          closest to it ("the person keeps doing this")
               copy_true  same with the TRUE current label (upper bound of any
                          recognise-then-copy method)
             -> results/lit_baselines_horizon/   (CPU, minutes)
  next       NEXT-step task of results/v3_next_step (same 5 candidates), ours
             vs blind / copy_rec / copy_true; also saves the recognised WHAT of
             every step for `antgpt`
             -> results/lit_baselines_next/      (CPU, minutes)
  antgpt     AntGPT-style two-stage method (Zhao et al., ICLR 2024): the WHATs
             of the last steps (true labels, or recognised from video) are
             given as text to an LLM (Qwen2.5-VL-3B-Instruct, text only), which
             picks the next step among the same 5 candidates. Resumable.
             (GPU ~15 min; CPU possible but hours)
  antgpt-analyze   paired comparison with ours -> results/lit_baselines_next/antgpt_*.csv
  fusion     z(F) + gamma * z(LLM) on the NEXT candidates, gamma chosen per outer
             fold on the other folds (as v3 Stage 3) -> results/lit_baselines_next/fusion_*

Typical run on the lab machine while the GPU trains:
  export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
  $PY scripts/run_literature_baselines.py selftest
  $PY scripts/run_literature_baselines.py horizon
  $PY scripts/run_literature_baselines.py next
After training (GPU free):
  $PY scripts/run_literature_baselines.py antgpt --device cuda
  $PY scripts/run_literature_baselines.py antgpt-analyze
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402
import run_ladder_v1 as L  # noqa: E402
import run_scene_cv as CV  # noqa: E402
import run_v3_supplementary as V  # noqa: E402

PREREG = ROOT / "results/prereg_literature_baselines.md"
B = 10000
METHODS = ("ours", "blind", "copy_rec", "copy_true")
# Stage 7 WHAT/C2 5-way accuracy (thesis Table 4.15), used as a gate
STAGE7_WHAT_C2 = {2: 0.380, 8: 0.335, 16: 0.291, 24: 0.277, 32: 0.273, 48: 0.262, 64: 0.240}
NEXT_PREFIXES = (2, 4, 6, 8)
ANTGPT_PREFIXES = (2, 8)
HISTORY = 5  # previous steps shown to the LLM (plus the current one)
LLM_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LETTERS = ("A", "B", "C", "D", "E")
QUESTION = ("A person is doing a task with their hands, filmed from their own head camera. "
            "The steps so far, oldest first, are:\n{history}\n"
            "Which step will the person do next?\nOptions: {options}\nAnswer with one letter.")


# ======================================================================
# scoring helpers (pure numpy)
# ======================================================================

def strict_top1(s: np.ndarray) -> np.ndarray:
    """s (n, K), column 0 = target. Ties count as wrong."""
    return (s[:, :1] > s[:, 1:]).all(axis=1).astype(float)


def blind_prior(emb: np.ndarray, label_ids_by_video: dict) -> dict:
    """Per video: unit mean of the label embeddings of all OTHER videos."""
    sums = {v: emb[sorted(ids)].sum(axis=0) for v, ids in label_ids_by_video.items()}
    counts = {v: len(ids) for v, ids in label_ids_by_video.items()}
    total, n = sum(sums.values()), sum(counts.values())
    return {v: L.normalize((total - sums[v]) / (n - counts[v])) for v in sums}


def fractional_credit(scores: np.ndarray) -> float:
    top = scores.max()
    tied = np.flatnonzero(scores == top)
    return float(1.0 / len(tied)) if 0 in tied else 0.0


def boot(values: np.ndarray, vids: np.ndarray):
    return AH.video_bootstrap(np.asarray(values, float), np.asarray(vids), B)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def compare_rows(task: str, key: dict, hits: dict, vids: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Accuracy of every method and ours-minus-method paired differences."""
    out = []
    for m, h in hits.items():
        ok = mask & ~np.isnan(h) & ~np.isnan(hits["ours"])
        acc = boot(h[ok], vids[ok])
        row = {"task": task, **key, "method": m, "n_queries": int(ok.sum()),
               "acc5": acc[0], "acc5_lo": acc[1], "acc5_hi": acc[2]}
        if m != "ours":
            d = boot(hits["ours"][ok] - h[ok], vids[ok])
            row.update({"ours_minus": d[0], "ours_minus_lo": d[1], "ours_minus_hi": d[2],
                        "ours_better": d[1] > 0, "ours_worse": d[2] < 0})
        out.append(row)
    return out


def print_rows(rows):
    for r in rows:
        extra = ""
        if "ours_minus" in r:
            extra = (f"  ours-minus {100 * r['ours_minus']:+.1f} "
                     f"[{100 * r['ours_minus_lo']:+.1f}; {100 * r['ours_minus_hi']:+.1f}]")
        key = " ".join(f"{k}={r[k]}" for k in ("label", "H", "subset", "k", "variant") if k in r)
        print(f"{r['task']} {key} {r['method']:<10} n={r['n_queries']:>6} "
              f"acc5 {100 * r['acc5']:.1f} [{100 * r['acc5_lo']:.1f}; {100 * r['acc5_hi']:.1f}]{extra}", flush=True)


def out_dir(name: str, overwrite: bool, marker: str = "summary.csv") -> Path:
    out = ROOT / "results" / name
    if (out / marker).exists() and not overwrite:
        raise FileExistsError(f"{out}/{marker} exists (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ======================================================================
# horizon
# ======================================================================

def cmd_horizon(args) -> int:
    out = out_dir("lit_baselines_horizon", args.overwrite)
    rows = AH.read_jsonl(L.MANIFEST)
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("manifest must contain train/val rows only")
    cache = L.LADDER_ROOT / V.ENC
    ids = json.loads((cache / "sample_ids.json").read_text())
    pos = {s: i for i, s in enumerate(ids)}
    frames = np.load(cache / "frame_emb.npy", mmap_mode="r")
    frames = L.normalize(np.asarray(frames[[pos[r["sample_id"]] for r in rows]], dtype=np.float32))
    timelines = AH.build_timelines(rows, frames)
    mu_frame = frames.reshape(-1, frames.shape[-1]).mean(axis=0)
    labels_keys = ("what", "why")
    vocab = sorted({r[k] for r in rows for k in labels_keys})
    emb = AH.encode_texts(V.ENCODER_ID, vocab, device=args.device)
    tix = {t: i for i, t in enumerate(vocab)}
    horizons = sorted(STAGE7_WHAT_C2) if args.quick else list(AH.DEFAULT_HORIZONS)
    summary = []
    for key in labels_keys:
        queries, _ = AH.make_queries(timelines, 8, horizons, rows, key, seed=0)
        qv, _ = AH.query_vectors(timelines, queries, 8, "C2", mu_frame)
        ours = AH.score_queries(qv, queries, tix, emb)["hit5"]
        by_video = {}
        for r in rows:
            by_video.setdefault(r["video_uid"], set()).add(tix[r[key]])
        prior = blind_prior(emb, by_video)
        n = len(queries)
        hits = {"ours": ours, "blind": np.full(n, np.nan), "copy_rec": np.full(n, np.nan),
                "copy_true": np.full(n, np.nan)}
        for j, q in enumerate(queries):
            if q["cands5"] is None:
                continue
            c = emb[[tix[t] for t in q["cands5"]]]
            tl = timelines[q["video_uid"]]
            true_now = rows[tl["rows"][q["anchor_step"]]][key]
            g = q["gallery"]
            rec_now = g[int(np.argmax(emb[[tix[t] for t in g]] @ qv[j]))]
            hits["blind"][j] = strict_top1((c @ prior[q["video_uid"]])[None])[0]
            hits["copy_rec"][j] = strict_top1((c @ emb[tix[rec_now]])[None])[0]
            hits["copy_true"][j] = strict_top1((c @ emb[tix[true_now]])[None])[0]
        vids = np.array([q["video_uid"] for q in queries])
        hs = np.array([q["H"] for q in queries])
        same = np.array([q["same_step"] for q in queries])
        for h in horizons:
            for subset, mask in (("all", hs == h), ("changed_step", (hs == h) & ~same)):
                if mask.any():
                    summary += compare_rows("horizon", {"label": key.upper(), "H": h, "subset": subset},
                                            hits, vids, mask)
        if key == "what":
            got = {h: float(np.nanmean(ours[hs == h])) for h in STAGE7_WHAT_C2}
            bad = {h: (round(got[h], 4), e) for h, e in STAGE7_WHAT_C2.items() if abs(got[h] - e) > 6e-4}
            print(f"GATE ours WHAT/C2 vs Stage 7: {'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
            if bad and not args.no_gate:
                raise SystemExit("gate failed: ours does not reproduce Stage 7")
    write_csv(out / "summary.csv", summary)
    (out / "config.json").write_text(json.dumps({"horizons": horizons, "context": 8, "centering": "C2",
                                                 "encoder": V.ENCODER_ID, "B": B, "test_split_read": False}, indent=2))
    print_rows([r for r in summary if r["subset"] == "all"])
    print(f"saved {out}", flush=True)
    return 0


# ======================================================================
# next
# ======================================================================

def next_setup(data):
    nxt = V.load_next_labels(data.rows)
    texts = [nxt[s] for s in data.sample_ids]
    c5 = CV.build_kway(data.vids.tolist(), texts, data.sample_ids, 5)
    return texts, c5


def ordered_members(data):
    """video -> row indices in step order."""
    out = {}
    for v in np.unique(data.vids):
        out[v] = sorted(np.flatnonzero(data.vids == v), key=lambda i: data.steps[i])
    return out


def cmd_next(args) -> int:
    out = out_dir("lit_baselines_next", args.overwrite)
    data = V.load_data()
    V.gate(data)
    texts, c5 = next_setup(data)
    v5 = (c5 >= 0).all(axis=1)
    whats = [r["what"] for r in data.rows]
    vocab = sorted(set(texts) | set(whats))
    emb = AH.encode_texts(V.ENCODER_ID, vocab, device=args.device)
    tix = {t: i for i, t in enumerate(vocab)}
    next_raw = emb[[tix[t] for t in texts]]
    what_raw = emb[[tix[t] for t in whats]]
    members = ordered_members(data)
    by_video = {v: {tix[texts[i]] for i in m} for v, m in members.items()}
    prior = blind_prior(emb, by_video)
    frames = data.frames[V.ENC]
    ctx, has = data.ctx[V.ENC]
    n = data.n
    hits = {k: {m: np.full(n, np.nan) for m in METHODS} for k in NEXT_PREFIXES}
    rec_row = {k: np.full(n, -1) for k in NEXT_PREFIXES}
    for fold in data.folds:
        idx, train_idx = data.fold_idx(fold)
        fr, ta, tb, ub = L.processed(frames, next_raw, "C2", train_idx, ctx, has)
        _, wa, wb, _ = L.processed(frames, what_raw, "C2", train_idx, ctx, has)
        for k in NEXT_PREFIXES:
            q = V.query_vec(fr[idx], "A0", k)
            hits[k]["ours"][idx] = L.strict_hits(V.cand_scores(q, ta, tb, ub, idx, c5[idx]), v5[idx])
            for li, i in enumerate(idx):
                g = np.asarray(members[data.vids[i]])
                table = wb[g] if ub[i] else wa[g]
                rec_row[k][i] = g[int(np.argmax(table @ q[li]))]
    for k in NEXT_PREFIXES:
        for i in range(n):
            if not v5[i]:
                hits[k]["ours"][i] = np.nan
                continue
            c = emb[[tix[texts[j]] for j in c5[i]]]
            hits[k]["blind"][i] = strict_top1((c @ prior[data.vids[i]])[None])[0]
            hits[k]["copy_rec"][i] = strict_top1((c @ what_raw[rec_row[k][i]])[None])[0]
            hits[k]["copy_true"][i] = strict_top1((c @ what_raw[i])[None])[0]
    # gate against the earlier NEXT run when it is present
    ref = ROOT / "results/v3_next_step/summary.csv"
    if ref.exists():
        exp = {int(r["k"]): float(r["5way_acc"]) for r in csv.DictReader(ref.open())
               if r["label"] == "NEXT" and r["centering"] == "C2"}
        got = {k: float(np.nanmean(hits[k]["ours"])) for k in NEXT_PREFIXES}
        bad = {k: (got[k], exp[k]) for k in exp if abs(got[k] - exp[k]) > 1e-9}
        print(f"GATE ours NEXT vs v3_next_step: {'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
        if bad and not args.no_gate:
            raise SystemExit("gate failed: ours does not reproduce results/v3_next_step")
    else:
        print("note: results/v3_next_step/summary.csv not found, NEXT gate skipped", flush=True)
    summary = []
    for k in NEXT_PREFIXES:
        summary += compare_rows("next", {"k": k}, hits[k], data.vids, v5)
    rec_acc = {k: float(np.mean([whats[rec_row[k][i]] == whats[i] for i in range(n)])) for k in NEXT_PREFIXES}
    write_csv(out / "summary.csv", summary)
    with gzip.open(out / "per_query.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["sample_id", "video_uid", "k", "method", "hit5"])
        for k in NEXT_PREFIXES:
            for m in METHODS:
                for i in range(n):
                    w.writerow([data.sample_ids[i], data.vids[i], k, m, hits[k][m][i]])
    (out / "recognized_what.json").write_text(json.dumps(
        {data.sample_ids[i]: {str(k): data.sample_ids[rec_row[k][i]] for k in NEXT_PREFIXES} for i in range(n)}))
    (out / "config.json").write_text(json.dumps({
        "encoder": V.ENCODER_ID, "centering": "C2", "B": B, "test_split_read": False,
        "what_recognition_top1_in_video": rec_acc}, indent=2))
    print_rows(summary)
    print("WHAT recognition top-1 within video:", {k: round(v, 3) for k, v in rec_acc.items()}, flush=True)
    print(f"saved {out}", flush=True)
    return 0


# ======================================================================
# AntGPT-style LLM
# ======================================================================

def history_texts(data, members, i, variant, rec, pos):
    """WHAT texts of up to HISTORY previous steps plus the current one."""
    m = members[data.vids[i]]
    p = m.index(i)
    steps = m[max(0, p - HISTORY): p + 1]
    if variant == "true":
        return [data.rows[j]["what"] for j in steps]
    k = variant.split("_k")[1]
    # previous steps fully seen (8 frames); the current one with k frames
    out = [data.rows[pos[rec[data.sample_ids[j]]["8"]]]["what"] for j in steps[:-1]]
    out.append(data.rows[pos[rec[data.sample_ids[i]][k]]]["what"])
    return out


def build_prompt(history, options):
    hist = "\n".join(f"{n}. {t}" + (" (current step)" if n == len(history) else "")
                     for n, t in enumerate(history, 1))
    opts = " ".join(f"{a}. {t}" for a, t in zip(LETTERS, options))
    return QUESTION.format(history=hist, options=opts)


class LLM:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(LLM_ID, token=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            LLM_ID, token=False, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        tok = self.processor.tokenizer
        self.letter_ids = []
        for a in LETTERS:
            t = tok.encode(a, add_special_tokens=False)
            assert len(t) == 1, (a, t)
            self.letter_ids.append(t[0])

    def letter_logprobs(self, text):
        torch = self.torch
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}], add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return [float(lp[i]) for i in self.letter_ids]

    def score(self, history, options):
        fwd = self.letter_logprobs(build_prompt(history, options))
        rev = self.letter_logprobs(build_prompt(history, options[::-1]))[::-1]
        return [(a + b) / 2 for a, b in zip(fwd, rev)]


def antgpt_variants():
    return ["true"] + [f"rec_k{k}" for k in ANTGPT_PREFIXES]


def cmd_antgpt(args) -> int:
    base = ROOT / "results/lit_baselines_next"
    rec_path = base / "recognized_what.json"
    if not rec_path.exists():
        raise SystemExit("run `next` first (needs recognized_what.json)")
    rec = json.loads(rec_path.read_text())
    data = V.load_data()
    texts, c5 = next_setup(data)
    members = ordered_members(data)
    pos = {s: j for j, s in enumerate(data.sample_ids)}
    order = [i for i in sorted(range(data.n), key=lambda i: data.sample_ids[i]) if (c5[i] >= 0).all()]
    if args.probe:
        order = order[: args.probe]
    score_dir = base / ("antgpt_probe" if args.probe else "antgpt_scores")
    score_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.device)
    t0, done = time.time(), 0
    for variant in antgpt_variants():
        path = score_dir / f"{variant}.jsonl"
        seen = set()
        if path.exists():
            seen = {json.loads(line)["sample_id"] for line in path.open() if line.strip()}
        todo = [i for i in order if data.sample_ids[i] not in seen]
        print(f"[{variant}] {len(order)} queries, {len(todo)} to score", flush=True)
        with path.open("a", encoding="utf-8") as handle:
            for n_done, i in enumerate(todo, 1):
                hist = history_texts(data, members, i, variant, rec, pos)
                opts = [texts[j] for j in c5[i]]
                rec_line = {"sample_id": data.sample_ids[i], "video_uid": data.vids[i], "variant": variant,
                            "history": hist, "candidate_sample_ids": [data.sample_ids[j] for j in c5[i]],
                            "score": llm.score(hist, opts)}
                handle.write(json.dumps(rec_line) + "\n")
                handle.flush()
                done += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"  {n_done}/{len(todo)} elapsed {el:.0f}s s/query {el / done:.2f}", flush=True)
    el = time.time() - t0
    (base / ("antgpt_probe_info.json" if args.probe else "antgpt_run_info.json")).write_text(json.dumps({
        "model": LLM_ID, "device": args.device, "n_queries": len(order), "seconds_per_query": el / max(1, done),
        "eta_full_hours": el / max(1, done) * 2503 * len(antgpt_variants()) / 3600}, indent=2))
    print(f"done; {el / max(1, done):.2f} s/query", flush=True)
    return 0


def cmd_antgpt_analyze(args) -> int:
    base = ROOT / "results/lit_baselines_next"
    data = V.load_data()
    texts, c5 = next_setup(data)
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    ours = {}
    with gzip.open(base / "per_query.csv.gz", "rt", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "ours":
                ours[(r["sample_id"], int(r["k"]))] = float(r["hit5"])
    summary = []
    for variant in antgpt_variants():
        recs = [json.loads(line) for line in (base / "antgpt_scores" / f"{variant}.jsonl").open() if line.strip()]
        for r in recs:
            i = pos[r["sample_id"]]
            assert r["candidate_sample_ids"] == [data.sample_ids[j] for j in c5[i]], "candidate mismatch"
        k = 8 if variant == "true" else int(variant.split("_k")[1])
        ks = NEXT_PREFIXES if variant == "true" else (k,)
        for kk in ks:
            vids = np.array([r["video_uid"] for r in recs])
            hits = {"ours": np.array([ours[(r["sample_id"], kk)] for r in recs]),
                    f"antgpt_{variant}": np.array([fractional_credit(np.asarray(r["score"])) for r in recs])}
            summary += compare_rows("next_antgpt", {"k": kk, "variant": variant}, hits, vids,
                                    np.ones(len(recs), bool))
    write_csv(base / "antgpt_summary.csv", summary)
    print_rows(summary)
    return 0


# ======================================================================
# fusion: F + AntGPT-style LLM scores, nested gamma
# ======================================================================

GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0)  # 0 = F alone, first among ties


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)


def nested_gamma(zf, zm, fold_of, folds):
    """Per outer fold choose gamma on the other folds (max strict 5-way), apply to the held-out fold."""
    hits, chosen = np.zeros(len(zf)), {}
    for f in folds:
        test, train = fold_of == f, fold_of != f
        acc = [strict_top1(zf[train] + g * zm[train]).mean() for g in GAMMAS]
        g = GAMMAS[int(np.argmax(acc))]
        chosen[int(f)] = g
        hits[test] = strict_top1(zf[test] + g * zm[test])
    return hits, chosen


def cmd_fusion(args) -> int:
    base = ROOT / "results/lit_baselines_next"
    data = V.load_data()
    V.gate(data)
    texts, c5 = next_setup(data)
    v5 = (c5 >= 0).all(axis=1)
    vocab = sorted(set(texts))
    emb = AH.encode_texts(V.ENCODER_ID, vocab, device=args.device)
    tix = {t: i for i, t in enumerate(vocab)}
    next_raw = emb[[tix[t] for t in texts]]
    frames = data.frames[V.ENC]
    ctx, has = data.ctx[V.ENC]
    fs = {k: np.full((data.n, 5), np.nan) for k in ANTGPT_PREFIXES}
    for fold in data.folds:
        idx, train_idx = data.fold_idx(fold)
        fr, ta, tb, ub = L.processed(frames, next_raw, "C2", train_idx, ctx, has)
        for k in ANTGPT_PREFIXES:
            q = V.query_vec(fr[idx], "A0", k)
            fs[k][idx] = V.cand_scores(q, ta, tb, ub, idx, c5[idx])
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    summary, chosen_all = [], {}
    for variant in antgpt_variants():
        recs = [json.loads(line) for line in (base / "antgpt_scores" / f"{variant}.jsonl").open() if line.strip()]
        ms = np.full((data.n, 5), np.nan)
        for r in recs:
            i = pos[r["sample_id"]]
            assert r["candidate_sample_ids"] == [data.sample_ids[j] for j in c5[i]], "candidate mismatch"
            ms[i] = r["score"]
        ks = ANTGPT_PREFIXES if variant == "true" else (int(variant.split("_k")[1]),)
        for k in ks:
            ok = v5 & ~np.isnan(ms).any(axis=1)
            zf, zm = zscore(fs[k][ok]), zscore(ms[ok])
            fused, chosen = nested_gamma(zf, zm, data.fold_of[ok], data.folds)
            chosen_all[f"{variant}|k{k}"] = chosen
            f_hits = strict_top1(fs[k][ok])
            hits = {"ours": f_hits, f"F+antgpt_{variant}": fused}
            summary += compare_rows("next_fusion", {"k": k, "variant": variant}, hits, data.vids[ok],
                                    np.ones(int(ok.sum()), bool))
    write_csv(base / "fusion_summary.csv", summary)
    (base / "fusion_gamma_by_fold.json").write_text(json.dumps(chosen_all, indent=2))
    print_rows(summary)
    print("gamma by fold:", chosen_all, flush=True)
    return 0


# ======================================================================
# selftest
# ======================================================================

def cmd_selftest(args) -> int:
    assert strict_top1(np.array([[1.0, 0.5, 0.2]]))[0] == 1
    assert strict_top1(np.array([[1.0, 1.0, 0.2]]))[0] == 0, "ties must count as wrong"
    rng = np.random.default_rng(0)
    emb = L.normalize(rng.normal(size=(6, 8)))
    pr = blind_prior(emb, {"a": {0, 1}, "b": {2, 3}, "c": {4, 5}})
    assert np.allclose(pr["a"], L.normalize(emb[2:].sum(axis=0) / 4)), "prior must exclude own video"
    assert fractional_credit(np.array([0.3, 0.3, 0.1])) == 0.5
    p = build_prompt(["cut bread", "spread butter"], list("abcde"))
    assert "2. spread butter (current step)" in p and "E. e" in p
    zf = np.array([[2.0, 0, 0], [0, 2.0, 0], [1.0, 0, 3.0], [3.0, 0, 1.0]])
    h, ch = nested_gamma(zf, np.zeros_like(zf), np.array([0, 0, 1, 1]), [0, 1])
    assert set(ch.values()) == {0.0} and np.array_equal(h, strict_top1(zf)), "ties must keep F"
    zm = np.array([[0, 0, 5.0], [5.0, 0, 0], [5.0, 0, 0], [5.0, 0, 0]])
    h, ch = nested_gamma(zf, zm, np.array([0, 0, 1, 1]), [0, 1])
    assert ch[0] > 0 and h[1] == 1, "a helpful LLM score must be picked on the other fold"
    print("selftest passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["selftest", "horizon", "next", "antgpt", "antgpt-analyze", "fusion"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cpu", help="text encoder / LLM device")
    p.add_argument("--probe", type=int, default=0, help="antgpt: score only the first N queries (timing)")
    p.add_argument("--quick", action="store_true", help="horizon: only the 7 gated horizons")
    p.add_argument("--no-gate", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.cmd != "selftest" and not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}")
    return {"selftest": cmd_selftest, "horizon": cmd_horizon, "next": cmd_next, "antgpt": cmd_antgpt,
            "antgpt-analyze": cmd_antgpt_analyze, "fusion": cmd_fusion}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
