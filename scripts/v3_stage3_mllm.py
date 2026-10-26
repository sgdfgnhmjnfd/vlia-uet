#!/usr/bin/env python3
"""prereg v3 Stage 3: Qwen2.5-VL multiple-choice goal baseline + nested fusion with F.

Scoring (GPU, resumable; one JSON per (sample, k[, control]) under --score-dir):
  k prefix frames (longer side 448 px) as images in one user message, the
  options "A. <why>" .. "E. <why>", answer = log_softmax at the first answer
  position over the letters A..E. Two option orders (canonical build_kway order
  and reversed); per-candidate score = mean letter log-prob over the orders.
  --probe N scores the first N sorted queries to measure throughput.
Analysis (--analyze, CPU): V3 (5-way lift CI lower bound > 1 at k=2 and k=8,
fractional tie credit), paired delta vs F, control, nested gamma fusion (V4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PREFIXES = (2, 8)
CONTROL_K = 8
LETTERS = ("A", "B", "C", "D", "E")
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0)
QUESTION = ("These are the first frames of a first-person video. What is the person's goal in this step? "
            "Options: {options} Answer with one letter.")


def fractional_credit(scores: np.ndarray) -> float:
    top = scores.max()
    tied = np.flatnonzero(scores == top)
    return float(1.0 / len(tied)) if 0 in tied else 0.0


def resize448(frame: np.ndarray):
    from PIL import Image

    img = Image.fromarray(frame)
    w, h = img.size
    s = 448 / max(w, h)
    return img.resize((max(28, round(w * s)), max(28, round(h * s))), Image.BICUBIC)


class MLLM:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, token=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, token=False, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        tok = self.processor.tokenizer
        self.letter_ids = []
        for L in LETTERS:
            ids = tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, (L, ids)
            self.letter_ids.append(ids[0])

    def letter_logprobs(self, images, texts):
        """images: PIL list; texts: candidate WHY strings in option order A..E."""
        torch = self.torch
        options = " ".join(f"{L}. {t}" for L, t in zip(LETTERS, texts))
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": QUESTION.format(options=options)}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return [float(lp[i]) for i in self.letter_ids], int(inputs["input_ids"].shape[1])

    def score(self, frames, k, texts):
        images = list(frames[:k])  # already resized to 448 px by load_small
        fwd, n_tok = self.letter_logprobs(images, texts)
        rev_lp, _ = self.letter_logprobs(images, texts[::-1])
        rev = rev_lp[::-1]  # back to canonical candidate order
        return {"lp_order1": fwd, "lp_order2_canonical": rev,
                "score": [(a + b) / 2 for a, b in zip(fwd, rev)], "n_tokens": n_tok}


def rows_and_sets():
    rows = [json.loads(line) for line in (ROOT / "results/hoi_detection_v1/manifest.jsonl").open() if line.strip()]
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("disallowed split")
    sid = [r["sample_id"] for r in rows]
    vids = [r["video_uid"] for r in rows]
    texts = [r["why"] for r in rows]
    cands = CV.build_kway(vids, texts, sid, 5)
    return rows, sid, vids, texts, cands


def donor_of(i, sid, rows):
    pool = [j for j in range(len(rows)) if rows[j]["scene"] != rows[i]["scene"] and rows[j]["video_uid"] != rows[i]["video_uid"]]
    rng = np.random.RandomState(zlib.crc32(f"{sid[i]}|control".encode()) & 0xFFFFFFFF)
    return pool[rng.randint(len(pool))]


def run(args) -> int:
    from hoi_utils import sample_exact_video_frames

    def load_small(path):
        # resize in the worker: keeps 8 x 448 px PIL images instead of full-res
        # frames in memory (first full run was OOM-killed at ~14 GB RSS)
        return [resize448(f) for f in sample_exact_video_frames(path)]

    rows, sid, vids, texts, cands = rows_and_sets()
    order = sorted(range(len(rows)), key=lambda i: sid[i])
    run_set = order[:: args.stride]
    if args.probe:
        run_set = order[: args.probe]
    control = set(run_set[::8]) if not args.probe else set()
    out = Path(args.score_dir) / ("probe" if args.probe else "scores")
    out.mkdir(parents=True, exist_ok=True)

    def fname(i, k, ctrl=False):
        return out / f"{sid[i].replace('/', '__')}__k{k}{'__control' if ctrl else ''}.json"

    def todo(i):
        j = [(k, False) for k in PREFIXES if not fname(i, k).exists()]
        if i in control and not fname(i, CONTROL_K, True).exists():
            j.append((CONTROL_K, True))
        return j

    pending = [i for i in run_set if todo(i)]
    print(f"{len(run_set)} queries (stride {args.stride}); {len(pending)} to score; controls {len(control)}", flush=True)
    m = MLLM()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {}
        need = []
        for i in pending:
            need.append(i)
            if i in control:
                need.append(donor_of(i, sid, rows))
        for j in need[:6]:
            futs.setdefault(j, pool.submit(load_small, rows[j]["video_path"]))
        cursor = 6
        for n_done, i in enumerate(pending, 1):
            while cursor < min(len(need), 2 * n_done + 6):
                futs.setdefault(need[cursor], pool.submit(load_small, rows[need[cursor]]["video_path"]))
                cursor += 1
            futs.setdefault(i, pool.submit(load_small, rows[i]["video_path"]))
            frames = futs.pop(i).result()
            ci = [int(c) for c in cands[i]]
            assert min(ci) >= 0
            ctexts = [texts[c] for c in ci]
            for k, ctrl in todo(i):
                rec = {"sample_id": sid[i], "video_uid": vids[i], "k": k, "control": ctrl,
                       "candidate_sample_ids": [sid[c] for c in ci]}
                if ctrl:
                    d = donor_of(i, sid, rows)
                    futs.setdefault(d, pool.submit(load_small, rows[d]["video_path"]))
                    df = futs.pop(d).result()
                    rec["donor_sample_id"] = sid[d]
                    rec.update(m.score(df, k, ctexts))
                else:
                    rec.update(m.score(frames, k, ctexts))
                tmp = fname(i, k, ctrl).with_suffix(".tmp")
                tmp.write_text(json.dumps(rec))
                tmp.replace(fname(i, k, ctrl))
            upcoming = set(need[need.index(i) + 1:]) if i in need else set()
            for j in [j for j in futs if j not in upcoming]:
                futs.pop(j)
            if n_done % 10 == 0 or n_done == len(pending):
                el = time.time() - t0
                print(f"queries {n_done}/{len(pending)} elapsed {el:.0f}s s/query {el / n_done:.2f} "
                      f"eta {el / n_done * (len(pending) - n_done):.0f}s", flush=True)
    el = time.time() - t0
    info = {"model": MODEL_ID, "n_queries": len(run_set), "stride": args.stride, "probe": args.probe,
            "seconds_per_query": el / max(1, len(pending)), "eta_full_hours": el / max(1, len(pending)) * len(rows) / 3600}
    (Path(args.score_dir) / ("probe_info.json" if args.probe else "run_info.json")).write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    return 0


def zscore(s):
    s = np.asarray(s, dtype=float)
    return (s - s.mean(axis=-1, keepdims=True)) / np.maximum(s.std(axis=-1, keepdims=True), 1e-6)


def analyze(args) -> int:
    import run_ladder_v1 as L
    import v3_common as V

    out = ROOT / "results/v3_stage3_mllm"
    out.mkdir(parents=True, exist_ok=True)
    data = V.load_data()
    mf = V.f_gate(data)
    recs = [json.loads(p.read_text()) for p in sorted((Path(args.score_dir) / "scores").glob("*.json"))]
    real = {(r["sample_id"], r["k"]): r for r in recs if not r["control"]}
    ctrl = [r for r in recs if r["control"]]
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    # F candidate scores out of fold (for fusion)
    f_scores = {k: np.zeros((data.n, 5)) for k in PREFIXES}
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        fr, ta, tb, ub = V.prep_f(data, tr)
        for k in PREFIXES:
            f_scores[k][idx] = L.candidate_scores(fr, ta, tb, ub, idx, data.cands5[idx], "A0", k)
    evaluated = sorted({pos[s] for (s, k) in real if all((s, kk) in real for kk in PREFIXES)})
    ev = np.asarray(evaluated)
    for i in ev:
        r = real[(data.sample_ids[i], 2)]
        assert r["candidate_sample_ids"] == [data.sample_ids[c] for c in data.cands5[i]]
    summary, pairs = [], []
    m_scores = {k: np.asarray([real[(data.sample_ids[i], k)]["score"] for i in ev]) for k in PREFIXES}
    m_credit = {k: np.asarray([fractional_credit(s) for s in m_scores[k]]) for k in PREFIXES}
    v3_pref = []
    for k in PREFIXES:
        pt, lo, hi = CV.video_bootstrap(data.vids[ev], m_credit[k], np.full(len(ev), 0.2))
        summary.append({"method": "MLLM", "k": k, "n_queries": len(ev), "5way_acc": float(m_credit[k].mean()),
                        "5way_lift": pt, "5way_lift_ci_lo": lo, "5way_lift_ci_hi": hi,
                        "F_5way_acc_same_queries": float(mf[k]["kway5"][ev].mean())})
        if lo > 1:
            v3_pref.append(k)
        d = m_credit[k] - mf[k]["kway5"][ev]
        dp, dlo, dhi = CV.video_bootstrap(data.vids[ev], d, np.ones(len(ev)))
        pairs.append({"comparison": "MLLM - F", "k": k, "metric": "kway5", "delta": dp, "ci95_lo": dlo, "ci95_hi": dhi,
                      "n_queries": len(ev)})
    # control
    ctrl_rows = []
    if ctrl:
        cidx = np.asarray([pos[r["sample_id"]] for r in ctrl])
        cc = np.asarray([fractional_credit(np.asarray(r["score"])) for r in ctrl])
        rc = np.asarray([fractional_credit(np.asarray(real[(r["sample_id"], CONTROL_K)]["score"])) for r in ctrl])
        pt, lo, hi = CV.video_bootstrap(data.vids[cidx], cc, np.full(len(cc), 0.2))
        ctrl_rows.append({"k": CONTROL_K, "n_control": len(cc), "real_5way_acc": float(rc.mean()),
                          "control_5way_acc": float(cc.mean()), "control_lift": pt, "control_lift_ci_lo": lo,
                          "control_lift_ci_hi": hi})
    # nested gamma fusion
    in_ev = np.zeros(data.n, bool)
    in_ev[ev] = True
    zf = {k: zscore(f_scores[k]) for k in PREFIXES}
    zm = {k: np.zeros((data.n, 5)) for k in PREFIXES}
    for k in PREFIXES:
        zm[k][ev] = zscore(m_scores[k])
    fused_hits = {k: np.zeros(data.n) for k in PREFIXES}
    gamma_rows = []
    for f in data.folds:
        idx, tr = data.fold_idx(f)
        tr = tr[in_ev[tr]]
        idx = idx[in_ev[idx]]
        totals = []
        for g in GAMMAS:
            totals.append(sum(int(L.strict_hits(zf[k][tr] + g * zm[k][tr], data.valid5[tr]).sum()) for k in PREFIXES))
        gi = int(np.argmax(totals))
        gamma_rows.append({"fold": f, "gamma": GAMMAS[gi], **{f"train_hits_gamma_{g}": t for g, t in zip(GAMMAS, totals)}})
        for k in PREFIXES:
            fused_hits[k][idx] = L.strict_hits(zf[k][idx] + GAMMAS[gi] * zm[k][idx], data.valid5[idx])
    v4_pref = []
    for k in PREFIXES:
        d = fused_hits[k][ev] - mf[k]["kway5"][ev]
        dp, dlo, dhi = CV.video_bootstrap(data.vids[ev], d, np.ones(len(ev)))
        pairs.append({"comparison": "fusion - F", "k": k, "metric": "kway5", "delta": dp, "ci95_lo": dlo, "ci95_hi": dhi,
                      "n_queries": len(ev)})
        summary.append({"method": "fusion(F + gamma*MLLM)", "k": k, "n_queries": len(ev),
                        "5way_acc": float(fused_hits[k][ev].mean()), "5way_lift": "", "5way_lift_ci_lo": "",
                        "5way_lift_ci_hi": "", "F_5way_acc_same_queries": float(mf[k]["kway5"][ev].mean())})
        if dlo > 0:
            v4_pref.append(k)
    verdicts = {
        "V3": {"pass": v3_pref == list(PREFIXES), "prefixes_lift_ci_lo_gt1": v3_pref,
               "rule": "MLLM 5-way lift CI lower bound > 1 at both k (fractional tie credit)"},
        "V4": {"pass": v4_pref == list(PREFIXES), "prefixes_5way_delta_ci_lo_gt0": v4_pref,
               "rule": "nested fusion: 5-way delta vs F CI lower bound > 0 at both k"},
    }
    V.write_csv(out / "summary.csv", summary)
    V.write_csv(out / "paired_vs_F.csv", pairs)
    V.write_csv(out / "fusion_gamma_by_fold.csv", gamma_rows)
    if ctrl_rows:
        V.write_csv(out / "control.csv", ctrl_rows)
    run_info = json.loads((Path(args.score_dir) / "run_info.json").read_text()) if (Path(args.score_dir) / "run_info.json").exists() else {}
    probe_info = json.loads((Path(args.score_dir) / "probe_info.json").read_text()) if (Path(args.score_dir) / "probe_info.json").exists() else {}
    lines = [f"# v3 Stage 3: MLLM multiple-choice baseline ({MODEL_ID}) + fusion", "",
             f"Evaluated queries: {len(ev)} (stride {run_info.get('stride', '?')}). Video bootstrap B=10000, seed 0.", "",
             "| method | k | n | 5-way acc | lift [95% CI] | F 5-way (same queries) |", "|---|---:|---:|---:|---|---:|"]
    for r in summary:
        lift = f"{r['5way_lift']:.3f} [{r['5way_lift_ci_lo']:.3f}, {r['5way_lift_ci_hi']:.3f}]" if r["5way_lift"] != "" else "—"
        lines.append(f"| {r['method']} | {r['k']} | {r['n_queries']} | {r['5way_acc']:.4f} | {lift} | {r['F_5way_acc_same_queries']:.4f} |")
    lines += ["", "| comparison | k | 5-way delta [95% CI] |", "|---|---:|---|"]
    for r in pairs:
        lines.append(f"| {r['comparison']} | {r['k']} | {r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |")
    if ctrl_rows:
        r = ctrl_rows[0]
        lines += ["", f"Control (k=8, n={r['n_control']}): real {r['real_5way_acc']:.4f}, control {r['control_5way_acc']:.4f}, "
                      f"control lift {r['control_lift']:.3f} [{r['control_lift_ci_lo']:.3f}, {r['control_lift_ci_hi']:.3f}]"]
    lines += ["", "Selected gamma per fold: " + ", ".join(f"{r['fold']}:{r['gamma']}" for r in gamma_rows),
              "", "```", json.dumps(verdicts, indent=2), "```", ""]
    (out / "summary.md").write_text("\n".join(lines))
    (out / "config.json").write_text(json.dumps({"model": MODEL_ID, "why_not_7b": "7B bf16 weights 16.58 GB > 16 GB GPU",
                                                 "image_long_side": 448, "orders": ["canonical", "reversed"],
                                                 "run_info": run_info, "probe_info": probe_info, "gammas": GAMMAS,
                                                 "verdicts": verdicts, **V.meta()}, indent=2))
    print(json.dumps(verdicts, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-dir", default="/media/dhqg/d3/vlia_outputs/v3_mllm")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    return analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
