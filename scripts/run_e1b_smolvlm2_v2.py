#!/usr/bin/env python3
"""E1b v2: zero-shot K=5 goal scoring with SmolVLM2-500M-Video-Instruct (fixed scorer).

Replaces the invalid v1 scorer (see `results/e1b_smolvlm2_v1/INVALID_20261021.md`).

Scoring (per candidate WHY text c, prefix of k frames):
  * prefix ids come from ONE processor call on the prefix text with the k
    images, so the `<image>` placeholders are already expanded; candidate ids
    come from a separate tokenizer call (" " + c, no special tokens) and are
    concatenated (no boundary merging). `<end_of_utterance>` is not appended.
  * log_softmax over the logits, gathered at the candidate token positions only.
  * image pass: prefix with the k frames; null pass: the same prompt text with
    no images. The null score depends only on (candidate text, k) -- k enters
    through the question wording -- and is computed once per (text, k) and cached.
  * reported: sum and per-token mean for both passes. Primary = per-token-mean
    PMI = mean_img - mean_null; secondary: sum PMI, raw image log-probs.
  * ties: fractional credit (1 / number of candidates tied at the maximum when
    the positive is among them); no argmax tie-breaking.

Train/val manifest only. Per-(sample, k) JSONL under --score-dir (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PREFIXES = (2, 8)
CONTROL_K = 8


def question(k: int) -> str:
    return (f"These are the first {k} frames of a first-person video of a person doing a task. "
            "What is the person's goal?")


def prompt(k: int, image_token: str, with_images: bool) -> str:
    images = image_token * k if with_images else ""
    return f"<|im_start|>User:{images}{question(k)}<end_of_utterance>\nAssistant:"


def fractional_credit(scores: np.ndarray) -> float:
    """Candidate 0 is the positive; ties at the maximum share the credit."""
    top = scores.max()
    tied = np.flatnonzero(scores == top)
    return float(1.0 / len(tied)) if 0 in tied else 0.0


def selftest() -> None:
    assert fractional_credit(np.array([1.0, 0.5, 0.2])) == 1.0
    assert fractional_credit(np.array([1.0, 1.0, 0.2])) == 0.5
    assert fractional_credit(np.array([0.1, 1.0, 1.0])) == 0.0
    assert fractional_credit(np.zeros(5)) == 0.2
    assert "<image>" not in prompt(2, "<image>", False)
    assert prompt(2, "<image>", True).count("<image>") == 2
    print("selftest: PASS (fractional tie credit, prompts)")


def resolve_model_dir() -> str:
    rel = f"models--{MODEL_ID.replace('/', '--')}"
    for root in (Path.home() / ".cache/huggingface", Path(os.environ.get("HF_HOME", "/media/dhqg/d3/hf_cache"))):
        snaps = root / "hub" / rel / "snapshots"
        if snaps.exists():
            cands = sorted(p for p in snaps.iterdir() if p.is_dir() and any(p.glob("*.safetensors")))
            if cands:
                return str(cands[-1])
    raise FileNotFoundError(MODEL_ID)


class Scorer:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_dir = resolve_model_dir()
        self.processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir, local_files_only=True, dtype=dtype
        ).to(device).eval()
        self.dtype = str(dtype)
        self.device = device
        self.tok = self.processor.tokenizer
        self.image_token = self.processor.image_token
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.null_cache: dict[tuple[str, int], tuple[float, float, int]] = {}
        self.cand_cache: dict[str, list[int]] = {}
        self.checks = {"image_passes": 0, "null_passes": 0, "expanded_gt_unexpanded": 0}

    def cand_ids(self, text: str) -> list[int]:
        if text not in self.cand_cache:
            ids = self.tok(" " + text, add_special_tokens=False)["input_ids"]
            assert len(ids) > 0
            self.cand_cache[text] = ids
        return self.cand_cache[text]

    def _score(self, prefix_inputs, cand_lists):
        """prefix_inputs: processor output for ONE prefix (batch 1).
        Returns list of (sum, mean, n) for each candidate."""
        torch = self.torch
        pre = prefix_inputs["input_ids"][0].tolist()
        lp = len(pre)
        lens = [len(c) for c in cand_lists]
        width = lp + max(lens)
        b = len(cand_lists)
        ids = torch.full((b, width), self.pad_id, dtype=torch.long)
        att = torch.zeros((b, width), dtype=torch.long)
        for i, c in enumerate(cand_lists):
            ids[i, : lp + len(c)] = torch.tensor(pre + c)
            att[i, : lp + len(c)] = 1
        kwargs = {"input_ids": ids.to(self.device), "attention_mask": att.to(self.device)}
        if "pixel_values" in prefix_inputs:
            pv = prefix_inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
            kwargs["pixel_values"] = pv.expand(b, *pv.shape[1:]).contiguous()
            if "pixel_attention_mask" in prefix_inputs:
                pam = prefix_inputs["pixel_attention_mask"].to(self.device)
                kwargs["pixel_attention_mask"] = pam.expand(b, *pam.shape[1:]).contiguous()
        with torch.inference_mode():
            logits = self.model(**kwargs).logits
            # logits at position t predict token t+1: candidate tokens sit at lp..lp+n-1
            sl = logits[:, lp - 1 : lp - 1 + max(lens)].float()
            logp = torch.log_softmax(sl, dim=-1)
        out = []
        for i, c in enumerate(cand_lists):
            tgt = torch.tensor(c, device=logp.device)
            vals = logp[i, : len(c)].gather(-1, tgt[:, None])[:, 0]
            assert vals.numel() == len(c), "scored positions != candidate token count"
            # the scored positions must hold exactly the candidate ids
            assert ids[i, lp : lp + len(c)].tolist() == c
            s = float(vals.sum())
            out.append((s, s / len(c), len(c)))
        return out

    def image_scores(self, frames, k, texts):
        from PIL import Image

        text = prompt(k, self.image_token, True)
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames[:k]]
        inputs = self.processor(text=[text], images=[images], return_tensors="pt", do_image_splitting=False)
        unexpanded = len(self.tok(text, add_special_tokens=False)["input_ids"])
        expanded = inputs["input_ids"].shape[1]
        assert expanded > unexpanded, (expanded, unexpanded)
        self.checks["expanded_gt_unexpanded"] += 1
        self.checks["image_passes"] += 1
        self.checks["last_prefix_len"] = {"expanded": int(expanded), "unexpanded": int(unexpanded)}
        return self._score(inputs, [self.cand_ids(t) for t in texts])

    def null_scores(self, k, texts):
        missing = [t for t in dict.fromkeys(texts) if (t, k) not in self.null_cache]
        if missing:
            text = prompt(k, self.image_token, False)
            inputs = self.processor(text=[text], return_tensors="pt")
            for start in range(0, len(missing), 16):
                chunk = missing[start : start + 16]
                for t, res in zip(chunk, self._score(inputs, [self.cand_ids(x) for x in chunk])):
                    self.null_cache[(t, k)] = res
                self.checks["null_passes"] += 1
        return [self.null_cache[(t, k)] for t in texts]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_for(scorer, frames, k, texts, cand_sample_ids, meta):
    img = scorer.image_scores(frames, k, texts)
    null = scorer.null_scores(k, texts)
    mean_img = np.array([x[1] for x in img])
    mean_null = np.array([x[1] for x in null])
    sum_img = np.array([x[0] for x in img])
    sum_null = np.array([x[0] for x in null])
    assert [x[2] for x in img] == [x[2] for x in null]
    pmi_mean = mean_img - mean_null
    pmi_sum = sum_img - sum_null
    return {
        **meta, "k": k, "candidate_sample_ids": cand_sample_ids, "candidate_texts": texts,
        "n_tokens": [x[2] for x in img],
        "lp_img_sum": sum_img.tolist(), "lp_img_mean": mean_img.tolist(),
        "lp_null_sum": sum_null.tolist(), "lp_null_mean": mean_null.tolist(),
        "pmi_mean": pmi_mean.tolist(), "pmi_sum": pmi_sum.tolist(),
        "credit_pmi_mean": fractional_credit(pmi_mean), "credit_pmi_sum": fractional_credit(pmi_sum),
        "credit_lp_img_mean": fractional_credit(mean_img), "credit_lp_img_sum": fractional_credit(sum_img),
    }


def write_record(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec) + "\n")
    tmp.replace(path)


def smoke_stats(records):
    std_img = np.mean([np.std(r["lp_img_mean"]) for r in records])
    std_null = np.mean([np.std(r["lp_null_mean"]) for r in records])
    lp = np.concatenate([r["lp_img_mean"] for r in records])
    pmi = np.concatenate([r["pmi_mean"] for r in records])
    return {
        "n_records": len(records),
        "lp_img_mean_range": [float(lp.min()), float(lp.max())],
        "mean_within_query_std_lp_img": float(std_img),
        "mean_within_query_std_lp_null": float(std_null),
        "std_ratio_null_over_img": float(std_null / std_img),
        "fraction_pmi_gt0": float((pmi > 0).mean()),
        "credit_pmi_mean": float(np.mean([r["credit_pmi_mean"] for r in records])),
    }


def analyze(score_dir: Path, out: Path) -> int:
    """E1b-A (5-way lift CI lower bound > 1 at both prefixes, primary score) and
    E1b-B (descriptive SmolVLM2 minus SigLIP float32 baseline, same queries)."""
    import csv

    import run_ladder_v1 as L

    recs = [json.loads(p.read_text()) for p in sorted((score_dir / "scores").glob("*.jsonl"))]
    real = [r for r in recs if not r["control"]]
    ctrl = [r for r in recs if r["control"]]
    data = L.Data(encoders=(L.BASELINE_ENCODER,), minilm=False)
    pos = {s: i for i, s in enumerate(data.sample_ids)}
    fr = data.frames[L.BASELINE_ENCODER]
    t0 = data.text[L.BASELINE_ENCODER]["T0"]
    scores = ("pmi_mean", "pmi_sum", "lp_img_mean", "lp_img_sum")
    rows, pair_rows, diag = [], [], {}
    verdict_prefixes = []
    for k in PREFIXES:
        rk = sorted((r for r in real if r["k"] == k), key=lambda r: r["sample_id"])
        idx = np.asarray([pos[r["sample_id"]] for r in rk])
        for r in rk:  # candidate order must equal build_kway order
            assert r["candidate_sample_ids"] == [data.sample_ids[c] for c in data.cands5[pos[r["sample_id"]]]]
        vids = data.vids[idx]
        cos = L.candidate_scores(fr, t0, t0, np.zeros(data.n, bool), idx, data.cands5[idx], "A0", k)
        siglip = np.asarray([fractional_credit(c) for c in cos])
        for name in scores:
            credit = np.asarray([r[f"credit_{name}"] for r in rk])
            pt, lo, hi = CV.video_bootstrap(vids, credit, np.full(len(credit), 0.2))
            rows.append({"k": k, "score": name, "role": "primary" if name == "pmi_mean" else "secondary",
                         "n_queries": len(rk), "n_videos": int(len(np.unique(vids))),
                         "5way_acc": float(credit.mean()), "chance": 0.2, "acc_minus_chance": float(credit.mean() - 0.2),
                         "5way_lift": pt, "5way_lift_ci_lo": lo, "5way_lift_ci_hi": hi})
            if name == "pmi_mean" and lo > 1:
                verdict_prefixes.append(k)
            d, dlo, dhi = CV.video_bootstrap(vids, credit - siglip, np.ones(len(credit)))
            pair_rows.append({"k": k, "smolvlm2_score": name, "smolvlm2_acc": float(credit.mean()),
                              "siglip_acc": float(siglip.mean()), "delta": d, "ci95_lo": dlo, "ci95_hi": dhi,
                              "n_queries": len(rk)})
        img = np.asarray([r["lp_img_mean"] for r in rk])
        null = np.asarray([r["lp_null_mean"] for r in rk])
        pmi = np.asarray([r["pmi_mean"] for r in rk])
        diag[k] = {
            "lp_img_mean_range": [float(img.min()), float(img.max())],
            "mean_within_query_std_lp_img": float(img.std(axis=1).mean()),
            "mean_within_query_std_lp_null": float(null.std(axis=1).mean()),
            "std_ratio_null_over_img": float(null.std(axis=1).mean() / img.std(axis=1).mean()),
            "fraction_pmi_gt0": float((pmi > 0).mean()),
            "queries_with_tied_max_pmi_mean": int(sum(fractional_credit(p) not in (0.0, 1.0) for p in pmi)),
        }
    ctrl_rows = []
    if ctrl:
        by_id = {r["sample_id"]: r for r in real if r["k"] == CONTROL_K}
        cr = sorted(ctrl, key=lambda r: r["sample_id"])
        cv = data.vids[[pos[r["sample_id"]] for r in cr]]
        for name in scores:
            c_credit = np.asarray([r[f"credit_{name}"] for r in cr])
            r_credit = np.asarray([by_id[r["sample_id"]][f"credit_{name}"] for r in cr])
            pt, lo, hi = CV.video_bootstrap(cv, c_credit, np.full(len(cr), 0.2))
            ctrl_rows.append({"k": CONTROL_K, "score": name, "n_control_queries": len(cr),
                              "real_5way_acc": float(r_credit.mean()), "control_5way_acc": float(c_credit.mean()),
                              "control_lift": pt, "control_lift_ci_lo": lo, "control_lift_ci_hi": hi})
    e1b_a = len(verdict_prefixes) == len(PREFIXES)
    verdict = {"E1b-A": {"pass": e1b_a, "prefixes_lift_ci_lo_gt1": verdict_prefixes, "prefixes_evaluated": list(PREFIXES),
                         "rule": "primary per-token-mean PMI, fractional tie credit: 5-way lift CI lower bound > 1 "
                                 "(video bootstrap B=10000, seed 0) at BOTH evaluated prefixes"},
               "E1b-B": "descriptive only (paired_vs_siglip.csv)"}
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("summary.csv", rows), ("paired_vs_siglip.csv", pair_rows), ("control.csv", ctrl_rows)):
        if table:
            with (out / name).open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(table[0]))
                w.writeheader()
                w.writerows(table)
    lines = ["# E1b v2: SmolVLM2-500M zero-shot K=5 goal scoring", "",
             "Fractional tie credit; lift CI = video bootstrap B=10000, seed 0. Train/val only.", "",
             "| k | score | n | 5-way acc | lift [95% CI] |", "|---:|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['score']} | {r['n_queries']} | {r['5way_acc']:.4f} | "
                     f"{r['5way_lift']:.3f} [{r['5way_lift_ci_lo']:.3f}, {r['5way_lift_ci_hi']:.3f}] |")
    lines += ["", "## E1b-B (descriptive): SmolVLM2 minus SigLIP float32 baseline, identical queries", "",
              "| k | SmolVLM2 score | SmolVLM2 acc | SigLIP acc | delta [95% CI] |", "|---:|---|---:|---:|---|"]
    for r in pair_rows:
        lines.append(f"| {r['k']} | {r['smolvlm2_score']} | {r['smolvlm2_acc']:.4f} | {r['siglip_acc']:.4f} | "
                     f"{r['delta']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |")
    if ctrl_rows:
        lines += ["", f"## Mismatched-video control (every 8th query, k={CONTROL_K})", "",
                  "| score | n | real acc | control acc | control lift [95% CI] |", "|---|---:|---:|---:|---|"]
        for r in ctrl_rows:
            lines.append(f"| {r['score']} | {r['n_control_queries']} | {r['real_5way_acc']:.4f} | {r['control_5way_acc']:.4f} | "
                         f"{r['control_lift']:.3f} [{r['control_lift_ci_lo']:.3f}, {r['control_lift_ci_hi']:.3f}] |")
    lines += ["", "## Scorer diagnostics", "", "```", json.dumps(diag, indent=2), "```", "",
              "## Verdict", "", "```", json.dumps(verdict, indent=2), "```", ""]
    (out / "summary.md").write_text("\n".join(lines))
    (out / "analysis_config.json").write_text(json.dumps({
        "score_dir": str(score_dir), "n_records": len(recs), "n_real": len(real), "n_control": len(ctrl),
        "diagnostics": diag, "verdicts": verdict, "tie_rule": "fractional credit",
        "siglip_reference": f"{L.BASELINE_ENCODER}|A0|C0|T0 float32, fractional credit",
        "git_commit": L.git_commit(),
    }, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "results/hoi_detection_v1/manifest.jsonl")
    p.add_argument("--score-dir", type=Path, default=Path("/media/dhqg/d3/vlia_outputs/e1b_v2"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/e1b_smolvlm2_v2")
    p.add_argument("--smoke", action="store_true", help="20 queries, k in {2,8}; checks + stop rule")
    p.add_argument("--stride", type=int, default=1, help="use every n-th query in sorted sample_id order")
    p.add_argument("--probe", type=int, default=0, help="throughput probe: score only the first N queries")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--analyze", action="store_true", help="CPU analysis of the saved scores")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        selftest()
        return 0
    if args.analyze:
        return analyze(args.score_dir, args.output_dir)
    import torch
    import transformers

    from hoi_utils import sample_exact_video_frames

    rows = read_jsonl(args.manifest)
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("disallowed split")
    sample_ids = [r["sample_id"] for r in rows]
    vids = [r["video_uid"] for r in rows]
    texts = [r["why"] for r in rows]
    scenes = [r["scene"] for r in rows]
    cands5 = CV.build_kway(vids, texts, sample_ids, 5)
    order = sorted(range(len(rows)), key=lambda i: sample_ids[i])
    run_set = order[:: args.stride]
    if args.smoke:
        run_set = order[:20]
    elif args.probe:
        run_set = run_set[: args.probe]
    control_set = set(run_set[::8]) if not args.smoke else set()

    def donor_of(i):
        pool_ = [j for j in range(len(rows)) if scenes[j] != scenes[i] and vids[j] != vids[i]]
        rng = np.random.RandomState(zlib.crc32(f"{sample_ids[i]}|control".encode()) & 0xFFFFFFFF)
        return pool_[rng.randint(len(pool_))]

    score_dir = args.score_dir / ("smoke" if args.smoke else "scores")
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    scorer = Scorer(device)
    print(f"model {scorer.model_dir} dtype {scorer.dtype}; queries {len(run_set)}; controls {len(control_set)}", flush=True)

    def fname(i, k, control=False):
        return score_dir / f"{sample_ids[i].replace('/', '__')}__k{k}{'__control' if control else ''}.jsonl"

    def jobs_for(i):
        out = [(k, False) for k in PREFIXES if not fname(i, k).exists()]
        if i in control_set and not fname(i, CONTROL_K, True).exists():
            out.append((CONTROL_K, True))
        return out

    todo = [i for i in run_set if jobs_for(i)]
    print(f"{len(run_set) - len(todo)} queries already complete; {len(todo)} to score", flush=True)
    need_frames = []
    for i in todo:
        need_frames.append(i)
        if i in control_set:
            need_frames.append(donor_of(i))
    qpos = {}
    for pos, j in enumerate(need_frames):
        qpos.setdefault(j, pos)
    records = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {}

        def submit(j):
            if j not in futs:
                futs[j] = pool.submit(sample_exact_video_frames, rows[j]["video_path"])

        for j in need_frames[:12]:
            submit(j)
        cursor = 12
        for i in todo:
            while cursor < min(len(need_frames), qpos[i] + 12):
                submit(need_frames[cursor])
                cursor += 1
            submit(i)
            frames = futs.pop(i).result()
            ci = [int(c) for c in cands5[i]]
            if min(ci) < 0:
                raise ValueError(f"{sample_ids[i]} has no full K=5 set")
            ctexts = [texts[c] for c in ci]
            cids = [sample_ids[c] for c in ci]
            meta = {"sample_id": sample_ids[i], "video_uid": vids[i], "scene": scenes[i]}
            for k, control in jobs_for(i):
                if control:
                    d = donor_of(i)
                    submit(d)
                    dframes = futs.pop(d).result()
                    rec = record_for(scorer, dframes, k, ctexts, cids,
                                     {**meta, "control": True, "donor_sample_id": sample_ids[d]})
                else:
                    rec = record_for(scorer, frames, k, ctexts, cids, {**meta, "control": False})
                write_record(fname(i, k, control), rec)
                records.append(rec)
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"queries {done}/{len(todo)} elapsed {el:.0f}s eta {el / done * (len(todo) - done):.0f}s "
                      f"s/query {el / done:.2f} null_cache {len(scorer.null_cache)}", flush=True)
    elapsed = time.time() - t0
    info = {
        "model_id": MODEL_ID, "model_dir": scorer.model_dir, "dtype": scorer.dtype,
        "n_queries": len(run_set), "stride": args.stride, "prefixes": PREFIXES,
        "n_control": len(control_set), "elapsed_s": elapsed,
        "seconds_per_query": elapsed / max(1, done), "checks": scorer.checks,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": {p: sha256(ROOT / p) for p in (
            "results/prereg_daily_v1.md", "results/prereg_daily_v1_addendum.md",
            "results/prereg_ladder_v1.md", "results/prereg_ladder_v1_amendment_1.md")},
    }
    if args.smoke:
        recs = [r for r in records if not r["control"]]
        stats = smoke_stats(recs)
        stats_by_k = {k: smoke_stats([r for r in recs if r["k"] == k]) for k in PREFIXES}
        stop = stats["std_ratio_null_over_img"] > 3 or stats["fraction_pmi_gt0"] == 0
        info.update({"smoke": stats, "smoke_by_k": stats_by_k, "stop": stop})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "smoke_checks.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))
        return 4 if stop else 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "probe_info.json" if args.probe else "run_info.json"
    (args.output_dir / name).write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
