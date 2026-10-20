#!/usr/bin/env python3
"""E1b zero-shot goal scoring with SmolVLM2-500M-Video-Instruct."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV
from hoi_utils import sample_exact_video_frames

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PREFIXES = (2, 4, 6, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/hoi_detection_v1/manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/e1b_smolvlm2_v1")
    parser.add_argument("--score-output", type=Path, default=Path("/media/dhqg/d3/vlia_outputs/e1b/scores.jsonl"))
    parser.add_argument("--hf-home", type=Path, default=Path("/media/dhqg/d3/hf_cache"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_dir(model_id: str, hf_home: Path) -> str | None:
    rel = f"models--{model_id.replace('/', '--')}"
    roots = [
        Path.home() / ".cache" / "huggingface",
        hf_home,
    ]
    for root in roots:
        snapshot_root = root / "hub" / rel / "snapshots"
        if snapshot_root.exists():
            candidates = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
            if candidates:
                return str(candidates[-1])
    return None


def load_model_and_processor(hf_home: Path):
    model_dir = resolve_model_dir(MODEL_ID, hf_home)
    if model_dir is not None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True)
        return model, processor, model_dir
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, cache_dir=str(hf_home))
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=str(hf_home))
    return model, processor, model_id


def prompt_prefix(question: str, k: int, processor: Any) -> str:
    return f"<|im_start|>User:{processor.image_token * k}{question}<end_of_utterance>\nAssistant:"


def build_text_sequence(prefix: str, candidate: str, frames: list[Any] | None, processor: Any):
    full_text = prefix + candidate + "<end_of_utterance>"
    if frames is None:
        return full_text, None
    return full_text, [Image.fromarray(np.asarray(frame).astype(np.uint8)) for frame in frames]


def score_candidate(model: Any, processor: Any, device: torch.device, question: str, candidate: str, frames_k: list[np.ndarray] | None, k: int) -> tuple[float, float, int]:
    if frames_k is None:
        prefix = f"<|im_start|>User:{question}<end_of_utterance>\nAssistant:"
        full_text = prefix + candidate + "<end_of_utterance>"
        inputs = processor(text=[full_text], return_tensors="pt", padding=False)
        prefix_len = len(processor.tokenizer(prefix, add_special_tokens=False)["input_ids"])
    else:
        prefix = prompt_prefix(question, k, processor)
        full_text = prefix + candidate + "<end_of_utterance>"
        inputs = processor(text=[full_text], images=[frames_k], return_tensors="pt", padding=False, do_image_splitting=False)
        prefix_len = len(processor.tokenizer(prefix, add_special_tokens=False)["input_ids"])
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100
    if "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss.detach().float().cpu().item()
    n_tokens = len(processor.tokenizer(candidate, add_special_tokens=False)["input_ids"])
    lp = -loss
    return float(lp), float(lp), n_tokens


def average_logp_pmi(model, processor, device, question, candidate, frames_k, k):
    img_lp, _, n_tokens = score_candidate(model, processor, device, question, candidate, frames_k, k)
    null_lp, _, _ = score_candidate(model, processor, device, question, candidate, None, 0)
    return img_lp - null_lp, img_lp, null_lp, n_tokens


def compute_scores(
    rows: list[dict[str, Any]],
    output_path: Path,
    outdir: Path,
    smoke: bool = False,
    max_queries: int | None = None,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, model_path = load_model_and_processor(Path(os.environ.get("HF_HOME", "/media/dhqg/d3/hf_cache")))
    model.to(device)
    model.eval()
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device, dtype=torch.float16)
    sample_rows = []
    if smoke:
        prefixes = (2, 8)
        rows = rows[:20]
    else:
        prefixes = PREFIXES
    if max_queries is not None:
        rows = rows[:max_queries]
    for prefix in prefixes:
        for idx, record in enumerate(rows):
            if prefix > record["num_frames"]:
                continue
            sample_id = record["sample_id"]
            out_path = output_path.parent / f"{sample_id.replace('/', '__')}__k{prefix}.jsonl"
            if out_path.exists():
                if not resume:
                    raise FileExistsError(
                        f"Refusing to overwrite completed score {out_path}; use --resume."
                    )
                with out_path.open("r", encoding="utf-8") as handle:
                    saved = [json.loads(line) for line in handle if line.strip()]
                if len(saved) != 1 or saved[0].get("sample_id") != sample_id or saved[0].get("k") != prefix:
                    raise ValueError(f"Existing score file is invalid or mismatched: {out_path}")
                sample_rows.append({**saved[0], "prefix": prefix})
                continue
            question = f"These are the first {prefix} frames of a first-person video of a person doing a task. What is the person's goal?"
            frames = sample_exact_video_frames(record["video_path"])[:prefix]
            cands = CV.build_kway([r["video_uid"] for r in rows], [r["why"] for r in rows], [r["sample_id"] for r in rows], 5)[idx]
            # keep all rows in order of candidate list, ensuring exact current sample at position 0
            primary_scores, img_scores, null_scores, token_counts = [], [], [], []
            candidate_ids = []
            for cid in cands:
                if cid < 0:
                    continue
                cand_record = rows[int(cid)]
                candidate_ids.append(cand_record["sample_id"])
                p, li, ln, nt = average_logp_pmi(model, processor, device, question, cand_record["why"], frames, prefix)
                primary_scores.append(p)
                img_scores.append(li)
                null_scores.append(ln)
                token_counts.append(nt)
            if len(primary_scores) != 5:
                raise ValueError(f"Expected 5 candidate scores for {sample_id}, got {len(primary_scores)}")
            target_index = int(np.argmax(primary_scores))
            pred_sample = candidate_ids[target_index]
            correct = pred_sample == sample_id
            sample_rows.append({
                "sample_id": sample_id,
                "video_uid": record["video_uid"],
                "scene": record["scene"],
                "prefix": prefix,
                "candidate_sample_ids": candidate_ids,
                "primary": primary_scores,
                "lp_img": img_scores,
                "lp_null": null_scores,
                "n_tokens": token_counts,
                "pred_sample_id": pred_sample,
                "correct": bool(correct),
            })
            temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            if temp_path.exists():
                raise FileExistsError(f"Refusing to overwrite incomplete score file {temp_path}")
            with temp_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "sample_id": sample_id,
                    "video_uid": record["video_uid"],
                    "scene": record["scene"],
                    "k": prefix,
                    "candidate_sample_ids": candidate_ids,
                    "lp_img": img_scores,
                    "lp_null": null_scores,
                    "primary": primary_scores,
                    "n_tokens": token_counts,
                    "pred_sample_id": pred_sample,
                    "correct": bool(correct),
                }) + "\n")
            temp_path.replace(out_path)
    return sample_rows, {"model_id": MODEL_ID, "model_path": model_path, "n_rows": len(rows)}


def build_summary(rows: list[dict[str, Any]], output_dir: Path):
    rows_by_prefix = defaultdict(list)
    for row in rows:
        rows_by_prefix[row["prefix"]].append(row)
    all_summary = []
    for prefix, items in sorted(rows_by_prefix.items()):
        correct = np.asarray([1 if item["correct"] else 0 for item in items], dtype=np.float64)
        acc = float(correct.mean())
        chance = 1.0 / 5.0
        lift = acc / chance
        all_summary.append({
            "prefix_frames": prefix,
            "n_queries": len(items),
            "accuracy": acc,
            "chance": chance,
            "lift": lift,
            "acc_minus_chance": acc - chance,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prefix_frames", "n_queries", "accuracy", "chance", "lift", "acc_minus_chance"])
        writer.writeheader()
        writer.writerows(all_summary)
    return all_summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    rows = [r for r in rows if r.get("split") in {"train", "val"}]
    if args.smoke:
        rows = rows[:20]
        prefixes = (2, 8)
    else:
        prefixes = PREFIXES
    score_rows, meta = compute_scores(
        rows,
        args.score_output,
        args.output_dir,
        smoke=args.smoke,
        max_queries=args.max_queries,
        resume=args.resume,
    )
    build_summary(score_rows, args.output_dir)
    config = {
        "args": {
            "manifest": str(args.manifest),
            "output_dir": str(args.output_dir),
            "score_output": str(args.score_output),
            "smoke": args.smoke,
            "max_queries": args.max_queries,
            "resume": args.resume,
        },
        "model_id": MODEL_ID,
        "model_path": meta["model_path"],
        "n_rows": meta["n_rows"],
        "prefixes": list(prefixes),
        "git_commit": __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "preregistration_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                ROOT / "results/prereg_daily_v1.md",
                ROOT / "results/prereg_daily_v1_addendum.md",
                ROOT / "results/prereg_ladder_v1.md",
            )
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "pillow": __import__("PIL").__version__,
        },
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
