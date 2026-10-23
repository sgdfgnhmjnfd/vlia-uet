#!/usr/bin/env python3
"""Ladder Stage 2: multi-encoder frame/text embedding cache (train/val only).

Implements `results/prereg_ladder_v1.md` Stage 2 with amendment 1
(`results/prereg_ladder_v1_amendment_1.md`): the baseline encoder is stored
as float32, the other encoders as float16 (frames); text embeddings are float32.

Frames are decoded with the same sampler as E1 (`sample_exact_video_frames`)
and fed in the same 32-frame batches over consecutive samples as E1's
`_model_features`, so the baseline re-extraction reproduces E1's numerics as
closely as the GPU allows. All requested encoders are run on each decoded
batch (decode once). Per-sample SHA1 hashes of the decoded frames are written
on the first pass and checked on every later pass.

Output per encoder: <out-root>/<encoder_name>/
  frame_emb.npy (N, 8, D), sample_ids.json, texts.json (sorted unique WHY),
  text_T0.npy / text_T1.npy (U, D) float32, meta.json (truncation counts, ...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hoi_utils import sample_exact_video_frames  # noqa: E402

ENCODERS = (
    "google/siglip-base-patch16-224",
    "google/siglip-base-patch16-384",
    "google/siglip2-base-patch16-224",
    "google/siglip-so400m-patch14-384",
    "openai/clip-vit-large-patch14",
)
BASELINE = ENCODERS[0]
TEMPLATES = {"T0": "{why}", "T1": "a first-person photo of a person doing: {why}"}
SAMPLES_PER_BATCH = 4  # 4 samples x 8 frames = E1's 32-frame batches


def encoder_dir_name(name: str) -> str:
    return name.split("/")[-1]


def text_max_length(name: str) -> int:
    return 77 if "clip" in name else 64


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(manifest: Path) -> list[dict]:
    rows = read_jsonl(manifest)
    if any(row.get("split") not in ("train", "val") for row in rows):
        raise ValueError("Manifest must contain train/val records only.")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample_id in manifest.")
    return rows


def load_encoder(name: str, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(name, token=False)
    model = AutoModel.from_pretrained(name, token=False).to(device).eval()
    return model, processor


def image_features(model, processor, frames: list[np.ndarray], device) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    batch = [Image.fromarray(frame) for frame in frames]
    inputs = processor(images=batch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = F.normalize(features.float(), dim=-1)
    return features.cpu().numpy().astype(np.float32)


def text_features(model, processor, name: str, texts: list[str], device, batch_size: int):
    import torch
    import torch.nn.functional as F

    max_length = text_max_length(name)
    tokenizer = processor.tokenizer
    untruncated = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)["input_ids"]
    truncated = int(sum(len(ids) > max_length for ids in untruncated))
    out = []
    for start in range(0, len(texts), batch_size):
        tokens = processor(
            text=texts[start : start + batch_size],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            features = model.get_text_features(**tokens)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features.float(), dim=-1)
        out.append(features.cpu().numpy())
    return np.concatenate(out).astype(np.float32), truncated


def pixel_values(processor, frames: np.ndarray):
    from PIL import Image

    return processor(images=[Image.fromarray(f) for f in frames], return_tensors="pt")["pixel_values"]


def encode_pixels(model, pixel_values, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values.to(device))
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = F.normalize(features.float(), dim=-1)
    return features.cpu().numpy().astype(np.float32)


def frame_hash(frames: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(str(frames.shape).encode())
    digest.update(np.ascontiguousarray(frames).tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/hoi_detection_v1/manifest.jsonl")
    parser.add_argument("--out-root", type=Path, default=Path("/media/dhqg/d3/vlia_outputs/ladder"))
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    parser.add_argument("--limit", type=int, default=0, help="throughput probe on the first N samples")
    parser.add_argument("--batch-size", type=int, default=32, help="text batch size (E1 used 32)")
    parser.add_argument("--decode-workers", type=int, default=6)
    parser.add_argument(
        "--preprocess-in-workers",
        action="store_true",
        help="run each encoder's image processor in the decode threads (per-image, "
        "so pixel values are identical to a 32-frame processor call)",
    )
    return parser.parse_args()


def main() -> int:
    import torch
    import transformers

    args = parse_args()
    for name in args.encoders:
        if name not in ENCODERS:
            raise ValueError(f"Encoder not preregistered: {name}")
    rows = load_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = {name: args.out_root / encoder_dir_name(name) for name in args.encoders}
    if not args.limit:
        for name, path in dirs.items():
            if (path / "frame_emb.npy").exists():
                raise FileExistsError(f"Refusing to overwrite {path / 'frame_emb.npy'}")
    hash_path = args.out_root / "frame_hashes.json"
    known_hashes = json.loads(hash_path.read_text()) if hash_path.exists() else None

    t0 = time.time()
    models = {name: load_encoder(name, device) for name in args.encoders}
    load_seconds = time.time() - t0
    print(f"loaded {len(models)} encoders in {load_seconds:.1f}s on {device}", flush=True)

    unique_texts = sorted({row["why"] for row in rows})
    text_out: dict[str, dict] = {}
    for name, (model, processor) in models.items():
        text_out[name] = {}
        for tid, template in TEMPLATES.items():
            emb, truncated = text_features(
                model, processor, name, [template.format(why=t) for t in unique_texts], device, args.batch_size
            )
            text_out[name][tid] = (emb, truncated)
            print(f"{name} {tid}: {emb.shape} truncated={truncated}", flush=True)

    feats: dict[str, list[np.ndarray]] = {name: [] for name in args.encoders}
    hashes: dict[str, str] = {}
    timing = {name: 0.0 for name in args.encoders}
    decode_wait = 0.0
    t_start = time.time()
    def load(row):
        sampled = sample_exact_video_frames(row["video_path"])
        n_frames, digest = sampled.shape[0], frame_hash(sampled)
        if not args.preprocess_in_workers:
            return sampled, None, n_frames, digest
        # keep only the resized pixel tensors; full-resolution frames are dropped here
        return None, {name: pixel_values(proc, sampled) for name, (_, proc) in models.items()}, n_frames, digest

    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        futures = [pool.submit(load, row) for row in rows[: args.decode_workers + SAMPLES_PER_BATCH]]
        next_submit = len(futures)
        for start in range(0, len(rows), SAMPLES_PER_BATCH):
            group = rows[start : start + SAMPLES_PER_BATCH]
            frames = []
            pixels = {name: [] for name in args.encoders}
            for offset, row in enumerate(group):
                tw = time.time()
                sampled, pv, n_frames, digest = futures[start + offset].result()
                decode_wait += time.time() - tw
                futures[start + offset] = None
                if pv is not None:
                    for name in args.encoders:
                        pixels[name].append(pv[name])
                if next_submit < len(rows):
                    futures.append(pool.submit(load, rows[next_submit]))
                    next_submit += 1
                if n_frames != 8:
                    raise AssertionError(f"{row['sample_id']}: {n_frames} frames")
                if known_hashes is not None and known_hashes.get(row["sample_id"]) != digest:
                    raise AssertionError(f"Decoded frames differ from first pass: {row['sample_id']}")
                hashes[row["sample_id"]] = digest
                if sampled is not None:
                    frames.extend(list(sampled))
            for name, (model, processor) in models.items():
                tm = time.time()
                if args.preprocess_in_workers:
                    import torch as _t

                    feats[name].append(encode_pixels(model, _t.cat(pixels[name]), device))
                else:
                    feats[name].append(image_features(model, processor, frames, device))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timing[name] += time.time() - tm
            done = start + len(group)
            if done % 100 < SAMPLES_PER_BATCH or done == len(rows):
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(rows) - done)
                print(
                    f"samples {done}/{len(rows)} elapsed {elapsed:.0f}s eta {eta:.0f}s "
                    f"decode_wait {decode_wait:.0f}s "
                    + " ".join(f"{encoder_dir_name(n)}={t:.0f}s" for n, t in timing.items()),
                    flush=True,
                )
    wall = time.time() - t_start
    throughput = {
        "n_samples": len(rows),
        "wall_seconds": wall,
        "decode_wait_seconds": decode_wait,
        "encoder_seconds": {name: timing[name] for name in args.encoders},
        "projected_full_seconds_per_encoder": {
            name: timing[name] / len(rows) * 2503 for name in args.encoders
        },
        "projected_full_wall_seconds": wall / len(rows) * 2503,
    }
    print(json.dumps(throughput, indent=2), flush=True)
    if args.limit:
        return 0

    if known_hashes is None:
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        hash_path.write_text(json.dumps(hashes, indent=0, sort_keys=True))
    sample_ids = [row["sample_id"] for row in rows]
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    for name in args.encoders:
        path = dirs[name]
        path.mkdir(parents=True, exist_ok=True)
        dtype = np.float32 if name == BASELINE else np.float16
        values = np.concatenate(feats[name]).reshape(len(rows), 8, -1)
        if not np.isfinite(values).all():
            raise AssertionError(f"{name}: non-finite frame embeddings")
        np.save(path / "frame_emb.npy", values.astype(dtype))
        (path / "sample_ids.json").write_text(json.dumps(sample_ids))
        (path / "texts.json").write_text(json.dumps(unique_texts))
        for tid, (emb, _) in text_out[name].items():
            np.save(path / f"text_{tid}.npy", emb)
        meta = {
            "encoder": name,
            "frame_dtype": np.dtype(dtype).name,
            "text_dtype": "float32",
            "frame_shape": list(values.shape),
            "text_max_length": text_max_length(name),
            "templates": TEMPLATES,
            "truncated_unique_texts": {tid: t for tid, (_, t) in text_out[name].items()},
            "n_unique_texts": len(unique_texts),
            "versions": versions,
            "device": str(device),
            "image_batch_frames": SAMPLES_PER_BATCH * 8,
            "preprocess_in_workers": bool(args.preprocess_in_workers),
            "frame_hashes": str(hash_path),
            "frame_hashes_checked_against_first_pass": known_hashes is not None,
            "throughput": throughput,
            "hf_home": os.environ.get("HF_HOME"),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
