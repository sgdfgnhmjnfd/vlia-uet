#!/usr/bin/env python3
"""Anticipation in real seconds: context length x horizon on EgoIntent train/val.

Follow-up to `run_anticipation_horizon.py` (supervisor's question: "8 frames is
not even one action"). Instead of 8 frames per step, frames are re-sampled from
the raw clips at a FIXED rate (default 4 frames/s), so the context and the
horizon are measured in seconds. Rules: `results/prereg_time_horizon.md`.

Subcommands
  selftest  synthetic checks of the query geometry (no data)
  extract   decode every train/val step clip at --rate frames/s and encode the
            frames with SigLIP so400m (GPU). One .npz per video, resumable:
            <cache>/<video_uid>.npz with emb (M, D) float16, row (M,) manifest
            row of each frame, t (M,) seconds on the video timeline.
  run       for every anchor time t (every 1 s, same anchors for all settings)
            and context C in --contexts seconds, the query is the mean of the
            frames in (t-C, t], C2-centred (minus the mean of all earlier frames
            of the video; global mean if none). Target = WHAT (and WHY) of the
            step that contains t+H, H in --horizons seconds. 5 candidates =
            target + 4 distinct labels of the same video (seeded by video, t, H:
            identical for every C). Methods: ours, copy_rec (recognise the
            current step with the same context, choose the candidate closest to
            it), copy_true (true current label). Video-bootstrap 95% CIs and
            paired consecutive-H and consecutive-C differences.
            -> results/time_horizon_v1/

Typical run on the lab machine (GPU free):
  export HF_HOME=/media/dhqg/d3/hf_cache HF_HUB_OFFLINE=1
  PY=/home/dhqg/anaconda3/envs/lerobot/bin/python
  $PY scripts/run_time_horizon.py selftest
  $PY scripts/run_time_horizon.py extract            # GPU, ~30-60 min
  CUDA_VISIBLE_DEVICES="" $PY scripts/run_time_horizon.py run   # CPU, minutes
The locked test videos are never read (train/val manifest only).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402

ENCODER_ID = "google/siglip-so400m-patch14-384"
MANIFEST = ROOT / "results/hoi_detection_v1/manifest.jsonl"
CACHE = Path("/media/dhqg/d3/vlia_outputs/time_cache_4fps")
PREREG = ROOT / "results/prereg_time_horizon.md"
CONTEXTS = (1, 2, 4, 8, 16)
HORIZONS = (0.5, 1, 2, 4, 8, 12, 16, 20)
B = 2000


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("manifest must contain train/val rows only")
    return rows


def videos_in_order(rows):
    out = {}
    for i, r in enumerate(rows):
        out.setdefault(r["video_uid"], []).append(i)
    return {v: sorted(m, key=lambda i: int(rows[i]["step_id"])) for v, m in out.items()}


# ======================================================================
# extract (GPU)
# ======================================================================

def decode_at_rate(path: str, rate: float, size: int = 384):
    """RGB frames at times 0, 1/rate, 2/rate, ... < clip duration (at least one)."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        raise RuntimeError(f"cannot read frame count of {path}")
    duration = n / fps
    want = sorted({min(n - 1, int(round(j * fps / rate))) for j in range(max(1, int(np.ceil(duration * rate - 1e-9))))})
    frames, idx, j = [], 0, 0
    while j < len(want):
        ok = cap.grab()
        if not ok:
            break
        if idx == want[j]:
            ok, bgr = cap.retrieve()
            if ok:  # SigLIP squashes to 384x384 anyway; resizing here keeps memory small
                frames.append(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy())
            j += 1
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, duration


def cmd_extract(args) -> int:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    rows = read_rows(MANIFEST)
    order = videos_in_order(rows)
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(ENCODER_ID, token=False)
    model = AutoModel.from_pretrained(ENCODER_ID, token=False).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    def encode(frames):
        out = []
        for s in range(0, len(frames), args.batch_size):
            pv = processor(images=[Image.fromarray(f) for f in frames[s: s + args.batch_size]],
                           return_tensors="pt")["pixel_values"].to(device, dtype=next(model.parameters()).dtype)
            with torch.inference_mode():
                feats = model.get_image_features(pixel_values=pv)
                if hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
            out.append(F.normalize(feats.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    t0, done = time.time(), 0
    todo = [v for v in sorted(order) if not (args.cache / f"{v}.npz").exists()]
    print(f"{len(order)} videos, {len(todo)} to extract at {args.rate} frames/s", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for v in todo:
            members = order[v]
            decoded = list(pool.map(lambda i: decode_at_rate(rows[i]["video_path"], args.rate), members))
            embs, row_of, times, offset = [], [], [], 0.0
            for i, (frames, duration) in zip(members, decoded):
                embs.append(encode(frames))
                row_of += [i] * len(frames)
                times += [offset + j / args.rate for j in range(len(frames))]
                offset += duration
            emb = np.concatenate(embs).astype(np.float16)
            tmp = args.cache / f"{v}.tmp.npz"
            np.savez(tmp, emb=emb, row=np.asarray(row_of), t=np.asarray(times), rate=args.rate,
                     sample_ids=np.asarray([rows[i]["sample_id"] for i in members]))
            tmp.replace(args.cache / f"{v}.npz")
            done += 1
            el = time.time() - t0
            print(f"  {v}: {len(members)} steps, {len(emb)} frames ({offset:.0f} s); "
                  f"{done}/{len(todo)} videos, {el:.0f}s elapsed, eta {el / done * (len(todo) - done):.0f}s", flush=True)
    (args.cache / "meta.json").write_text(json.dumps({"encoder": ENCODER_ID, "rate": args.rate,
                                                      "dtype": "float16", "manifest": str(MANIFEST)}, indent=2))
    return 0


# ======================================================================
# run (CPU)
# ======================================================================

def query_vec(emb, a, n_ctx, mu):
    """Mean of C2-centred frames a-n_ctx+1..a (unit), ref = mean of frames before the window."""
    start = a - n_ctx + 1
    ref = emb[:start].mean(axis=0) if start > 0 else mu
    return normalize(normalize(emb[start: a + 1] - ref).mean(axis=0))


def candidates(uid, a, h, target, distinct, seed=0, n=5):
    others = [x for x in distinct if x != target]
    if len(others) < n - 1:
        return None
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{uid}|{a}|{h}".encode()))
    perm = rng.permutation(len(others))[: n - 1]
    return [target] + [others[j] for j in perm]


def boot(values, vids):
    return AH.video_bootstrap(np.asarray(values, float), np.asarray(vids), B)


def cmd_run(args) -> int:
    out = ROOT / "results/time_horizon_v1"
    if (out / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{out}/summary.csv exists (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(MANIFEST)
    order = videos_in_order(rows)
    cache = {}
    for v in sorted(order):
        z = np.load(args.cache / f"{v}.npz")
        cache[v] = {"emb": normalize(z["emb"].astype(np.float32)), "row": z["row"], "rate": float(z["rate"])}
    rate = {c["rate"] for c in cache.values()}
    assert len(rate) == 1, rate
    rate = rate.pop()
    mu = normalize(np.concatenate([c["emb"] for c in cache.values()])).mean(axis=0)
    labels = ("what", "why")
    vocab = sorted({r[k] for r in rows for k in labels})
    temb = AH.encode_texts(ENCODER_ID, vocab, device=args.device)
    tix = {t: i for i, t in enumerate(vocab)}
    n_c = {c: int(round(c * rate)) for c in args.contexts}
    n_h = {h: int(round(h * rate)) for h in args.horizons}
    max_c, max_h, step = max(n_c.values()), max(n_h.values()), int(round(rate))
    recs, dropped = [], []
    for v in sorted(cache):
        emb, row = cache[v]["emb"], cache[v]["row"]
        anchors = range(max_c - 1, len(emb) - max_h, step)
        if len(anchors) == 0:
            dropped.append(v)
            continue
        q = {c: np.stack([query_vec(emb, a, n_c[c], mu) for a in anchors]) for c in args.contexts}
        for key in labels:
            distinct = sorted({rows[i][key] for i in order[v]})
            g = temb[[tix[t] for t in distinct]]
            for c in args.contexts:
                rec_now = [distinct[int(np.argmax(g @ q[c][j]))] for j in range(len(anchors))]
                for j, a in enumerate(anchors):
                    now_row = int(row[a])
                    true_now = rows[now_row][key]
                    for h in args.horizons:
                        tgt_row = int(row[a + n_h[h]])
                        target = rows[tgt_row][key]
                        cand = candidates(v, a, h, target, distinct)
                        if cand is None:
                            continue
                        ce = temb[[tix[t] for t in cand]]
                        s_ours = ce @ q[c][j]
                        s_rec = ce @ temb[tix[rec_now[j]]]
                        s_true = ce @ temb[tix[true_now]]
                        recs.append((key, c, h, v, a, now_row == tgt_row,
                                     float((s_ours[0] > s_ours[1:]).all()),
                                     float((s_rec[0] > s_rec[1:]).all()),
                                     float((s_true[0] > s_true[1:]).all())))
        print(f"  {v}: {len(anchors)} anchors", flush=True)
    if dropped:
        print(f"dropped {len(dropped)} videos shorter than {max_c + max_h} frames: {dropped}", flush=True)
    key_a = np.array([r[0] for r in recs])
    c_a = np.array([r[1] for r in recs], float)
    h_a = np.array([r[2] for r in recs], float)
    vid_a = np.array([r[3] for r in recs])
    anc_a = np.array([r[4] for r in recs])
    same_a = np.array([r[5] for r in recs])
    hit = {"ours": np.array([r[6] for r in recs]), "copy_rec": np.array([r[7] for r in recs]),
           "copy_true": np.array([r[8] for r in recs])}
    summary = []
    for key in labels:
        for c in args.contexts:
            for h in args.horizons:
                m = (key_a == key) & (c_a == c) & (h_a == h)
                row = {"label": key.upper(), "context_s": c, "horizon_s": h, "n_queries": int(m.sum()),
                       "n_videos": len(np.unique(vid_a[m])), "frac_same_step": float(same_a[m].mean())}
                for name, x in hit.items():
                    p, lo, hi = boot(x[m], vid_a[m])
                    row.update({f"{name}_acc5": p, f"{name}_lo": lo, f"{name}_hi": hi})
                d = boot(hit["ours"][m] - hit["copy_rec"][m], vid_a[m])
                row.update({"ours_minus_copy_rec": d[0], "omc_lo": d[1], "omc_hi": d[2]})
                summary.append(row)

    def paired(axis_vals, fixed_name, moving_name, fixed_vals):
        res = []
        for key in labels:
            for f in fixed_vals:
                for x0, x1 in zip(axis_vals, axis_vals[1:]):
                    sel = {}
                    for x in (x0, x1):
                        m = (key_a == key) & ((c_a if fixed_name == "context_s" else h_a) == f) & \
                            ((h_a if fixed_name == "context_s" else c_a) == x)
                        sel[x] = dict(zip(zip(vid_a[m], anc_a[m]), hit["ours"][m]))
                    ks = sorted(set(sel[x0]) & set(sel[x1]))
                    a0 = np.array([sel[x0][k] for k in ks])
                    a1 = np.array([sel[x1][k] for k in ks])
                    d = boot(a1 - a0, np.array([k[0] for k in ks]))
                    res.append({"label": key.upper(), fixed_name: f, f"{moving_name}_from": x0, f"{moving_name}_to": x1,
                                "n": len(ks), "acc_from": a0.mean(), "acc_to": a1.mean(),
                                "diff": d[0], "diff_lo": d[1], "diff_hi": d[2], "flat": d[1] <= 0 <= d[2]})
        return res

    by_h = paired(list(args.horizons), "context_s", "horizon_s", args.contexts)
    by_c = paired(list(args.contexts), "horizon_s", "context_s", args.horizons)
    for name, rws in (("summary.csv", summary), ("paired_horizon.csv", by_h), ("paired_context.csv", by_c)):
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(rws[0]))
            w.writeheader()
            w.writerows(rws)
    (out / "config.json").write_text(json.dumps({
        "rate": rate, "contexts_s": list(args.contexts), "horizons_s": list(args.horizons),
        "anchor_step_s": 1.0, "centering": "C2", "encoder": ENCODER_ID, "B": B,
        "dropped_videos": dropped, "test_split_read": False}, indent=2))
    plot(summary, out / "time_horizon.png")
    for r in summary:
        print(f"{r['label']} C={r['context_s']:>2}s H={r['horizon_s']:>4}s n={r['n_queries']:>5} "
              f"ours {100 * r['ours_acc5']:.1f} [{100 * r['ours_lo']:.1f}; {100 * r['ours_hi']:.1f}] "
              f"copy_rec {100 * r['copy_rec_acc5']:.1f} copy_true {100 * r['copy_true_acc5']:.1f} "
              f"ours-copy_rec {100 * r['ours_minus_copy_rec']:+.1f} [{100 * r['omc_lo']:+.1f}; {100 * r['omc_hi']:+.1f}]",
              flush=True)
    print("--- longer context, same horizon (ours, diff = longer minus shorter) ---")
    for r in by_c:
        print(f"{r['label']} H={r['horizon_s']:>4}s C {r['context_s_from']:>2}->{r['context_s_to']:>2}s "
              f"{100 * r['acc_from']:.1f} -> {100 * r['acc_to']:.1f} diff {100 * r['diff']:+.1f} "
              f"[{100 * r['diff_lo']:+.1f}; {100 * r['diff_hi']:+.1f}] {'FLAT' if r['flat'] else ''}")
    print(f"saved {out}", flush=True)
    return 0


def plot(summary, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key in zip(axes, ("WHAT", "WHY")):
        for c in sorted({r["context_s"] for r in summary}):
            rs = [r for r in summary if r["label"] == key and r["context_s"] == c]
            ax.plot([r["horizon_s"] for r in rs], [100 * r["ours_acc5"] for r in rs], marker="o", label=f"see {c} s")
        ax.axhline(20, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(key)
        ax.set_xlabel("seconds ahead")
    axes[0].set_ylabel("5-way accuracy (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ======================================================================
# selftest
# ======================================================================

def cmd_selftest(args) -> int:
    rng = np.random.default_rng(0)
    emb = normalize(rng.normal(size=(40, 8)))
    mu = emb.mean(axis=0)
    q = query_vec(emb, 9, 4, mu)
    ref = emb[:6].mean(axis=0)
    assert np.allclose(q, normalize(normalize(emb[6:10] - ref).mean(axis=0))), "window must be a-3..a"
    assert np.allclose(query_vec(emb, 3, 4, mu), normalize(normalize(emb[0:4] - mu).mean(axis=0))), "fallback"
    d = ["a", "b", "c", "d", "e", "f"]
    c1, c2 = candidates("v", 10, 2, "c", d), candidates("v", 10, 2, "c", d)
    assert c1 == c2 and c1[0] == "c" and len(set(c1)) == 5, "candidates must be seeded and distinct"
    assert candidates("v", 10, 2, "a", ["a", "b"]) is None
    print("selftest passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["selftest", "extract", "run"])
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--rate", type=float, default=4.0, help="frames per second (extract)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=3, help="decode threads (extract)")
    p.add_argument("--device", default="cuda", help="extract: GPU; run: text encoder device")
    p.add_argument("--contexts", type=float, nargs="+", default=list(CONTEXTS))
    p.add_argument("--horizons", type=float, nargs="+", default=list(HORIZONS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.cmd != "selftest" and not PREREG.exists():
        raise SystemExit(f"missing preregistration {PREREG}")
    if args.cmd == "run" and args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            args.device = "cpu"
    return {"selftest": cmd_selftest, "extract": cmd_extract, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
