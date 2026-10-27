#!/usr/bin/env python3
"""Anticipation-horizon study on the EgoIntent train/val cache (training-free).

Question (suggested by the lab group, 2026-10-27): "intention in robotics is a
future action". At an anchor frame t the model sees a fixed context of the last
C frames (t-C+1..t) and must pick the action (WHAT) -- or, optionally, the WHY --
of the step that contains frame t+H. Sweeping the horizon H shows how far ahead
the goal can be predicted and how accuracy and uncertainty change with H.

Timeline. Each video is the concatenation of its steps in step order, 8 cached
SigLIP frames per step (the ladder cache, frame_emb.npy (N, 8, D)), so H is in
"cache frames": H=8 is on average one step ahead. With --video-root-check the
real clip durations are read with OpenCV and the mean horizon in seconds is
reported too (each cache frame covers duration/8 of its step).

Fairness across H. The SAME anchors are used at every H: t in
[C-1, T-1-max(H)] for a video with T cache frames. Videos too short for that
are dropped (printed).

Scoring (same as the frozen pipeline, no training):
  query  = normalise(mean of the C context frames), optionally C2-centred
           (minus the mean of all frames strictly before the context window,
           fallback C1 = minus the mean frame of all queries) ;
  text   = SigLIP text embedding of "a first-person photo of a person doing: {label}" ;
  5-way  = target label + 4 distinct labels of other steps of the same video
           (seeded), strict top-1 (ties count as wrong) ; 10-way likewise ;
  MRR    = rank of the target among all distinct labels of the video.
Reported per H, overall and split into "same step" (t+H still in the step of
t: recognition) and "changed step" (true anticipation), plus
  oracle = the target step's own 8 frames as the query (upper bound),
  margin = top-1 minus top-2 score among the 5 candidates (uncertainty),
  selective accuracy at 50% / 20% coverage (most confident queries).
95% CIs: bootstrap over videos (B resamples, seed 0).

Outputs in --out: summary.csv, per_query.csv.gz, config.json, horizon.png.
Only train/val rows are read; the locked test videos are never touched.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "a first-person photo of a person doing: {label}"
DEFAULT_HORIZONS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ----------------------------------------------------------------------
# timeline and queries (pure numpy, unit-tested on synthetic data)
# ----------------------------------------------------------------------

def build_timelines(rows: list[dict], frames: np.ndarray) -> dict[str, dict]:
    """Per video: frames (T, D) in step order, step index of every frame, and
    the row index of every step."""
    by_video = defaultdict(list)
    for i, row in enumerate(rows):
        by_video[row["video_uid"]].append(i)
    out = {}
    for uid, members in by_video.items():
        members = sorted(members, key=lambda i: int(rows[i]["step_id"]))
        steps = [int(rows[i]["step_id"]) for i in members]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{uid}: duplicate step ids")
        f = frames[members]  # (S, 8, D)
        s, per, d = f.shape
        out[uid] = {
            "rows": members,
            "frames": f.reshape(s * per, d).astype(np.float32),
            "step_of": np.repeat(np.arange(s), per),
            "per_step": per,
        }
    return out


def make_queries(timelines: dict, context: int, horizons, rows, label_key: str,
                 n5: int = 5, n10: int = 10, seed: int = 0):
    """Anchors shared by every H. Returns a list of dicts (one per anchor x H)."""
    hmax = max(horizons)
    queries, dropped = [], []
    for uid in sorted(timelines):
        tl = timelines[uid]
        t_len = len(tl["frames"])
        anchors = range(context - 1, t_len - hmax)
        if len(anchors) == 0:
            dropped.append(uid)
            continue
        labels = [rows[i][label_key] for i in tl["rows"]]
        distinct = sorted(set(labels))
        for t in anchors:
            for h in horizons:
                target_step = int(tl["step_of"][t + h])
                target = labels[target_step]
                others = [x for x in distinct if x != target]
                rng = np.random.default_rng(zlib.crc32(f"{seed}|{uid}|{t}|{h}".encode()))
                perm = list(rng.permutation(len(others)))
                c5 = [target] + [others[j] for j in perm[: n5 - 1]] if len(others) >= n5 - 1 else None
                c10 = [target] + [others[j] for j in perm[: n10 - 1]] if len(others) >= n10 - 1 else None
                queries.append({
                    "video_uid": uid, "t": t, "H": h,
                    "anchor_step": int(tl["step_of"][t]), "target_step": target_step,
                    "same_step": int(tl["step_of"][t]) == target_step,
                    "target": target, "cands5": c5, "cands10": c10, "gallery": distinct,
                })
    return queries, dropped


def query_vectors(timelines: dict, queries: list[dict], context: int, centering: str,
                  mu_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Context query and oracle (target step frames) query, both (Q, D)."""
    q = np.zeros((len(queries), mu_frame.shape[0]), dtype=np.float32)
    o = np.zeros_like(q)
    for n, item in enumerate(queries):
        tl = timelines[item["video_uid"]]
        f, t, per = tl["frames"], item["t"], tl["per_step"]
        start = t - context + 1
        window = f[start : t + 1]
        target = f[item["target_step"] * per : (item["target_step"] + 1) * per]
        if centering == "C0":
            ref = np.zeros_like(mu_frame)
        elif centering == "C1" or start == 0:
            ref = mu_frame
        elif centering == "C2":
            ref = f[:start].mean(axis=0)  # strictly before the context window
        else:
            raise ValueError(centering)
        q[n] = normalize(normalize(window - ref).mean(axis=0))
        o[n] = normalize(normalize(target - ref).mean(axis=0))
    return q, o


def score_queries(qvec: np.ndarray, queries: list[dict], text_index: dict, text_emb: np.ndarray):
    """Per-query metrics for one set of query vectors."""
    res = {"hit5": [], "hit10": [], "rr": [], "margin": []}
    for n, item in enumerate(queries):
        def scores(cands):
            return text_emb[[text_index[c] for c in cands]] @ qvec[n]
        if item["cands5"] is None:
            res["hit5"].append(np.nan)
            res["margin"].append(np.nan)
        else:
            s = scores(item["cands5"])
            res["hit5"].append(float((s[0] > s[1:]).all()))
            top = np.sort(s)[::-1]
            res["margin"].append(float(top[0] - top[1]))
        if item["cands10"] is None:
            res["hit10"].append(np.nan)
        else:
            s = scores(item["cands10"])
            res["hit10"].append(float((s[0] > s[1:]).all()))
        g = scores(item["gallery"])
        tgt = g[item["gallery"].index(item["target"])]
        res["rr"].append(1.0 / (1 + int((g > tgt).sum())))
    return {k: np.asarray(v, dtype=np.float64) for k, v in res.items()}


def video_bootstrap(values: np.ndarray, vids: np.ndarray, b: int, seed: int = 0):
    """Mean and 95% CI, resampling videos with replacement."""
    ok = ~np.isnan(values)
    values, vids = values[ok], vids[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uniq = np.unique(vids)
    sums = np.array([values[vids == u].sum() for u in uniq])
    counts = np.array([(vids == u).sum() for u in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(b, len(uniq)))
    boot = sums[idx].sum(axis=1) / np.maximum(counts[idx].sum(axis=1), 1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def selective_accuracy(hits: np.ndarray, margin: np.ndarray, coverage: float) -> float:
    ok = ~np.isnan(hits)
    hits, margin = hits[ok], margin[ok]
    if len(hits) == 0:
        return np.nan
    keep = max(1, int(round(coverage * len(hits))))
    order = np.argsort(-margin, kind="stable")
    return float(hits[order[:keep]].mean())


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def encode_texts(encoder: str, texts: list[str], batch_size: int = 32, device: str | None = None) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoProcessor

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoProcessor.from_pretrained(encoder, token=False)
    model = AutoModel.from_pretrained(encoder, token=False).to(device).eval()
    out = []
    for start in range(0, len(texts), batch_size):
        tokens = processor(text=[TEMPLATE.format(label=t) for t in texts[start : start + batch_size]],
                           padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.inference_mode():
            feats = model.get_text_features(**tokens)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            out.append(F.normalize(feats.float(), dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def step_seconds(rows: list[dict]) -> dict[str, float]:
    import cv2

    out = {}
    for row in rows:
        cap = cv2.VideoCapture(row["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        out[row["sample_id"]] = float(n / fps) if fps > 0 else np.nan
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "results/hoi_detection_v1/manifest.jsonl")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("/media/dhqg/d3/vlia_outputs/ladder/siglip-so400m-patch14-384"))
    p.add_argument("--encoder", default="google/siglip-so400m-patch14-384")
    p.add_argument("--out", type=Path, default=ROOT / "results/anticipation_horizon_v1")
    p.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    p.add_argument("--context", type=int, default=8, help="context frames C (8 = one step)")
    p.add_argument("--labels", nargs="+", default=["what", "why"], choices=["what", "why"])
    p.add_argument("--centerings", nargs="+", default=["C2", "C0"], choices=["C0", "C1", "C2"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-root-check", action="store_true",
                   help="read clip durations with OpenCV to report horizons in seconds")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (args.out / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{args.out}/summary.csv exists (use --overwrite)")
    rows = read_jsonl(args.manifest)
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("manifest must contain train/val rows only")
    ids = json.loads((args.cache_dir / "sample_ids.json").read_text())
    pos = {sid: i for i, sid in enumerate(ids)}
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in pos]
    if missing:
        raise ValueError(f"{len(missing)} manifest rows not in cache, e.g. {missing[:3]}")
    frames = np.load(args.cache_dir / "frame_emb.npy", mmap_mode="r")
    frames = np.asarray(frames[[pos[r["sample_id"]] for r in rows]], dtype=np.float32)
    frames = normalize(frames)
    print(f"frames {frames.shape} from {args.cache_dir}", flush=True)

    timelines = build_timelines(rows, frames)
    mu_frame = frames.reshape(-1, frames.shape[-1]).mean(axis=0)
    seconds = step_seconds(rows) if args.video_root_check else None

    all_labels = sorted({r[k] for r in rows for k in args.labels})
    text_emb = encode_texts(args.encoder, all_labels)
    text_index = {t: i for i, t in enumerate(all_labels)}
    print(f"encoded {len(all_labels)} label texts", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary, per_query_rows = [], []
    for label_key in args.labels:
        queries, dropped = make_queries(timelines, args.context, args.horizons, rows, label_key, seed=args.seed)
        if dropped:
            print(f"[{label_key}] dropped {len(dropped)} short videos: {dropped}", flush=True)
        vids = np.array([q["video_uid"] for q in queries])
        hs = np.array([q["H"] for q in queries])
        same = np.array([q["same_step"] for q in queries])
        ahead = np.array([q["target_step"] - q["anchor_step"] for q in queries], dtype=float)
        if seconds is not None:
            cum = {}
            for uid, tl in timelines.items():
                per = tl["per_step"]
                sec = np.repeat([seconds[rows[i]["sample_id"]] / per for i in tl["rows"]], per)
                cum[uid] = np.concatenate([[0.0], np.cumsum(sec)])
            secs = np.array([cum[q["video_uid"]][q["t"] + q["H"] + 1] - cum[q["video_uid"]][q["t"] + 1]
                             for q in queries])
        for centering in args.centerings:
            qv, ov = query_vectors(timelines, queries, args.context, centering, mu_frame)
            m = score_queries(qv, queries, text_index, text_emb)
            mo = score_queries(ov, queries, text_index, text_emb)
            for h in args.horizons:
                sel = hs == h
                for subset, mask in (("all", sel), ("changed_step", sel & ~same), ("same_step", sel & same)):
                    if not mask.any():
                        continue
                    acc5 = video_bootstrap(m["hit5"][mask], vids[mask], args.bootstrap, args.seed)
                    acc10 = video_bootstrap(m["hit10"][mask], vids[mask], args.bootstrap, args.seed)
                    mrr = video_bootstrap(m["rr"][mask], vids[mask], args.bootstrap, args.seed)
                    orc = video_bootstrap(mo["hit5"][mask], vids[mask], args.bootstrap, args.seed)
                    summary.append({
                        "label": label_key, "centering": centering, "H": h, "subset": subset,
                        "n_queries": int(mask.sum()), "n_videos": int(len(np.unique(vids[mask]))),
                        "frac_same_step": float(same[sel].mean()) if subset == "all" else "",
                        "mean_steps_ahead": float(ahead[mask].mean()),
                        "mean_seconds_ahead": float(np.nanmean(secs[mask])) if seconds is not None else "",
                        "acc5": acc5[0], "acc5_lo": acc5[1], "acc5_hi": acc5[2],
                        "acc10": acc10[0], "acc10_lo": acc10[1], "acc10_hi": acc10[2],
                        "mrr_video": mrr[0], "mrr_video_lo": mrr[1], "mrr_video_hi": mrr[2],
                        "oracle_acc5": orc[0], "oracle_acc5_lo": orc[1], "oracle_acc5_hi": orc[2],
                        "mean_margin": float(np.nanmean(m["margin"][mask])),
                        "margin_correct": float(np.nanmean(m["margin"][mask][m["hit5"][mask] == 1])),
                        "margin_wrong": float(np.nanmean(m["margin"][mask][m["hit5"][mask] == 0])),
                        "sel_acc5_cov50": selective_accuracy(m["hit5"][mask], m["margin"][mask], 0.5),
                        "sel_acc5_cov20": selective_accuracy(m["hit5"][mask], m["margin"][mask], 0.2),
                    })
            for n, q in enumerate(queries):
                per_query_rows.append({
                    "label": label_key, "centering": centering, "video_uid": q["video_uid"], "t": q["t"],
                    "H": q["H"], "anchor_step": q["anchor_step"], "target_step": q["target_step"],
                    "same_step": int(q["same_step"]), "hit5": m["hit5"][n], "hit10": m["hit10"][n],
                    "rr": m["rr"][n], "margin": m["margin"][n], "oracle_hit5": mo["hit5"][n],
                })
            print(f"[{label_key}|{centering}] {len(queries)} queries done", flush=True)

    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with gzip.open(args.out / "per_query.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query_rows[0]))
        writer.writeheader()
        writer.writerows(per_query_rows)
    (args.out / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "template": TEMPLATE, "chance5": 0.2, "chance10": 0.1,
    }, indent=2))
    plot(summary, args.out / "horizon.png")
    print(f"saved {args.out}", flush=True)
    return 0


def plot(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted({r["label"] for r in summary})
    fig, axes = plt.subplots(2, len(labels), figsize=(5.2 * len(labels), 6.4), squeeze=False)
    for col, lab in enumerate(labels):
        ax, axm = axes[0, col], axes[1, col]
        for (cen, subset), style in {("C2", "all"): "-o", ("C2", "changed_step"): "--s",
                                     ("C0", "all"): ":^"}.items():
            rs = sorted([r for r in summary if r["label"] == lab and r["centering"] == cen
                         and r["subset"] == subset], key=lambda r: r["H"])
            if not rs:
                continue
            h = [r["H"] for r in rs]
            y = np.array([100 * r["acc5"] for r in rs])
            lo = np.array([100 * r["acc5_lo"] for r in rs])
            hi = np.array([100 * r["acc5_hi"] for r in rs])
            ax.errorbar(h, y, yerr=[y - lo, hi - y], fmt=style, capsize=2, label=f"{cen} {subset}")
            axm.plot(h, [r["mean_margin"] for r in rs], style, label=f"{cen} {subset}")
        orc = sorted([r for r in summary if r["label"] == lab and r["centering"] == "C2"
                      and r["subset"] == "all"], key=lambda r: r["H"])
        if orc:
            ax.plot([r["H"] for r in orc], [100 * r["oracle_acc5"] for r in orc], "k-", lw=0.8,
                    label="oracle (sees target step)")
        ax.axhline(20, ls="--", c="gray", lw=0.8, label="chance 20%")
        for a in (ax, axm):
            a.set_xscale("log", base=2)
            a.set_xlabel("horizon H (cache frames; 8 = one step)")
        ax.set_ylabel("5-way accuracy (%)")
        axm.set_ylabel("mean top1-top2 margin")
        ax.set_title(f"label = {lab.upper()}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    raise SystemExit(main())
