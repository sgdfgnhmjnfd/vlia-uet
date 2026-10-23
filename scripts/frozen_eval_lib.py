#!/usr/bin/env python3
"""One-shot locked-test evaluation of the frozen pipeline (prereg v2, Stage 1).

`evaluate_frozen_test(frozen, test_manifest, output_dir)` evaluates the frozen
pipeline (FROZEN.json) and the baseline on the 407-sample / 4-UID candidate
test set. It is called only by `scripts/final_frozen_eval.py`, which requires a
user-created approval marker and the FROZEN.json hash.

Pre-registered verdicts (results/prereg_v2.md):
  T1: frozen 5-way lift CI lower bound > 1 at >= 3 of 4 prefixes.
  T2: paired frozen - baseline 5-way delta CI lower bound > 0 at >= 2 of 4 prefixes.
Bootstrap over the 4 test videos (B=10000, seed 0) -- "exploratory: 4 clusters".

Frame loading and the encoders are injectable so the unit tests
(`scripts/test_frozen_eval_lib.py`) run on synthetic data only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ladder_v1 as L  # noqa: E402
import run_prefix_length_ablation as P  # noqa: E402
import run_scene_cv as CV  # noqa: E402

PREFIXES = (2, 4, 6, 8)
CANDIDATE_MANIFEST = ROOT / "results/test_protocol_full_v1/candidate_test_manifest.csv"
LOCKED_PROTOCOL = ROOT / "results/test_protocol_full_v1/locked_protocol.json"
TRAIN_VAL_MANIFEST = ROOT / "results/hoi_detection_v1/manifest.jsonl"
CLUSTER_LABEL = "exploratory: 4 clusters"
STEP_RE = re.compile(r"step(\d+)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# test rows
# ----------------------------------------------------------------------

def load_test_rows(test_manifest: Path, protocol: dict, train_val_ids: set[str],
                   expected_n: int | None = 407) -> list[dict]:
    """Candidate test rows; refuses any train/val sample id or UID and the
    excluded UID."""
    with Path(test_manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id in test manifest")
    leaked = sorted(set(ids) & train_val_ids)
    if leaked:
        raise ValueError(f"test manifest contains train/val sample ids: {leaked[:3]}")
    uids = {r["video_uid"] for r in rows}
    forbidden = (set(protocol["train_video_uids"]) | set(protocol["validation_video_uids"])
                 | set(protocol["excluded_test_video_uids"]))
    if uids & forbidden:
        raise ValueError(f"test manifest contains non-candidate UIDs: {sorted(uids & forbidden)}")
    if uids != set(protocol["candidate_test_video_uids"]):
        raise ValueError("test UIDs differ from the locked candidate UIDs")
    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"expected {expected_n} test samples, got {len(rows)}")
    root = Path(protocol["dataset_root"])
    out = []
    for r in rows:
        m = STEP_RE.search(r["sample_id"])
        if m is None:
            raise ValueError(f"cannot parse step from {r['sample_id']}")
        out.append({**r, "step": int(m.group(1)), "video_path": str(root / f"{r['sample_id']}.mp4")})
    return out


# ----------------------------------------------------------------------
# frozen pipeline pieces (numpy)
# ----------------------------------------------------------------------

def frozen_process(frames, texts, vids, steps, mu_f, mu_t):
    """C2 with test-video context only; C1 fallback with the frozen train+val
    means. Returns (frames', text_raw, text_c1, use_c1_text)."""
    ctx, has = L.earlier_context(frames, list(vids), np.asarray(steps))
    f1 = L.normalize(frames - mu_f)
    f2 = L.normalize(frames - ctx[:, None, :])
    out = np.where(has[:, None, None], f2, f1)
    t1 = L.normalize(texts - mu_t)
    return out, texts, t1, ~has


def per_query_metrics(fr, ta, tb, ub, text_id, cands5, cands10, k, agg="A0"):
    idx = np.arange(len(fr))
    gs = L.gallery_scores(fr, ta, tb, ub, idx, agg, k)
    ranks = P.first_positive_rank(gs, text_id[:, None] == text_id[None, :])
    v5 = (cands5 >= 0).all(axis=1)
    v10 = (cands10 >= 0).all(axis=1)
    s5 = L.candidate_scores(fr, ta, tb, ub, idx, cands5, agg, k)
    s10 = L.candidate_scores(fr, ta, tb, ub, idx, cands10, agg, k)
    return {
        "rr": 1.0 / ranks, "top1": (ranks <= 1).astype(float), "top3": (ranks <= 3).astype(float),
        "kway5": L.strict_hits(s5, v5).astype(float), "kway10": L.strict_hits(s10, v10).astype(float),
    }


def control_donors(sample_ids, scenes, vids):
    """Every 8th query (sorted sample_id); donor from a different scene, else
    a different video (fallback recorded)."""
    order = sorted(range(len(sample_ids)), key=lambda i: sample_ids[i])
    ctrl = order[::8]
    donors, fallback = [], 0
    for i in ctrl:
        pool = [j for j in range(len(sample_ids)) if scenes[j] != scenes[i]]
        if not pool:
            pool = [j for j in range(len(sample_ids)) if vids[j] != vids[i]]
            fallback += 1
        rng = np.random.RandomState(zlib.crc32(f"{sample_ids[i]}|control".encode()) & 0xFFFFFFFF)
        donors.append(pool[rng.randint(len(pool))])
    return np.asarray(ctrl), np.asarray(donors), fallback


# ----------------------------------------------------------------------
# real encoders (used only by the one-shot run)
# ----------------------------------------------------------------------

def real_encoder(encoder_id: str, device):
    import torch

    import extract_ladder_embeddings as X

    model, processor = X.load_encoder(encoder_id, device)

    def image_fn(frames_batch: list[np.ndarray]) -> np.ndarray:
        pv = torch.cat([X.pixel_values(processor, np.stack(frames_batch[i:i + 8]))
                        for i in range(0, len(frames_batch), 8)])
        return X.encode_pixels(model, pv, device)

    def text_fn(texts: list[str]) -> tuple[np.ndarray, int]:
        return X.text_features(model, processor, encoder_id, texts, device, 32)

    return image_fn, text_fn


def embed(rows, frame_loader, image_fn, text_fn, template, frame_dtype):
    feats = []
    for start in range(0, len(rows), 4):  # 4 samples x 8 frames = 32-frame batches, as in the cache
        frames = []
        for r in rows[start:start + 4]:
            f = frame_loader(r["video_path"])
            if f.shape[0] != 8:
                raise AssertionError(f"{r['sample_id']}: {f.shape[0]} frames")
            frames.extend(list(f))
        feats.append(image_fn(frames))
    fr = np.concatenate(feats).reshape(len(rows), 8, -1).astype(frame_dtype).astype(np.float32)
    uniq = sorted({r["why"] for r in rows})
    temb, truncated = text_fn([template.format(why=t) for t in uniq])
    pos = {t: i for i, t in enumerate(uniq)}
    return fr, temb[[pos[r["why"]] for r in rows]].astype(np.float32), truncated


# ----------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------

def evaluate_frozen_test(frozen: dict, test_manifest: Path, output_dir: Path, *,
                         frame_loader: Callable | None = None, encoder_factory: Callable | None = None,
                         protocol: dict | None = None, train_val_ids: set[str] | None = None,
                         expected_n: int | None = 407, bootstrap_B: int = 10000) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot output: {output_dir}")
    if protocol is None:
        protocol = json.loads(LOCKED_PROTOCOL.read_text())
    if train_val_ids is None:
        train_val_ids = {json.loads(line)["sample_id"] for line in TRAIN_VAL_MANIFEST.open() if line.strip()}
    rows = load_test_rows(test_manifest, protocol, train_val_ids, expected_n)
    output_dir.mkdir(parents=True)

    if frame_loader is None:
        from hoi_utils import sample_exact_video_frames as frame_loader
    if encoder_factory is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_factory = lambda name: real_encoder(name, device)  # noqa: E731

    sample_ids = [r["sample_id"] for r in rows]
    vids = np.asarray([r["video_uid"] for r in rows])
    scenes = np.asarray([r["scene"] for r in rows])
    steps = np.asarray([r["step"] for r in rows])
    texts = [r["why"] for r in rows]
    vocab = {t: i for i, t in enumerate(sorted(set(texts)))}
    text_id = np.asarray([vocab[t] for t in texts])
    cands5 = CV.build_kway(vids.tolist(), texts, sample_ids, 5)
    cands10 = CV.build_kway(vids.tolist(), texts, sample_ids, 10)
    chance = CV.gallery_chance(text_id, np.zeros(len(rows), dtype=int))

    pipes = {}
    trunc = {}
    # frozen pipeline
    img, txt = encoder_factory(frozen["encoder_id"])
    fdtype = np.float16 if frozen["frame_dtype"] == "float16" else np.float32
    fr, tt, trunc["frozen"] = embed(rows, frame_loader, img, txt, frozen["template_text"], fdtype)
    means = np.load(ROOT / frozen["c1_means_file"]) if not Path(frozen["c1_means_file"]).is_absolute() \
        else np.load(frozen["c1_means_file"])
    assert frozen["aggregator"] == "A0" and frozen["centering"] == "C2" and float(frozen["beta"]) == 0.0
    pipes["frozen"] = frozen_process(fr, tt, vids, steps, means["mu_frame"], means["mu_text"])
    c2_fallback = float(pipes["frozen"][3].mean())
    del img, txt
    # baseline
    base = frozen["baseline"]
    img, txt = encoder_factory(base["encoder_id"])
    bfr, btt, trunc["baseline"] = embed(rows, frame_loader, img, txt, base["template_text"], np.float32)
    pipes["baseline"] = (bfr, btt, btt, np.zeros(len(rows), bool))
    del img, txt

    per_q = {name: {k: per_query_metrics(*p, text_id, cands5, cands10, k) for k in PREFIXES}
             for name, p in pipes.items()}
    B = bootstrap_B
    n = len(rows)
    summary, paired = [], []
    for name in pipes:
        for k in PREFIXES:
            m = per_q[name][k]
            row = {"pipeline": name, "k": k, "n_queries": n, "n_videos": int(len(np.unique(vids))),
                   "MRR": m["rr"].mean(), "MRR_chance": chance["MRR"].mean(), "Top1": m["top1"].mean(),
                   "Top1_chance": chance["Top1"].mean(), "Top3": m["top3"].mean(), "Top3_chance": chance["Top3"].mean(),
                   "5way_acc": m["kway5"].mean(), "10way_acc": m["kway10"].mean(),
                   "n_valid_5way": int((cands5 >= 0).all(1).sum()), "n_valid_10way": int((cands10 >= 0).all(1).sum()),
                   "bootstrap": CLUSTER_LABEL}
            for metric, num, den in (("MRR", m["rr"], chance["MRR"]), ("5way", m["kway5"], np.full(n, 0.2)),
                                     ("10way", m["kway10"], np.full(n, 0.1))):
                pt, lo, hi = CV.video_bootstrap(vids, num, den, B=B)
                row[f"{metric}_lift"], row[f"{metric}_lift_ci_lo"], row[f"{metric}_lift_ci_hi"] = pt, lo, hi
            summary.append({key: (float(v) if isinstance(v, (np.floating,)) else v) for key, v in row.items()})
    for k in PREFIXES:
        for metric in ("rr", "kway5", "kway10"):
            d = per_q["frozen"][k][metric] - per_q["baseline"][k][metric]
            pt, lo, hi = CV.video_bootstrap(vids, d, np.ones(n), B=B)
            paired.append({"k": k, "metric": "MRR" if metric == "rr" else metric, "delta_frozen_minus_baseline": pt,
                           "ci95_lo": lo, "ci95_hi": hi, "bootstrap": CLUSTER_LABEL})

    ctrl, donors, fallback = control_donors(sample_ids, scenes, vids)
    control = []
    for name, (f, ta, tb, ub) in pipes.items():
        real = L.strict_hits(L.candidate_scores(f, ta, tb, ub, ctrl, cands5[ctrl], "A0", 8),
                             (cands5[ctrl] >= 0).all(1))
        safe = np.where(cands5[ctrl] >= 0, cands5[ctrl], 0)
        texts_c = np.where(ub[donors][:, None, None], tb[safe], ta[safe])
        fake = L.strict_hits(L.aggregate(f[donors], texts_c, "A0", 8), (cands5[ctrl] >= 0).all(1))
        control.append({"pipeline": name, "k": 8, "n_control": len(ctrl), "real_5way_acc": float(real.mean()),
                        "control_5way_acc": float(fake.mean()), "donor_scene_fallbacks": fallback})

    fz = [r for r in summary if r["pipeline"] == "frozen"]
    t1_pref = [r["k"] for r in fz if r["5way_lift_ci_lo"] > 1]
    t2_pref = [r["k"] for r in paired if r["metric"] == "kway5" and r["ci95_lo"] > 0]
    verdicts = {
        "T1": {"pass": len(t1_pref) >= 3, "prefixes": t1_pref,
               "rule": "frozen 5-way lift CI lower bound > 1 at >= 3 of 4 prefixes"},
        "T2": {"pass": len(t2_pref) >= 2, "prefixes": t2_pref,
               "rule": "paired frozen - baseline 5-way delta CI lower bound > 0 at >= 2 of 4 prefixes; "
                       "with 4 clusters a FAIL is expected to be uninformative"},
        "bootstrap": CLUSTER_LABEL,
    }

    def write(name, table):
        with (output_dir / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)

    write("summary.csv", summary)
    write("paired.csv", paired)
    write("control.csv", control)
    per_example = []
    for i in range(n):
        rec = {"sample_id": sample_ids[i], "video_uid": vids[i], "scene": scenes[i], "step": int(steps[i]),
               "c2_fallback": bool(pipes["frozen"][3][i])}
        for name in pipes:
            for k in PREFIXES:
                for metric in ("rr", "kway5", "kway10"):
                    rec[f"{name}_k{k}_{metric}"] = float(per_q[name][k][metric][i])
        per_example.append(rec)
    write("per_example.csv", per_example)
    lines = ["# Locked candidate test: frozen pipeline vs baseline", "",
             f"{n} test samples, {len(np.unique(vids))} videos. Bootstrap over videos: **{CLUSTER_LABEL}** (B={B}, seed 0).", "",
             "| pipeline | k | MRR (chance) | MRR lift [CI] | Top-1 | Top-3 | 5-way | 5-way lift [CI] | 10-way | 10-way lift [CI] |",
             "|---|---:|---|---|---:|---:|---:|---|---:|---|"]
    for r in summary:
        lines.append(f"| {r['pipeline']} | {r['k']} | {r['MRR']:.4f} ({r['MRR_chance']:.4f}) | {r['MRR_lift']:.3f} "
                     f"[{r['MRR_lift_ci_lo']:.3f}, {r['MRR_lift_ci_hi']:.3f}] | {r['Top1']:.4f} | {r['Top3']:.4f} | "
                     f"{r['5way_acc']:.4f} | {r['5way_lift']:.3f} [{r['5way_lift_ci_lo']:.3f}, {r['5way_lift_ci_hi']:.3f}] | "
                     f"{r['10way_acc']:.4f} | {r['10way_lift']:.3f} [{r['10way_lift_ci_lo']:.3f}, {r['10way_lift_ci_hi']:.3f}] |")
    lines += ["", "| k | metric | frozen − baseline [CI] |", "|---:|---|---|"]
    for r in paired:
        lines.append(f"| {r['k']} | {r['metric']} | {r['delta_frozen_minus_baseline']:+.4f} [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] |")
    lines += ["", "| pipeline | n control | real 5-way | control 5-way |", "|---|---:|---:|---:|"]
    for r in control:
        lines.append(f"| {r['pipeline']} | {r['n_control']} | {r['real_5way_acc']:.4f} | {r['control_5way_acc']:.4f} |")
    lines += ["", "## Verdicts (pre-registered)", "", "```", json.dumps(verdicts, indent=2), "```", ""]
    (output_dir / "summary.md").write_text("\n".join(lines))
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    config = {
        "frozen": frozen, "test_manifest": str(test_manifest), "n_samples": n,
        "video_uids": sorted(set(vids.tolist())), "c2_fallback_fraction": c2_fallback,
        "text_truncations": trunc, "control_donor_scene_fallbacks": fallback, "verdicts": verdicts,
        "bootstrap": {"B": B, "seed": 0, "cluster": "video_uid", "label": CLUSTER_LABEL},
        "input_sha256": {"test_manifest": sha256(test_manifest)}, "git_commit": commit,
    }
    for key in ("c1_means_file",):
        p = Path(frozen[key]) if Path(frozen[key]).is_absolute() else ROOT / frozen[key]
        config["input_sha256"][key] = sha256(p)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return {"verdicts": verdicts, "summary": summary, "paired": paired, "control": control}
