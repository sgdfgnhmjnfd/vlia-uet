#!/usr/bin/env python3
"""Stage 14b Part D: W writes the action on EPIC-KITCHENS-100 (subset); early recognition and anticipation.

Subcommands: selftest | units | extract | train | write | svm | qwen | analyze.
Pre-registration: results/prereg_stage14b_epic.md. Videos, frames and features stay in /media/dhqg/d3 and are never committed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_scene_cv as CV  # noqa: E402
import run_stage14_generate as S14  # noqa: E402
import run_stage14b_writer as WR  # noqa: E402

EPIC = Path("/media/dhqg/d3/data/epic")
ANN = EPIC / "annotations"
VID_ROOT = EPIC / "EPIC-KITCHENS"
WORK = Path("/media/dhqg/d3/vlia_outputs/stage14b_epic")
RES = ROOT / "results/stage14b_epic"
EARLY_P = (25, 50, 75, 100)
AHEAD_G = (0.0, 0.5, 1.0)
CONDS = [f"early_p{p}" for p in EARLY_P] + [f"ahead_g{g}" for g in AHEAD_G]
N_UNITS = 4000
N_QWEN = 500
IMG_W, IMG_H = 448, 252
PROMPT_EARLY = "These frames are from a first-person video in a kitchen. In two or three words, verb then object, say what the person is doing."
PROMPT_AHEAD = "These frames are from a first-person video in a kitchen. In two or three words, verb then object, say what the person is about to do next."
log = S14.log


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def cond_times(start: float, stop: float, fps: float, cond: str) -> np.ndarray | None:
    """8 timestamps (s) for one unit and condition; None when an anticipation window starts before the video."""
    if cond.startswith("early"):
        p = int(cond.split("_p")[1])
        return np.linspace(start, start + p / 100 * (stop - start), 8)
    g = float(cond.split("_g")[1])
    lo, hi = start - g - 2.0, start - g - 1.0 / fps
    if lo < 0:
        return None
    return np.linspace(lo, hi, 8)


def seen_left(start, stop, cond):
    if cond.startswith("early"):
        p = int(cond.split("_p")[1]) / 100
        return p * (stop - start), (1 - p) * (stop - start)
    return 2.0, float(cond.split("_g")[1])


# ----------------------------------------------------------------------
# scoring (no candidate list)
# ----------------------------------------------------------------------

def norm_words(s: str) -> list[str]:
    return re.sub(r"[^\w\s-]", " ", s.lower()).split()


def load_classes():
    verbs = {int(r["id"]): ast.literal_eval(r["instances"]) for r in csv.DictReader((ANN / "EPIC_100_verb_classes.csv").open())}
    nouns = {int(r["id"]): ast.literal_eval(r["instances"]) for r in csv.DictReader((ANN / "EPIC_100_noun_classes.csv").open())}
    return verbs, nouns


def verb_ok(text: str, instances: list[str]) -> bool:
    w = norm_words(text)
    if not w:
        return False
    return any(w[0] == i or w[0] == i.split("-")[0] for i in instances)


def singulars(w: str) -> set:
    out = {w}
    if w.endswith("es"):
        out.add(w[:-2])
    if w.endswith("s"):
        out.add(w[:-1])
    return out


def noun_ok(text: str, instances: list[str]) -> bool:
    w = norm_words(text)[1:]
    singles = set().union(*[singulars(x) for x in w]) if w else set()
    pairs = {f"{a} {b}" for a, b in zip(w, w[1:])}
    for i in instances:
        if ":" in i:
            parts = i.split(":")
            if len(parts) == 2 and (f"{parts[1]} {parts[0]}" in pairs or f"{parts[0]} {parts[1]}" in pairs):
                return True
        elif i in singles:
            return True
    return False


# ----------------------------------------------------------------------
# units, folds, features
# ----------------------------------------------------------------------

def complete_videos() -> list[str]:
    out = []
    for f in sorted(VID_ROOT.glob("P*/videos/*.MP4")):
        if not Path(str(f) + ".part").exists():
            out.append(f.stem)
    return out


def cmd_units(_) -> int:
    sel = [r["video_id"] for r in csv.DictReader((EPIC / "selected_videos.csv").open())]
    done = [v for v in complete_videos() if v in sel]
    rows = [r for f in ("EPIC_100_train.csv", "EPIC_100_validation.csv") for r in csv.DictReader((ANN / f).open()) if r["video_id"] in done]
    info = {r["video_id"]: r for r in csv.DictReader((ANN / "EPIC_100_video_info.csv").open())}
    by_v = {}
    for r in rows:
        by_v.setdefault(r["video_id"], []).append(r)
    N = len(rows)
    keep = []
    if N > N_UNITS:
        quota = {v: N_UNITS * len(x) / N for v, x in by_v.items()}
        base = {v: int(np.floor(q)) for v, q in quota.items()}
        rest = N_UNITS - sum(base.values())
        for v in sorted(quota, key=lambda v: (-(quota[v] - base[v]), v))[:rest]:
            base[v] += 1
        rng = np.random.default_rng(0)
        for v in sorted(by_v):
            x = sorted(by_v[v], key=lambda r: r["narration_id"])
            keep += [x[i] for i in sorted(rng.choice(len(x), size=base[v], replace=False))]
    else:
        keep = sorted(rows, key=lambda r: (r["video_id"], r["narration_id"]))
    parts = sorted({r["participant_id"] for r in keep})
    fold_of = {p: i // 2 for i, p in enumerate(parts)}
    units = []
    for r in keep:
        fps = float(info[r["video_id"]]["fps"])
        st, sp = secs(r["start_timestamp"]), secs(r["stop_timestamp"])
        units.append({"narration_id": r["narration_id"], "participant": r["participant_id"], "video_id": r["video_id"], "start": st, "stop": sp,
                      "fps": fps, "duration_video": float(info[r["video_id"]]["duration"]), "narration": r["narration"],
                      "verb": r["verb"], "verb_class": int(r["verb_class"]), "noun": r["noun"], "noun_class": int(r["noun_class"]),
                      "fold": fold_of[r["participant_id"]],
                      "conds": [c for c in CONDS if cond_times(st, sp, fps, c) is not None]})
    rng = np.random.default_rng(0)
    q = sorted(rng.choice(len(units), size=min(N_QWEN, len(units)), replace=False).tolist())
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "units.json").write_text(json.dumps({"units": units, "videos": done, "participants": parts, "fold_of": fold_of, "qwen_units": q,
                                                 "n_segments_available": N}))
    log(f"{len(done)} complete videos, {len(parts)} participants, {N} segments, {len(units)} units, folds {fold_of}")
    return 0


def load_units():
    return json.loads((WORK / "units.json").read_text())


def cmd_extract(_) -> int:
    """One sequential decode per video; SigLIP embedding of every needed frame; 448x252 images kept for the Qwen units only."""
    import av
    import torch

    import extract_ladder_embeddings as X

    U = load_units()
    units = U["units"]
    qset = set(U["qwen_units"])
    (WORK / "feat").mkdir(parents=True, exist_ok=True)
    (WORK / "qwen_frames").mkdir(parents=True, exist_ok=True)
    model, processor = None, None
    for vid in U["videos"]:
        out = WORK / "feat" / f"{vid}.npz"
        if out.exists():
            continue
        if model is None:
            model, processor = X.load_encoder(WR.S14.ENCODER, torch.device("cuda"))
        mine = [(i, u) for i, u in enumerate(units) if u["video_id"] == vid]
        fps = mine[0][1]["fps"]
        need = set()
        for i, u in mine:
            for c in u["conds"]:
                need |= {int(round(t * fps)) for t in cond_times(u["start"], u["stop"], fps, c)}
        qneed = set()
        for i, u in mine:
            if i in qset:
                for c in ("early_p100", "ahead_g1.0"):
                    if c in u["conds"]:
                        qneed |= {int(round(t * fps)) for t in cond_times(u["start"], u["stop"], fps, c)}
        path = next(VID_ROOT.glob(f"*/videos/{vid}.MP4"))
        t0 = time.time()
        emb, imgs, buf, bidx = {}, {}, [], []

        def flush():
            if buf:
                e = X.image_features(model, processor, buf, torch.device("cuda"))
                emb.update(zip(bidx, e.astype(np.float16)))
                buf.clear()
                bidx.clear()

        last = max(need)
        with av.open(str(path)) as cont:
            s = cont.streams.video[0]
            s.thread_type = "AUTO"
            for n, frame in enumerate(cont.decode(s)):
                if n in need:
                    a = frame.reformat(width=IMG_W, height=IMG_H, format="rgb24").to_ndarray()
                    buf.append(a)
                    bidx.append(n)
                    if n in qneed:
                        imgs[n] = a
                    if len(buf) >= 64:
                        flush()
                if n >= last:
                    break
        flush()
        missing = need - set(emb)
        if missing:  # frames past the decodable end: use the last decoded frame
            lastn = max(emb)
            for m in missing:
                emb[m] = emb[lastn]
        keys = np.array(sorted(emb))
        np.savez(out, idx=keys, emb=np.stack([emb[k] for k in keys]), n_missing=len(missing))
        if imgs:
            np.savez(WORK / "qwen_frames" / f"{vid}.npz", idx=np.array(sorted(imgs)), img=np.stack([imgs[k] for k in sorted(imgs)]))
        log(f"{vid}: {len(need)} frames embedded in {time.time() - t0:.0f}s ({len(missing)} past the end)")
    return 0


def unit_feats(U):
    """feats[(i, cond)] = (8, 1152) float32."""
    cache = {}
    out = {}
    for i, u in enumerate(U["units"]):
        if u["video_id"] not in cache:
            z = np.load(WORK / "feat" / f"{u['video_id']}.npz")
            cache[u["video_id"]] = dict(zip(z["idx"].tolist(), z["emb"].astype(np.float32)))
        e = cache[u["video_id"]]
        for c in u["conds"]:
            ix = [int(round(t * u["fps"])) for t in cond_times(u["start"], u["stop"], u["fps"], c)]
            out[(i, c)] = np.stack([e[j] for j in ix])
    return out


def tag_of(c: str) -> str:
    return ("early" if c.startswith("early") else "ahead") + " act"


def folds(U):
    return sorted(set(u["fold"] for u in U["units"]))


def cmd_train(_) -> int:
    U = load_units()
    F = unit_feats(U)
    units = U["units"]
    (WORK / "w").mkdir(parents=True, exist_ok=True)
    w = None
    for f in folds(U):
        path = WORK / "w" / f"fold{f}.pt"
        if path.exists():
            continue
        w = w or WR.W()
        tr = [i for i, u in enumerate(units) if u["fold"] != f]
        assert not {units[i]["participant"] for i in tr} & {u["participant"] for u in units if u["fold"] == f}
        val_v = WR.video_val([units[i]["video_id"] for i in tr])
        va = [i for i in tr if units[i]["video_id"] in val_v]
        tr = [i for i in tr if units[i]["video_id"] not in val_v]

        def exs(idx, rng):
            out = []
            for i in idx:
                c = units[i]["conds"][int(rng.integers(len(units[i]["conds"])))]
                out.append((F[(i, c)], tag_of(c), units[i]["narration"]))
            return out

        val_exs = exs(va, np.random.default_rng(0))
        t0 = time.time()
        info = w.fit(lambda ep, rng: exs(tr, rng), val_exs, 0, f"epic fold {f}")
        mu = np.stack([F[(i, c)] for i in tr for c in units[i]["conds"]]).reshape(-1, WR.D_IN).mean(0)
        info.update({"fold": f, "n_train": len(tr), "n_val": len(va), "seconds": time.time() - t0, "mean_train_emb": mu.tolist()})
        w.torch.save({"state": w.state(), "info": info}, path)
    return 0


def cmd_write(_) -> int:
    out = WORK / "written.json"
    if out.exists():
        return 0
    U = load_units()
    F = unit_feats(U)
    units = U["units"]
    w = WR.W()
    rec, t0 = {}, time.time()
    for f in folds(U):
        ck = w.torch.load(WORK / "w" / f"fold{f}.pt", weights_only=False)
        w.load(ck["state"])
        mu = np.asarray(ck["info"]["mean_train_emb"], np.float32)
        te = [i for i, u in enumerate(units) if u["fold"] == f]
        for c in CONDS:
            idx = [i for i in te if c in units[i]["conds"]]
            for i, s in zip(idx, w.generate([F[(i, c)] for i in idx], tag_of(c))):
                rec[f"{i}|{c}"] = s
            rec[f"blank|{f}|{c}"] = w.generate([np.tile(mu, (8, 1))], tag_of(c))[0]
        log(f"epic fold {f} written")
    out.write_text(json.dumps({"written": rec, "gpu_seconds": time.time() - t0}))
    return 0


def cmd_svm(_) -> int:
    out = WORK / "svm.json"
    if out.exists():
        return 0
    import run_stage15_supervised as S15

    U = load_units()
    F = unit_feats(U)
    units = U["units"]
    pred = {}
    for f in folds(U):
        for c in CONDS:
            tr = [i for i, u in enumerate(units) if u["fold"] != f and c in u["conds"]]
            te = [i for i, u in enumerate(units) if u["fold"] == f and c in u["conds"]]
            Xtr = S14.normalize(np.stack([F[(i, c)].mean(0) for i in tr]))
            Xte = S14.normalize(np.stack([F[(i, c)].mean(0) for i in te]))
            for kind in ("verb_class", "noun_class"):
                cls = sorted({units[i][kind] for i in tr})
                m = {k: j for j, k in enumerate(cls)}
                svm = S15.LinearSVM(1.0, len(cls)).fit(Xtr, [m[units[i][kind]] for i in tr])
                p = np.asarray(cls)[np.argmax(svm.scores(Xte), axis=1)]
                for i, y in zip(te, p):
                    pred.setdefault(f"{i}|{c}", {})[kind] = int(y)
        log(f"svm fold {f} done")
    out.write_text(json.dumps(pred))
    return 0


def cmd_qwen(_) -> int:
    out = WORK / "qwen.json"
    U = load_units()
    units = U["units"]
    rec = json.loads(out.read_text()) if out.exists() else {}
    gen = None
    cache = {}
    t0 = time.time()
    from PIL import Image

    for i in U["qwen_units"]:
        u = units[i]
        for c, prompt in (("early_p100", PROMPT_EARLY), ("ahead_g1.0", PROMPT_AHEAD)):
            if c not in u["conds"] or f"{i}|{c}" in rec:
                continue
            if u["video_id"] not in cache:
                z = np.load(WORK / "qwen_frames" / f"{u['video_id']}.npz")
                cache = {u["video_id"]: dict(zip(z["idx"].tolist(), z["img"]))}
            imgs = [Image.fromarray(cache[u["video_id"]][int(round(t * u["fps"]))])
                    for t in cond_times(u["start"], u["stop"], u["fps"], c)]
            gen = gen or S14.Generator()
            s, raw, _ = gen(imgs, prompt)
            rec[f"{i}|{c}"] = s
        if len(rec) % 50 == 0:
            out.write_text(json.dumps(rec))
    out.write_text(json.dumps(rec))
    (WORK / "qwen_seconds.json").write_text(json.dumps({"seconds": time.time() - t0}))
    return 0


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def cmd_analyze(_) -> int:
    U = load_units()
    units = U["units"]
    F = unit_feats(U)
    verbs, nouns = load_classes()
    wr = json.loads((WORK / "written.json").read_text())["written"]
    svm = json.loads((WORK / "svm.json").read_text())
    qw = json.loads((WORK / "qwen.json").read_text()) if (WORK / "qwen.json").exists() else {}
    part = np.array([u["participant"] for u in units])
    mf, nn = {}, {}
    for f in folds(U):
        tr = [i for i, u in enumerate(units) if u["fold"] != f]
        vals, cnt = np.unique([units[i]["narration"] for i in tr], return_counts=True)
        mf[f] = sorted(v for v, c in zip(vals, cnt) if c == cnt.max())[0]
        for c in CONDS:
            trc = [i for i in tr if c in units[i]["conds"]]
            te = [i for i, u in enumerate(units) if u["fold"] == f and c in u["conds"]]
            A = S14.normalize(np.stack([F[(i, c)].mean(0) for i in trc]))
            B = S14.normalize(np.stack([F[(i, c)].mean(0) for i in te]))
            for i, j in zip(te, np.argmax(B @ A.T, axis=1)):
                nn[(i, c)] = units[trc[j]]["narration"]

    def sc(i, text):
        u = units[i]
        v, n = verb_ok(text, verbs[u["verb_class"]]), noun_ok(text, nouns[u["noun_class"]])
        return {"verb": float(v), "noun": float(n), "action": float(v and n), "f1": WR.token_f1(text, u["narration"])}

    rows, per = [], {}
    for c in CONDS:
        idx = [i for i, u in enumerate(units) if c in u["conds"]]
        sl = np.array([seen_left(units[i]["start"], units[i]["stop"], c) for i in idx])
        models = {"W": [sc(i, wr[f"{i}|{c}"]) for i in idx],
                  "W-blank": [sc(i, wr[f"blank|{units[i]['fold']}|{c}"]) for i in idx],
                  "most-frequent": [sc(i, mf[units[i]["fold"]]) for i in idx],
                  "nearest-segment": [sc(i, nn[(i, c)]) for i in idx]}
        sv = [{"verb": float(svm[f"{i}|{c}"]["verb_class"] == units[i]["verb_class"]), "noun": float(svm[f"{i}|{c}"]["noun_class"] == units[i]["noun_class"])} for i in idx]
        models["SVM reference (classes)"] = [dict(x, action=float(x["verb"] and x["noun"]), f1=np.nan) for x in sv]
        qidx = [i for i in idx if f"{i}|{c}" in qw]
        if qidx:
            models["Qwen (500 units)"] = [sc(i, qw[f"{i}|{c}"]) for i in qidx]
        for m, vals in models.items():
            ii = qidx if m.startswith("Qwen") else idx
            per[(c, m)] = dict(zip(ii, vals))
            row = {"condition": c, "model": m, "n": len(ii), "seconds_seen": float(sl[:, 0].mean()),
                   ("seconds_to_end" if c.startswith("early") else "seconds_to_start"): float(sl[:, 1].mean())}
            for k in ("verb", "noun", "action", "f1"):
                v = np.array([x[k] for x in vals], float)
                if np.isnan(v).all():
                    continue
                a = CV.video_bootstrap(part[ii], v, np.ones(len(v)))
                row.update({k: a[0], f"{k}_lo": a[1], f"{k}_hi": a[2]})
            rows.append(row)
    RES.mkdir(parents=True, exist_ok=True)
    S14.write_csv(RES / "epic_writer.csv", rows)
    # rules
    idx = [i for i, u in enumerate(units) if "early_p100" in u["conds"]]
    w100 = np.array([per[("early_p100", "W")][i]["action"] for i in idx])
    mf100 = np.array([per[("early_p100", "most-frequent")][i]["action"] for i in idx])
    w25 = np.array([per[("early_p25", "W")][i]["action"] for i in idx])
    e1 = CV.video_bootstrap(part[idx], w100 - mf100, np.ones(len(idx)))
    e2 = CV.video_bootstrap(part[idx], w100 - w25, np.ones(len(idx)))
    rules = {"E-1": {"rule": "action(W, p=100) - action(most frequent), CI lower bound > 0", "delta": e1[0], "ci95": [e1[1], e1[2]], "pass": e1[1] > 0},
             "E-2": {"rule": "action(W, p=100) - action(W, p=25), CI lower bound > 0", "delta": e2[0], "ci95": [e2[1], e2[2]], "pass": e2[1] > 0}}
    sent = [{"unit": i, "narration_id": units[i]["narration_id"], "condition": c, "true_narration": units[i]["narration"], "written": wr[f"{i}|{c}"]}
            for i in range(len(units)) for c in units[i]["conds"]]
    S14.write_csv(RES / "sentences_epic.csv", sent)

    def f(x):
        return "–" if x is None else f"{100 * x:.1f}"

    md = ["# W on EPIC-KITCHENS-100 subset (folds over participants; scored without any candidate list)", "",
          "| condition | model | n | s seen | s to end / start | verb | noun | action | token F1 |", "|---|---|---:|---:|---:|---|---|---|---|"]
    for r in rows:
        def c(k):
            return f"{f(r[k])} [{f(r[k + '_lo'])}; {f(r[k + '_hi'])}]" if k in r else "–"
        md.append(f"| {r['condition']} | {r['model']} | {r['n']} | {r['seconds_seen']:.2f} | {r.get('seconds_to_end', r.get('seconds_to_start')):.2f} | "
                  f"{c('verb')} | {c('noun')} | {c('action')} | {c('f1')} |")
    md += ["", "| rule | delta | 95% CI | verdict |", "|---|---:|---|---|"]
    md += [f"| {k}: {r['rule']} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {'PASS' if r['pass'] else 'FAIL'} |" for k, r in rules.items()]
    (RES / "epic_writer.md").write_text("\n".join(md) + "\n")
    # data summary
    seg = np.array([u["stop"] - u["start"] for u in units])
    vids = U["videos"]
    ds = ["# Stage 14b Part D: data summary (EPIC-KITCHENS-100, CC BY-NC 4.0; no video or frame is committed)", "",
          f"- Videos used ({len(vids)}): {', '.join(vids)}", f"- Participants: {', '.join(U['participants'])}",
          f"- Segments available in these videos: {U['n_segments_available']}; units used: {len(units)}",
          f"- Distinct narrations: {len({u['narration'] for u in units})}; verb classes present: {len({u['verb_class'] for u in units})}; "
          f"noun classes present: {len({u['noun_class'] for u in units})}",
          f"- Frame rates: {sorted({round(u['fps'], 2) for u in units})}; segment length mean {seg.mean():.2f} s (min {seg.min():.2f}, max {seg.max():.2f})", "",
          "| fold | test participants | test units | train units |", "|---:|---|---:|---:|"]
    for fo in folds(U):
        tp = sorted({u["participant"] for u in units if u["fold"] == fo})
        ds.append(f"| {fo} | {', '.join(tp)} | {sum(u['fold'] == fo for u in units)} | {sum(u['fold'] != fo for u in units)} |")
    ds += ["", "| condition | units | s seen | s to end (early) / to start (ahead) |", "|---|---:|---:|---:|"]
    for c in CONDS:
        idx = [i for i, u in enumerate(units) if c in u["conds"]]
        sl = np.array([seen_left(units[i]["start"], units[i]["stop"], c) for i in idx])
        ds.append(f"| {c} | {len(idx)} | {sl[:, 0].mean():.2f} | {sl[:, 1].mean():.2f} |")
    (RES / "data_summary.md").write_text("\n".join(ds) + "\n")
    w_info = [__import__("torch").load(WORK / "w" / f"fold{fo}.pt", weights_only=False)["info"] for fo in folds(U)]
    gpu = sum(i["seconds"] for i in w_info) + json.loads((WORK / "written.json").read_text())["gpu_seconds"]
    if (WORK / "qwen_seconds.json").exists():
        gpu += json.loads((WORK / "qwen_seconds.json").read_text())["seconds"]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (RES / "config.json").write_text(json.dumps({"git_commit": commit, "W": "same as Part B (SmolVLM2-500M LM + connector)", "tags": ["early act:", "ahead act:"],
                                                 "epochs_per_fold": [i["epochs_run"] for i in w_info], "gpu_minutes_train_write_qwen": gpu / 60,
                                                 "svm": "Stage 15 LinearSVM, C=1, mean frame embedding", "qwen_prompts": [PROMPT_EARLY, PROMPT_AHEAD],
                                                 "rules": rules, "first10": sent[:10]}, indent=1, default=float))
    print("\n".join(md))
    return 0


def cmd_selftest(_) -> int:
    verbs, nouns = load_classes()
    assert verb_ok("open door", verbs[3]) and not verb_ok("close door", verbs[3])
    assert noun_ok("open door", nouns[3]) and noun_ok("open cupboards", nouns[3]) and not noun_ok("open fridge", nouns[3])
    assert noun_ok("turn on water tap", nouns[0]) and noun_ok("rinse the plates", nouns[2])
    assert verb_ok("pick up spoon", verbs[0]) and noun_ok("pick up spoon", nouns[1])
    t = cond_times(10.0, 12.0, 50.0, "ahead_g1.0")
    assert t[-1] < 10.0 - 1.0 and np.isclose(t[0], 7.0)
    assert cond_times(2.5, 4.0, 50.0, "ahead_g1.0") is None and cond_times(2.5, 4.0, 50.0, "ahead_g0.5") is not None
    assert cond_times(0.0, 4.0, 50.0, "ahead_g0.0") is None
    e = cond_times(10.0, 14.0, 50.0, "early_p50")
    assert np.isclose(e[0], 10.0) and np.isclose(e[-1], 12.0)
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "units", "extract", "train", "write", "svm", "qwen", "analyze"])
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "units": cmd_units, "extract": cmd_extract, "train": cmd_train, "write": cmd_write, "svm": cmd_svm,
            "qwen": cmd_qwen, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
