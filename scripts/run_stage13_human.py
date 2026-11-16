#!/usr/bin/env python3
"""Stage 13: a real person starts a task -> F picks the goal among the 10 LIBERO-10 tasks -> token -> VLIA acts.

Subcommands: selftest | extract | score | plan | analyze. Pre-registration: results/prereg_stage13_human_video.md.
Rollouts run through scripts/libero_goal_ablation.py (condition E2E-human, --e2e-tokens .../e2e_human_tokens.json).
Videos, frames and contact sheets stay under /media/dhqg/d3 and are never committed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import token_matrix as TM  # noqa: E402
from run_stage12_e2e import (  # noqa: E402
    ask_threshold, cluster_bootstrap, e2e_ask, encode, normalize, query_vec, strict_correct, top2)

N_TASKS = 10
N_FRAMES = 8
REPS = (1, 2, 3)
PREFIXES = (0, 25, 50, 75, 100)
PRIMARY_P = 50
PRIMARY_PERSON = "p1"
SCENE_TASKS = {"A": (0, 1, 7), "B": (4, 6), "C": (2, 8), "D": (3, 9), "E": (5,)}
SCENE_OF = {t: s for s, ts in SCENE_TASKS.items() for t in ts}
CONTEXTS = ("C-scene", "C0", "C-libero")
TEMPLATES = {"T1": "a first-person photo of a person doing: {why}", "T0": "{why}"}
ENCODER = "google/siglip-so400m-patch14-384"
VIDEOS = Path("/media/dhqg/d3/datasets/own_libero10")
WORK = Path("/media/dhqg/d3/vlia_outputs/stage13")
RES = ROOT / "results/stage13_human_video"
TABLE = "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings_v2.npz"
LIBERO_CTX = Path("/media/dhqg/d3/vlia_outputs/stage12/scores.npz")
EVAL = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval")
VLIA = "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000/pretrained_model"
STAGE12_SUMMARY = ROOT / "results/stage12_end_to_end/summary.csv"


# ----------------------------------------------------------------------
# labels, windows, sampling (numpy only)
# ----------------------------------------------------------------------

def read_labels(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(Path(path).open()):
        rows.append({"video_id": r["video_id"].strip(), "task": int(r["task"]), "scene": r["scene"].strip(),
                     "rep": int(r["rep"]), "person": r["person"].strip(), "start_s": float(r["start_s"]), "end_s": float(r["end_s"])})
    return rows


def gate_l1(rows: list[dict], durations: dict, partial: bool = False) -> list[str]:
    """Errors of gate L1 (empty list = pass). partial=True skips the 'all 30 clips' completeness check."""
    errs = []
    prim = [r for r in rows if r["person"] == PRIMARY_PERSON]
    if not partial:
        have = {(r["task"], r["rep"]) for r in prim}
        need = {(t, rep) for t in range(N_TASKS) for rep in REPS}
        if have != need or len(prim) != len(need):
            errs.append(f"primary set: missing {sorted(need - have)}, extra {sorted(have - need)}, rows {len(prim)}")
    for r in rows:
        v = r["video_id"]
        if r["person"] == PRIMARY_PERSON and v != f"t{r['task']}_r{r['rep']}":
            errs.append(f"{v}: name does not match t{r['task']}_r{r['rep']}")
        if not 0 <= r["task"] < N_TASKS:
            errs.append(f"{v}: task {r['task']} out of range")
        elif r["scene"] != SCENE_OF[r["task"]]:
            errs.append(f"{v}: scene {r['scene']} != {SCENE_OF[r['task']]} for task {r['task']}")
        if v not in durations:
            errs.append(f"{v}: clip missing")
            continue
        if not 0 < r["start_s"] < r["end_s"] <= durations[v]:
            errs.append(f"{v}: need 0 < start_s < end_s <= duration ({r['start_s']}, {r['end_s']}, {durations[v]:.3f})")
        if r["start_s"] < 0.5:
            errs.append(f"{v}: start_s {r['start_s']} < 0.5")
    return errs


def prefix_window(start_s: float, end_s: float, p: int) -> tuple[float, float]:
    return 0.0, start_s + p / 100 * (end_s - start_s)


def switch_windows(start_s: float, switch_s: float, end_s: float) -> dict:
    return {"before": (0.0, start_s + 0.5 * (switch_s - start_s)), "after": (max(0.0, end_s - 4.0), end_s)}


def window_indices(a: float, b: float, fps: float, n_frames: int) -> np.ndarray:
    """8 frames at linspace(a, b, 8); the last one is the frame at the window end b."""
    return np.minimum(np.round(np.linspace(a, b, N_FRAMES) * fps).astype(int), n_frames - 1)


def rep_of(e: int) -> int:
    return e % 3 + 1


def loo_context(idle: dict, groups: dict, clip: str) -> np.ndarray:
    """Mean of the idle-part frame embeddings of the other clips of the clip's (scene, person) group."""
    others = [c for c in groups[clip] if c != clip]
    if not others:
        raise SystemExit(f"STOP: no other clip in the context group of {clip}")
    return np.concatenate([idle[c] for c in others]).mean(axis=0)


def masked(scores: np.ndarray, cands) -> tuple[int, float]:
    """Top-1 (as a task id) and margin among `cands` only."""
    cands = list(cands)
    y, m = top2(scores[cands]) if len(cands) > 1 else (0, float("nan"))
    return cands[y], m


def masked_correct(scores: np.ndarray, true: int, cands) -> bool:
    cands = list(cands)
    return strict_correct(scores[cands], cands.index(true))


# ----------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------

def decoder_name() -> str:
    try:
        import cv2  # noqa: F401

        return "cv2 (PY) " + cv2.__version__
    except ImportError:
        import av

        return "av (PY) " + av.__version__


def decode(path: Path, wanted=None):
    """(frames {index: RGB uint8}, fps, n_frames). Sequential read, no seeking. cv2 applies rotation metadata."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"STOP: cannot open {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames, i = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if wanted is None or i in wanted:
                frames[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
        return frames, fps, i
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        frames, i = {}, 0
        for fr in c.decode(s):
            if wanted is None or i in wanted:
                frames[i] = fr.to_ndarray(format="rgb24")
            i += 1
    return frames, fps, i


def probe(path: Path) -> dict:
    _, fps, n = decode(path, wanted=set())
    return {"fps": fps, "n_frames": n, "duration": n / fps if fps else 0.0}


def rotate(frames: np.ndarray, degrees: int) -> np.ndarray:
    """Clockwise rotation fix (gate O1), applied at scoring time."""
    k = {0: 0, 90: 3, 180: 2, 270: 1}[degrees]
    return np.rot90(frames, k=k, axes=(1, 2)).copy() if k else frames


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def read_switches(videos: Path) -> list[dict]:
    p = videos / "switches.csv"
    if not p.exists():
        return []
    return [{"video_id": r["video_id"].strip(), "scene": r["scene"].strip(), "task_first": int(r["task_first"]),
             "task_final": int(r["task_final"]), "start_s": float(r["start_s"]), "switch_s": float(r["switch_s"]),
             "end_s": float(r["end_s"])} for r in csv.DictReader(p.open())]


def clip_windows(videos: Path, labels: list[dict]) -> dict:
    """video_id -> {window name: (a, b)} for labelled clips and change-of-mind clips."""
    out = {r["video_id"]: {f"p{p}": prefix_window(r["start_s"], r["end_s"], p) for p in PREFIXES} for r in labels}
    for r in read_switches(videos):
        out[r["video_id"]] = switch_windows(r["start_s"], r["switch_s"], r["end_s"])
    return out


def contact_sheet(items, out_path: Path, side: int = 192) -> None:
    """items: list of (label, RGB frame) shown in rows of 6 thumbnails."""
    from PIL import Image, ImageDraw

    cols = 6
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * side, rows * side), "white")
    for i, (label, fr) in enumerate(items):
        img = Image.fromarray(fr)
        img.thumbnail((side, side))
        ImageDraw.Draw(img).text((3, 3), label, fill=(255, 255, 0))
        sheet.paste(img, ((i % cols) * side, (i // cols) * side))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def cmd_extract(args) -> int:
    videos, work = Path(args.videos), Path(args.work)
    fdir = work / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels) if args.labels else videos / "labels.csv"
    if not labels_path.exists():  # trial clips: contact sheet only (first frame and the frame at 2 s)
        items, meta = [], {}
        for p in sorted(videos.glob("*.mp4")):
            m = probe(p)
            fr, _, _ = decode(p, wanted={0, min(int(round(2 * m["fps"])), m["n_frames"] - 1)})
            meta[p.stem] = {**m, "shape": list(next(iter(fr.values())).shape)}
            items += [(f"{p.stem} {i / m['fps']:.1f}s", f) for i, f in sorted(fr.items())]
        contact_sheet(items, work / args.sheet)
        (work / (Path(args.sheet).stem + "_meta.json")).write_text(json.dumps({"decoder": decoder_name(), "clips": meta}, indent=2))
        print(f"no labels: wrote {work / args.sheet} for {len(meta)} clips ({decoder_name()})")
        return 0
    labels = read_labels(labels_path)
    ids = [r["video_id"] for r in labels] + [r["video_id"] for r in read_switches(videos)]
    meta = {v: probe(videos / f"{v}.mp4") for v in ids if (videos / f"{v}.mp4").exists()}
    errs = gate_l1(labels, {v: m["duration"] for v, m in meta.items()}, partial=args.allow_partial)
    print(f"GATE L1{' (partial)' if args.allow_partial else ''}: {'PASS' if not errs else 'FAIL'}")
    for e in errs:
        print("  " + e)
    if errs:
        raise SystemExit("STOP: L1 failed")
    by_id = {r["video_id"]: r for r in labels}
    items = []
    for v, wins in clip_windows(videos, labels).items():
        m = meta[v]
        idx = {w: window_indices(a, b, m["fps"], m["n_frames"]) for w, (a, b) in wins.items()}
        start = by_id[v]["start_s"] if v in by_id else read_switches(videos)[[r["video_id"] for r in read_switches(videos)].index(v)]["start_s"]
        sheet_idx = {0, min(int(round(start * m["fps"])), m["n_frames"] - 1)}
        todo = {w: i for w, i in idx.items() if not (fdir / f"{v}_{w}.npy").exists()}
        want = {int(x) for i in todo.values() for x in i} | sheet_idx
        fr, _, _ = decode(videos / f"{v}.mp4", wanted=want)
        for w, i in todo.items():
            np.save(fdir / f"{v}_{w}.npy", np.stack([fr[int(x)] for x in i]).astype(np.uint8))
        m["shape"] = list(fr[0].shape)
        m["frame_indices"] = {w: i.tolist() for w, i in idx.items()}
        items += [(f"{v} 0s", fr[0]), (f"{v} start", fr[max(sheet_idx)])]
    contact_sheet(items, work / args.sheet)
    (work / "extract_meta.json").write_text(json.dumps({"decoder": decoder_name(), "clips": meta}, indent=2))
    print(f"extracted {len(meta)} clips with {decoder_name()}; contact sheet {work / args.sheet}")
    return 0


def context_groups(labels: list[dict]) -> dict:
    """video_id -> clips of the same (scene, person) among the labelled clips."""
    by = {}
    for r in labels:
        by.setdefault((r["scene"], r["person"]), []).append(r["video_id"])
    return {r["video_id"]: by[(r["scene"], r["person"])] for r in labels}


def cmd_score(args) -> int:
    import torch

    import extract_ladder_embeddings as X

    videos, work = Path(args.videos), Path(args.work)
    labels = read_labels(videos / "labels.csv")
    switches = read_switches(videos)
    wins = clip_windows(videos, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = X.load_encoder(ENCODER, device)
    emb = {}
    for v, ws in wins.items():
        for w in ws:
            fr = rotate(np.load(work / "frames" / f"{v}_{w}.npy"), args.rotate)
            emb[(v, w)] = encode(model, processor, fr, device, X)
    original = {int(k): x for k, x in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    desc = [original[t] for t in range(N_TASKS)]
    text = {}
    for name, tmpl in TEMPLATES.items():
        text[name], trunc = X.text_features(model, processor, ENCODER, [tmpl.format(why=x) for x in desc], device, 32)
        assert trunc == 0
    table = np.load(TABLE)
    rows = dict(zip(map(str, table["tasks"]), table["emb"]))
    e1 = max(float(np.abs(text["T1"][t] - rows[desc[t]]).max()) for t in range(N_TASKS))
    print(f"GATE E1 candidate text vs token table max abs diff = {e1:.3g}: {'PASS' if e1 < 1e-6 else 'FAIL'}")
    if e1 >= 1e-6:
        raise SystemExit("STOP: E1 failed")
    idle = {r["video_id"]: emb[(r["video_id"], "p0")] for r in labels}
    groups = context_groups(labels)
    mu_libero = np.load(LIBERO_CTX)["context_mean"]
    keys, out = [], {f"{c}_{t}": [] for c in CONTEXTS for t in TEMPLATES}
    for v, ws in wins.items():
        if v in groups:
            mu_scene = loo_context(idle, groups, v)
        else:  # change-of-mind clip: idle parts of the 30 primary clips of its scene (person p1)
            sc = next(r["scene"] for r in switches if r["video_id"] == v)
            mu_scene = np.concatenate([idle[r["video_id"]] for r in labels if r["scene"] == sc and r["person"] == PRIMARY_PERSON]).mean(axis=0)
        for w in ws:
            keys.append((v, w))
            for c, mu in (("C-scene", mu_scene), ("C0", None), ("C-libero", mu_libero)):
                q = query_vec(emb[(v, w)], mu)
                for t, tt in text.items():
                    out[f"{c}_{t}"].append(q @ tt.astype(np.float64).T)
    np.savez(work / "scores.npz", keys=np.asarray(keys), **{k: np.asarray(x) for k, x in out.items()})
    info = {r["video_id"]: r for r in labels}
    pred = []
    for i, (v, w) in enumerate(keys):
        r = info.get(v)
        for name, s in out.items():
            s = s[i]
            y, m = top2(s)
            row = {"video_id": v, "window": w, "context": name.rsplit("_", 1)[0], "template": name.rsplit("_", 1)[1], "pred10": y, "margin10": m}
            if r is not None:
                cands = SCENE_TASKS[r["scene"]]
                yw, mw = masked(s, cands)
                row.update({"task": r["task"], "scene": r["scene"], "rep": r["rep"], "person": r["person"],
                            "correct10": strict_correct(s, r["task"]), "pred_ws": yw, "margin_ws": mw,
                            "correct_ws": masked_correct(s, r["task"], cands) if len(cands) > 1 else None})
            else:
                sw = next(x for x in switches if x["video_id"] == v)
                cands = SCENE_TASKS[sw["scene"]]
                yw, mw = masked(s, cands)
                row.update({"scene": sw["scene"], "task_first": sw["task_first"], "task_final": sw["task_final"], "pred_ws": yw, "margin_ws": mw})
            pred.append(row)
    (work / "predictions.json").write_text(json.dumps(pred))
    (work / "score_meta.json").write_text(json.dumps({"rotate_degrees": args.rotate, "E1_max_abs_diff": e1, "n_entries": len(keys)}, indent=2))
    return 0


def units() -> list[tuple[int, int]]:
    return [(t, e) for t in range(N_TASKS) for e in range(30)]


def cmd_plan(args) -> int:
    work = Path(args.work)
    original = {int(k): x for k, x in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    pred = json.loads((work / "predictions.json").read_text())
    prim = {r["video_id"]: r for r in pred if r.get("person") == PRIMARY_PERSON and r["window"] == f"p{PRIMARY_P}"
            and r["context"] == "C-scene" and r["template"] == "T1"}
    assert len(prim) == 30, len(prim)
    tau = ask_threshold([r["margin10"] for r in prim.values()])
    plan = {"tau_h": tau, "n_clips": len(prim), "tokens": {}, "ask": {}, "correct": {}, "clip": {}}
    for t, e in units():
        r = prim[f"t{t}_r{rep_of(e)}"]
        k = f"{t},{e}"
        plan["tokens"][k], plan["ask"][k], plan["correct"][k], plan["clip"][k] = original[r["pred10"]], r["margin10"] < tau, r["correct10"], r["video_id"]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e2e_human_tokens.json").write_text(json.dumps(plan, indent=1))
    print(f"tau_h {tau:.6f} on 30 clips; ask rate {np.mean(list(plan['ask'].values())):.3f}")
    ok = all(json.loads((EVAL / f"episodes/I-only_t{t}_e{e}.json").read_text())["z_int_task"] == original[t]
             for t, e in ((0, 0), (4, 5), (9, 9)))
    print(f"GATE R1 3 I-only episodes hold their own description: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("STOP: R1 failed")
    return 0


def write_csv(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def recognition_rows(pred: list[dict], person: str) -> tuple[list, dict]:
    """10-way and within-scene accuracy per (p, C, T) for one person; CIs clustered by task."""
    rows, hits = [], {}
    for c in CONTEXTS:
        for tm in TEMPLATES:
            for p in PREFIXES:
                sel = sorted((r for r in pred if r.get("person") == person and r["window"] == f"p{p}" and r["context"] == c
                              and r["template"] == tm), key=lambda r: r["video_id"])
                if not sel:
                    continue
                h10 = np.array([float(r["correct10"]) for r in sel])
                ws = [r for r in sel if r["correct_ws"] is not None]
                hws = np.array([float(r["correct_ws"]) for r in ws])
                hits[(c, tm, p)] = (sel, h10, ws, hws)
                a, lo, hi = cluster_bootstrap(np.array([r["task"] for r in sel]), h10)
                b, blo, bhi = cluster_bootstrap(np.array([r["task"] for r in ws]), hws)
                chance = float(np.mean([1 / len(SCENE_TASKS[r["scene"]]) for r in ws]))
                rows.append({"person": person, "context": c, "template": tm, "p": p, "n_clips": len(sel), "acc10": a,
                             "acc10_lo": lo, "acc10_hi": hi, "n_within_scene": len(ws), "acc_ws": b, "acc_ws_lo": blo,
                             "acc_ws_hi": bhi, "chance_ws": chance})
    return rows, hits


def cmd_analyze(args) -> int:
    videos, work = Path(args.videos), Path(args.work)
    original = {int(k): x for k, x in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    pred = json.loads((work / "predictions.json").read_text())
    plan = json.loads((RES / "e2e_human_tokens.json").read_text())
    smeta = json.loads((work / "score_meta.json").read_text())
    emeta = json.loads((work / "extract_meta.json").read_text())
    rec, hits = recognition_rows(pred, PRIMARY_PERSON)
    persons = sorted({r.get("person") for r in pred if r.get("person") and r.get("person") != PRIMARY_PERSON})
    for pp in persons:
        rec += recognition_rows(pred, pp)[0]
    sel50, h10_50, ws50, hws50 = hits[("C-scene", "T1", PRIMARY_P)]
    _, _, ws0, hws0 = hits[("C-scene", "T1", 0)]
    assert [r["video_id"] for r in ws50] == [r["video_id"] for r in ws0]
    h1 = cluster_bootstrap(np.array([r["task"] for r in sel50]), h10_50)
    h2 = cluster_bootstrap(np.array([r["task"] for r in ws50]), hws50 - hws0)
    ws_vs_chance = cluster_bootstrap(np.array([r["task"] for r in ws50]), hws50 - np.array([1 / len(SCENE_TASKS[r["scene"]]) for r in ws50]))
    conf = np.zeros((N_TASKS, N_TASKS), int)
    for r in sel50:
        conf[r["task"], r["pred10"]] += 1
    us = units()
    # Amendment 1: E2E-human is read from the seed-0 token matrix (no E2E-human rollouts).
    # The E2E-human rollout path of libero_goal_ablation.py is kept for reference only.
    mat = TM.load_matrix(0)
    looked = TM.lookup(mat, plan["tokens"], original)
    succ = {"E2E-human": np.array([looked[f"{t},{e}"] for t, e in us]),
            "I-only": np.array([mat[t, t, e] for t, e in us]),
            "I-wrong": np.array([mat[t, G.wrong_task(t), e] for t, e in us])}
    seed1 = None
    if TM.matrix_path(1).exists():  # descriptive seed-1 E2E-human by the same look-up
        m1 = TM.load_matrix(1)
        seed1 = {"E2E-human": int(sum(TM.lookup(m1, plan["tokens"], original).values())), "I-only": int(sum(m1[t, t, e] for t, e in us)), "n": len(us)}
    ask = np.array([plan["ask"][f"{t},{e}"] for t, e in us])
    right = np.array([plan["correct"][f"{t},{e}"] for t, e in us])
    clip = np.array([plan["clip"][f"{t},{e}"] for t, e in us])
    succ["E2E-human-ask"] = e2e_ask(succ["E2E-human"], succ["I-only"], ask)
    s12 = {r["condition"]: r for r in csv.DictReader(STAGE12_SUMMARY.open())}["E2E"]
    summary = []
    for c in ("E2E-human", "E2E-human-ask", "I-only", "I-wrong"):
        k, n = int(succ[c].sum()), len(us)
        lo, hi = G.wilson(k, n)
        summary.append({"condition": c, "successes": k, "n": n, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
    lo, hi = float(s12["wilson_lo"]), float(s12["wilson_hi"])
    summary.append({"condition": "Stage 12 E2E (LIBERO demos, other query set)", "successes": int(s12["successes"]), "n": int(s12["n"]),
                    "success_rate": float(s12["success_rate"]), "wilson_lo": lo, "wilson_hi": hi})
    paired = [{"rule": "H2", "comparison": "within-scene acc p=50 - p=0 (27 clips)", "delta": h2[0], "ci95_lo": h2[1], "ci95_hi": h2[2], "method": "bootstrap clustered by task"},
              {"rule": "descriptive", "comparison": "within-scene acc p=50 - chance", "delta": ws_vs_chance[0], "ci95_lo": ws_vs_chance[1], "ci95_hi": ws_vs_chance[2], "method": "bootstrap clustered by task"}]
    for rule, a, b in (("H3", "E2E-human", "I-wrong"), ("descriptive", "E2E-human", "I-only"), ("descriptive", "E2E-human-ask", "E2E-human")):
        x = G.paired_bootstrap(succ[a] - succ[b])
        paired.append({"rule": rule, "comparison": f"{a} - {b}", "delta": x[0], "ci95_lo": x[1], "ci95_hi": x[2], "method": "paired_bootstrap"})
    x = cluster_bootstrap(clip, succ["E2E-human"] - succ["I-wrong"])
    paired.append({"rule": "descriptive", "comparison": "E2E-human - I-wrong (clustered by clip)", "delta": x[0], "ci95_lo": x[1], "ci95_hi": x[2], "method": "bootstrap clustered by clip (30)"})
    per_clip = []
    look = {(r["video_id"], r["window"]): r for r in pred if r["context"] == "C-scene" and r["template"] == "T1"}
    for r in sorted((r for r in sel50), key=lambda r: (r["task"], r["rep"])):
        v = r["video_id"]
        a0 = look[(v, "p0")]
        per_clip.append({"video_id": v, "task": r["task"], "scene": r["scene"], "rep": r["rep"], "pred10_p0": a0["pred10"], "pred10_p50": r["pred10"],
                         "pred_ws_p0": a0["pred_ws"], "pred_ws_p50": r["pred_ws"], "margin10_p50": r["margin10"],
                         "asked": r["margin10"] < plan["tau_h"]})
    per_task = []
    for t in range(N_TASKS):
        m = np.array([u[0] == t for u in us])
        per_task.append({"task": t, "scene": SCENE_OF[t], "recognition_p50_of_3": int(sum(r["correct10"] for r in sel50 if r["task"] == t)),
                         "E2E-human": int(succ["E2E-human"][m].sum()), "I-only": int(succ["I-only"][m].sum()), "I-wrong": int(succ["I-wrong"][m].sum()), "n": int(m.sum())})
    switch_rows = []
    for v in sorted({r["video_id"] for r in pred if "task_first" in r}):
        b = next(r for r in pred if r["video_id"] == v and r["window"] == "before" and r["context"] == "C-scene" and r["template"] == "T1")
        a = next(r for r in pred if r["video_id"] == v and r["window"] == "after" and r["context"] == "C-scene" and r["template"] == "T1")
        switch_rows.append({"video_id": v, "scene": b["scene"], "task_first": b["task_first"], "task_final": b["task_final"],
                            "pred_ws_before": b["pred_ws"], "before_is_first": b["pred_ws"] == b["task_first"], "margin_ws_before": b["margin_ws"],
                            "pred_ws_after": a["pred_ws"], "after_is_final": a["pred_ws"] == a["task_final"], "margin_ws_after": a["margin_ws"]})
    agree = float((succ["E2E-human"][right] == succ["I-only"][right]).mean()) if right.any() else float("nan")
    h3 = next(r for r in paired if r["rule"] == "H3")
    verdicts = {
        "H1": {"pass": h1[1] > 0.10, "accuracy": h1[0], "ci95": [h1[1], h1[2]], "rule": "10-way acc p=50 CI lower bound > 10%"},
        "H2": {"pass": h2[1] > 0, "delta": h2[0], "ci95": [h2[1], h2[2]], "rule": "within-scene acc p=50 - p=0 CI lower bound > 0"},
        "H3": {"pass": h3["ci95_lo"] > 0, "delta": h3["delta"], "ci95": [h3["ci95_lo"], h3["ci95_hi"]], "rule": "success(E2E-human) - success(I-wrong) CI lower bound > 0"},
    }
    RES.mkdir(parents=True, exist_ok=True)
    write_csv(RES / "recognition.csv", rec)
    write_csv(RES / "summary.csv", summary)
    write_csv(RES / "paired.csv", paired)
    write_csv(RES / "per_clip.csv", per_clip)
    write_csv(RES / "per_task.csv", per_task)
    if switch_rows:
        write_csv(RES / "switches.csv", switch_rows)
    with (RES / "confusion_p50.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["true\\pred"] + list(range(N_TASKS)))
        for t in range(N_TASKS):
            w.writerow([t] + conf[t].tolist())
    lines = ["# Stage 13: a real person in the loop", "", f"300 units. tau_h = {plan['tau_h']:.6f} on 30 clips; ask rate {ask.mean():.3f}.", "",
             "| condition | successes / n | rate | Wilson 95% |", "|---|---:|---:|---|"]
    lines += [f"| {r['condition']} | {r['successes']} / {r['n']} | {r['success_rate']:.3f} | [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |" for r in summary]
    lines += ["", "| rule | comparison | delta [95% CI] |", "|---|---|---|"]
    lines += [f"| {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in paired]
    lines += ["", "```", json.dumps(verdicts, indent=2), "```", ""]
    (RES / "summary.md").write_text("\n".join(lines))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (RES / "config.json").write_text(json.dumps({
        "git_commit": commit, "vlia_checkpoint": VLIA, "token_table": TABLE, "encoder": ENCODER, "tau_h": plan["tau_h"],
        "ask_rate": float(ask.mean()), "decoder": emeta["decoder"], "rotate_degrees": smeta["rotate_degrees"],
        "clips": {v: {k: m[k] for k in ("fps", "n_frames", "duration", "shape") if k in m} for v, m in emeta["clips"].items()},
        "E1_gate_max_abs_diff": smeta["E1_max_abs_diff"],
        "robot_source": "token matrix look-up (amendment 1), seed 0", "seed1_lookup": seed1,
        "D1": {"same_success_rate_when_F_right": agree, "n_units_F_right": int(right.sum()),
               "note": "identical by construction under the look-up; run-to-run noise is gate L1 of results/token_matrix"},
        "E2E_human_by_F_right": {"F_right": {"n": int(right.sum()), "successes": int(succ["E2E-human"][right].sum())},
                                 "F_wrong": {"n": int((~right).sum()), "successes": int(succ["E2E-human"][~right].sum())}},
        "p2_persons": persons, "verdicts": verdicts}, indent=2))
    labels_src = videos / "labels.csv"
    if labels_src.exists() and not (RES / "labels.csv").exists():
        (RES / "labels.csv").write_text(labels_src.read_text())
    print((RES / "summary.md").read_text())
    return 0


def cmd_selftest(_) -> int:
    # window sampler: 8 frames, window end included, p=0 ends at start_s
    a, b = prefix_window(2.0, 6.0, 0)
    assert (a, b) == (0.0, 2.0) and prefix_window(2.0, 6.0, 50) == (0.0, 4.0) and prefix_window(2.0, 6.0, 100) == (0.0, 6.0)
    i = window_indices(0.0, 2.0, 30.0, 300)
    assert len(i) == N_FRAMES and i[0] == 0 and i[-1] == 60 and (np.diff(i) >= 0).all()
    assert window_indices(0.0, 10.0, 30.0, 290)[-1] == 289, "clipped to the last decoded frame"
    sw = switch_windows(2.0, 6.0, 9.0)
    assert sw["before"] == (0.0, 4.0) and sw["after"] == (5.0, 9.0) and switch_windows(1.0, 2.0, 3.0)["after"] == (0.0, 3.0)
    # leave-one-out scene mean never contains the clip itself
    idle = {"a": np.array([[1.0, 0.0]]), "b": np.array([[0.0, 1.0]]), "c": np.array([[0.0, 3.0]])}
    groups = {"a": ["a", "b", "c"], "b": ["a", "b", "c"], "c": ["a", "b", "c"]}
    assert np.allclose(loo_context(idle, groups, "a"), [0.0, 2.0]) and np.allclose(loo_context(idle, groups, "c"), [0.5, 0.5])
    # within-scene masking; a tie counts as wrong
    s = np.zeros(N_TASKS)
    s[[0, 1, 7, 9]] = [0.2, 0.5, 0.4, 0.9]
    assert masked(s, SCENE_TASKS["A"])[0] == 1 and np.isclose(masked(s, SCENE_TASKS["A"])[1], 0.1)
    assert masked_correct(s, 1, SCENE_TASKS["A"]) and not masked_correct(s, 1, range(N_TASKS))
    s[7] = 0.5
    assert not masked_correct(s, 1, SCENE_TASKS["A"])
    assert np.isclose(np.mean([1 / len(SCENE_TASKS[SCENE_OF[t]]) for t in range(N_TASKS) if t != 5 for _ in REPS]), 12 / 27), "per-clip chance (DEVIATIONS 2)"
    # gate L1
    good = [{"video_id": f"t{t}_r{r}", "task": t, "scene": SCENE_OF[t], "rep": r, "person": "p1", "start_s": 2.0, "end_s": 6.0}
            for t in range(N_TASKS) for r in REPS]
    dur = {x["video_id"]: 7.0 for x in good}
    assert gate_l1(good, dur) == []
    for change in ({"scene": "B"}, {"start_s": 0.3}, {"end_s": 7.5}, {"start_s": 6.5}):
        bad = [dict(x) for x in good]
        bad[4].update(change)
        assert gate_l1(bad, dur), change
    assert gate_l1(good[:-1], dur) and not gate_l1(good[:-1], dur, partial=True)
    assert gate_l1(good, {k: v for k, v in dur.items() if k != "t9_r3"}), "a missing clip fails L1"
    # rep mapping r = (e mod 3) + 1
    assert [rep_of(e) for e in range(6)] == [1, 2, 3, 1, 2, 3] and rep_of(29) == 3
    # rotation fix
    f = np.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3)
    assert rotate(f, 90).shape == (2, 4, 3, 3) and np.array_equal(rotate(rotate(f, 90), 270), f) and rotate(f, 0) is f
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "extract", "score", "plan", "analyze"])
    ap.add_argument("--videos", default=str(VIDEOS))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--labels", help="labels.csv (default: <videos>/labels.csv)")
    ap.add_argument("--sheet", default="contact_sheet.png")
    ap.add_argument("--allow-partial", action="store_true", help="L1 without the 'all 30 clips' check (fake / trial data only)")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270))
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "extract": cmd_extract, "score": cmd_score, "plan": cmd_plan, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
