#!/usr/bin/env python3
"""Stage 12: video -> F picks the task -> token of the picked task -> VLIA acts (LIBERO-10).

Subcommands: selftest | extract | score | plan | analyze. Pre-registration: results/prereg_stage12_end_to_end.md.
Rollouts run through scripts/libero_goal_ablation.py (condition E2E, --e2e-tokens results/.../e2e_tokens.json).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402

N_TASKS = 10
N_FRAMES = 8
N_QUERY = 30  # demos 0..29
N_CAL_END = 50  # calibration demos 30..49
MIN_DEMOS = 29  # prereg amendment A1
PREFIXES = (0, 10, 25, 50, 100)
PRIMARY_P = 25
CONTEXT_MAX = 2000
ASK_Q = 0.25
CAM = "observation.images.image"
ENCODER = "google/siglip-so400m-patch14-384"
TEMPLATES = {"T1": "a first-person photo of a person doing: {why}", "T0": "{why}"}
SNAP_GLOB = str(Path.home() / ".cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/*/")
WORK = Path("/media/dhqg/d3/vlia_outputs/stage12")
RES = ROOT / "results/stage12_end_to_end"
TABLE = "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings_v2.npz"
EVAL = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval")
VLIA = "/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000/pretrained_model"


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def norm_task(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip().rstrip(".").rstrip()


def map_tasks(dataset_tasks: dict, original: dict) -> dict:
    """dataset task_index -> eval task id, exact match of normalised text; stops unless one-to-one."""
    inv = {norm_task(v): int(k) for k, v in original.items()}
    out = {ti: inv.get(norm_task(txt)) for ti, txt in dataset_tasks.items()}
    if None in out.values() or sorted(out.values()) != sorted(inv.values()) or len(out) != len(inv):
        raise SystemExit(f"STOP: task mapping is not one-to-one: {out}")
    return out


def prefix_indices(length: int, p: int) -> np.ndarray:
    """p = 0: the first frame only. Otherwise 8 frames evenly spaced over the first p%, last one included."""
    if p == 0:
        return np.array([0])
    n = max(1, math.ceil(length * p / 100))
    return np.round(np.linspace(0, n - 1, N_FRAMES)).astype(int)


def query_vec(emb: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
    """emb (F, D) unit frames; mu = fixed mean (C2 fallback) or None (C0). F's A0 query."""
    f = emb if mu is None else normalize(emb - mu)
    return normalize(f.mean(axis=0))


def top2(scores: np.ndarray):
    """(prediction, margin, correct given true index): a tie for the top counts as wrong."""
    order = np.argsort(-scores, kind="stable")
    return int(order[0]), float(scores[order[0]] - scores[order[1]])


def strict_correct(scores: np.ndarray, true: int) -> bool:
    return bool((scores[true] > np.delete(scores, true)).all())


def ask_threshold(cal_margins) -> float:
    return float(np.quantile(np.asarray(cal_margins, float), ASK_Q))


def e2e_ask(e2e: np.ndarray, ionly: np.ndarray, ask: np.ndarray) -> np.ndarray:
    return np.where(ask, ionly, e2e)


def cluster_bootstrap(clusters, values, B: int = 10000, seed: int = 0):
    """Mean of values, 95% CI resampling whole clusters with replacement (ratio of sums)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=values, minlength=len(uniq))
    sd = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(B, len(uniq)))
    boots = sn[idx].sum(axis=1) / sd[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(sn.sum() / sd.sum()), float(lo), float(hi)


def write_csv(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def load_meta():
    """Demo table from the dataset metadata: demos[t] = sorted episode indices, plus episode rows."""
    import pandas as pd

    snap = sorted(glob.glob(SNAP_GLOB))[-1]
    tasks = pd.read_parquet(Path(snap) / "meta/tasks.parquet")
    ds_tasks = {int(i): str(txt) for txt, i in tasks["task_index"].items() if int(i) < N_TASKS}
    original = json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"]
    mapping = map_tasks(ds_tasks, original)
    data = pd.concat(pd.read_parquet(f, columns=["episode_index", "task_index"])
                     for f in sorted(glob.glob(str(Path(snap) / "data/**/*.parquet"), recursive=True)))
    ep_task = data.groupby("episode_index").task_index.first()
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(Path(snap) / "meta/episodes/**/*.parquet"), recursive=True)))
    eps = eps.set_index("episode_index")
    demos = {t: [] for t in range(N_TASKS)}
    for ep, ti in ep_task.items():
        if ti < N_TASKS:
            demos[mapping[int(ti)]].append(int(ep))
    demos = {t: sorted(v) for t, v in demos.items()}
    others = sorted(int(ep) for ep, ti in ep_task.items() if ti >= N_TASKS)
    n = {t: len(v) for t, v in demos.items()}
    m1 = min(n.values()) >= MIN_DEMOS
    print(f"GATE M1 one-to-one mapping, demos per task {n} (min required {MIN_DEMOS}): {'PASS' if m1 else 'FAIL'}")
    if not m1:
        raise SystemExit("STOP: M1 failed")
    return snap, eps, demos, others, {int(k): v for k, v in original.items()}


def context_episodes(others):
    rng = np.random.default_rng(0)
    return sorted(int(x) for x in rng.permutation(others)[:CONTEXT_MAX])


def video_loc(eps, ep):
    r = eps.loc[ep]
    return (int(r[f"videos/{CAM}/chunk_index"]), int(r[f"videos/{CAM}/file_index"]),
            float(r[f"videos/{CAM}/from_timestamp"]), int(r["length"]))


def cmd_extract(_) -> int:
    import av
    from PIL import Image, ImageDraw

    snap, eps, demos, others, _orig = load_meta()
    fdir = WORK / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    fps = 10.0
    jobs = []  # (out path, episode, frame indices)
    for t, v in demos.items():
        for d, ep in enumerate(v[:N_CAL_END]):
            length = video_loc(eps, ep)[3]
            for p in PREFIXES:
                jobs.append((fdir / f"t{t}_d{d}_p{p}.npy", ep, prefix_indices(length, p)))
    for ep in context_episodes(others):
        length = video_loc(eps, ep)[3]
        jobs.append((fdir / f"ctx_{ep}.npy", ep, np.round(np.linspace(0, length - 1, N_FRAMES)).astype(int)))
    todo = [j for j in jobs if not j[0].exists()]
    print(f"{len(jobs)} frame sets, {len(todo)} to extract", flush=True)
    by_file = {}
    for path, ep, idx in todo:
        chunk, fi, t0, _ = video_loc(eps, ep)
        start = int(round(t0 * fps))
        by_file.setdefault((chunk, fi), []).append((path, start + idx))
    for n, ((chunk, fi), items) in enumerate(sorted(by_file.items())):
        want = {int(i) for _, ids in items for i in ids}
        last = max(want)
        got = {}
        with av.open(str(Path(snap) / f"videos/{CAM}/chunk-{chunk:03d}/file-{fi:03d}.mp4")) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for frame in c.decode(s):
                i = int(round(float(frame.pts * s.time_base) * fps))
                if i in want:
                    got[i] = frame.to_ndarray(format="rgb24")
                if i >= last:
                    break
        missing = want - set(got)
        if missing:
            raise SystemExit(f"STOP: {len(missing)} frames not decoded in file {fi}")
        for path, ids in items:
            np.save(path, np.stack([got[int(i)] for i in ids]).astype(np.uint8))
        print(f"file {n + 1}/{len(by_file)}: {len(want)} frames", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    side = 256
    sheet = Image.new("RGB", (5 * side, 2 * side), "white")
    for t in range(N_TASKS):
        img = Image.fromarray(np.load(fdir / f"t{t}_d0_p0.npy")[0])
        ImageDraw.Draw(img).text((4, 4), f"task {t}", fill=(255, 255, 0))
        sheet.paste(img, ((t % 5) * side, (t // 5) * side))
    sheet.save(RES / "contact_sheet.png")
    print("contact sheet written", flush=True)
    return 0


def encode(model, processor, frames, device, X, bs=64):
    out = [X.encode_pixels(model, X.pixel_values(processor, frames[i:i + bs]), device) for i in range(0, len(frames), bs)]
    return np.concatenate(out)


def cmd_score(args) -> int:
    import torch

    import extract_ladder_embeddings as X

    rotate = bool(args.rotate180)
    snap, eps, demos, others, original = load_meta()
    fdir = WORK / "frames"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = X.load_encoder(ENCODER, device)

    def emb_of(path):
        fr = np.load(path)
        if rotate:
            fr = fr[:, ::-1, ::-1].copy()
        return encode(model, processor, fr, device, X)

    ctx_eps = context_episodes(others)
    ctx = [emb_of(fdir / f"ctx_{ep}.npy") for ep in ctx_eps]
    mu = np.concatenate(ctx).mean(axis=0)
    print(f"context mean from {len(ctx_eps)} episodes", flush=True)
    keys, embs = [], {}
    for t, v in demos.items():
        for d in range(min(len(v), N_CAL_END)):
            for p in PREFIXES:
                keys.append((t, d, p))
                embs[(t, d, p)] = emb_of(fdir / f"t{t}_d{d}_p{p}.npy")
    print(f"{len(keys)} queries embedded", flush=True)
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
    scores = {}
    for c, m in (("C2", mu), ("C0", None)):
        q = np.stack([query_vec(embs[k], m) for k in keys])
        for name, tt in text.items():
            scores[f"{c}_{name}"] = q @ tt.astype(np.float64).T
    np.savez(WORK / "scores.npz", keys=np.asarray(keys), context_mean=mu, **scores)
    pred = []
    s = scores["C2_T1"]
    for i, (t, d, p) in enumerate(keys):
        y, m = top2(s[i])
        pred.append({"task": t, "demo": d, "p": p, "pred": y, "margin": m, "correct": strict_correct(s[i], t)})
    (WORK / "predictions.json").write_text(json.dumps(pred))
    (WORK / "score_meta.json").write_text(json.dumps({
        "rotate180": rotate, "E1_max_abs_diff": e1, "n_context_episodes": len(ctx_eps),
        "demos_per_task": {t: len(v) for t, v in demos.items()}}, indent=2))
    return 0


def cmd_plan(_) -> int:
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    pred = json.loads((WORK / "predictions.json").read_text())
    tau = ask_threshold([r["margin"] for r in pred if r["p"] == PRIMARY_P and r["demo"] >= N_QUERY])
    n_cal = sum(r["p"] == PRIMARY_P and r["demo"] >= N_QUERY for r in pred)
    q = [r for r in pred if r["p"] == PRIMARY_P and r["demo"] < N_QUERY]
    plan = {"tau": tau, "n_calibration": n_cal, "tokens": {f"{r['task']},{r['demo']}": original[r["pred"]] for r in q},
            "ask": {f"{r['task']},{r['demo']}": r["margin"] < tau for r in q},
            "correct": {f"{r['task']},{r['demo']}": r["correct"] for r in q}}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e2e_tokens.json").write_text(json.dumps(plan, indent=1))
    print(f"tau {tau:.6f} on {n_cal} calibration demos; {len(q)} query units; ask rate {np.mean(list(plan['ask'].values())):.3f}")
    ok = True
    for t, e in ((0, 0), (4, 5), (9, 9)):
        z = json.loads((EVAL / f"episodes/I-only_t{t}_e{e}.json").read_text())["z_int_task"]
        ok &= z == original[t]
    print(f"GATE R1 3 I-only episodes hold their own description: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("STOP: R1 failed")
    return 0


def cmd_analyze(_) -> int:
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    d = np.load(WORK / "scores.npz")
    keys = d["keys"]
    meta = json.loads((WORK / "score_meta.json").read_text())
    plan = json.loads((RES / "e2e_tokens.json").read_text())
    rec, p2_rows = [], []
    q = keys[:, 1] < N_QUERY
    hits = {}
    for name in ("C2_T1", "C2_T0", "C0_T1", "C0_T0"):
        s = d[name]
        for p in PREFIXES:
            sel = np.where(q & (keys[:, 2] == p))[0]
            h = np.array([strict_correct(s[i], keys[i, 0]) for i in sel], float)
            hits[(name, p)] = h
            pt, lo, hi = cluster_bootstrap(keys[sel, 0], h)
            rec.append({"context": name[:2], "template": name[3:], "p": p, "n_demos": len(sel), "accuracy": pt, "ci95_lo": lo, "ci95_hi": hi})
    sel = np.where(q & (keys[:, 2] == PRIMARY_P))[0]
    sel0 = np.where(q & (keys[:, 2] == 0))[0]
    assert (keys[sel][:, :2] == keys[sel0][:, :2]).all()
    tasks, demos_ = keys[sel, 0], keys[sel, 1]
    pp = cluster_bootstrap(tasks, hits[("C2_T1", PRIMARY_P)] - hits[("C2_T1", 0)])
    p1 = cluster_bootstrap(tasks, hits[("C2_T1", PRIMARY_P)])
    conf = np.zeros((N_TASKS, N_TASKS), int)
    for i in sel:
        conf[keys[i, 0], int(np.argmax(d["C2_T1"][i]))] += 1
    units = [(int(t), int(e)) for t, e in zip(tasks, demos_)]
    conds = ("E2E", "I-only", "I-wrong")
    by = {}
    for c in conds:
        for t, e in units:
            r = json.loads((EVAL / f"episodes/{c}_t{t}_e{e}.json").read_text())
            assert r["seed"] == 1000 + e and r["init_state_id"] == e
            by[(c, t, e)] = r
    for t, e in units:
        assert by[("E2E", t, e)]["z_int_task"] == plan["tokens"][f"{t},{e}"], (t, e)
    succ = {c: np.array([float(by[(c, t, e)]["success"]) for t, e in units]) for c in conds}
    ask = np.array([plan["ask"][f"{t},{e}"] for t, e in units])
    right = np.array([plan["correct"][f"{t},{e}"] for t, e in units])
    succ["E2E-ask"] = e2e_ask(succ["E2E"], succ["I-only"], ask)
    summary = []
    for c in ("E2E", "E2E-ask", "I-only", "I-wrong"):
        k, n = int(succ[c].sum()), len(units)
        lo, hi = G.wilson(k, n)
        summary.append({"condition": c, "successes": k, "n": n, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
    paired = [{"rule": "P2", "comparison": "accuracy p=25 - accuracy p=0", "delta": pp[0], "ci95_lo": pp[1], "ci95_hi": pp[2], "n": len(sel), "method": "bootstrap clustered by task"}]
    for rule, a, b in (("E1", "E2E", "I-wrong"), ("descriptive", "E2E", "I-only"), ("descriptive", "E2E-ask", "E2E")):
        x = G.paired_bootstrap(succ[a] - succ[b])
        paired.append({"rule": rule, "comparison": f"{a} - {b}", "delta": x[0], "ci95_lo": x[1], "ci95_hi": x[2], "n": len(units), "method": "paired_bootstrap"})
    tr = []
    for t in range(N_TASKS):
        m = tasks == t
        tr.append({"task": t, "description": original[t], "n": int(m.sum()), "recognition_p25": float(hits[("C2_T1", PRIMARY_P)][m].mean()),
                   "E2E": int(succ["E2E"][m].sum()), "I-only": int(succ["I-only"][m].sum()), "I-wrong": int(succ["I-wrong"][m].sum()),
                   "most_predicted_wrong": int(np.argmax(np.where(np.arange(N_TASKS) == t, -1, conf[t])))})
    agree = float((succ["E2E"][right] == succ["I-only"][right]).mean())
    split = {"F_right": {"n": int(right.sum()), "E2E_successes": int(succ["E2E"][right].sum())},
             "F_wrong": {"n": int((~right).sum()), "E2E_successes": int(succ["E2E"][~right].sum())}}
    token_ok = all(by[("E2E", t, e)]["z_int_task"] == by[("I-only", t, e)]["z_int_task"] for (t, e), r in zip(units, right) if r)
    e1 = next(r for r in paired if r["rule"] == "E1")
    verdicts = {
        "P1": {"pass": p1[1] > 0.10, "accuracy": p1[0], "ci95": [p1[1], p1[2]], "rule": "accuracy p=25 CI lower bound > 10%"},
        "P2": {"pass": pp[1] > 0, "delta": pp[0], "ci95": [pp[1], pp[2]], "rule": "accuracy p=25 - p=0 CI lower bound > 0"},
        "E1": {"pass": e1["ci95_lo"] > 0, "delta": e1["delta"], "ci95": [e1["ci95_lo"], e1["ci95_hi"]], "rule": "success(E2E) - success(I-wrong) CI lower bound > 0"},
    }
    write_csv(RES / "recognition.csv", rec)
    write_csv(RES / "summary.csv", summary)
    write_csv(RES / "paired.csv", paired)
    write_csv(RES / "per_task.csv", tr)
    with (RES / "confusion_p25.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["true\\pred"] + list(range(N_TASKS)))
        for t in range(N_TASKS):
            w.writerow([t] + conf[t].tolist())
    lines = ["# Stage 12: from video to robot action", "", f"{len(units)} paired units. tau = {plan['tau']:.6f} on {plan['n_calibration']} calibration demos; ask rate {ask.mean():.3f}.", "",
             "| condition | successes / n | rate | Wilson 95% |", "|---|---:|---:|---|"]
    lines += [f"| {r['condition']} | {r['successes']} / {r['n']} | {r['success_rate']:.3f} | [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |" for r in summary]
    lines += ["", "| rule | comparison | delta [95% CI] |", "|---|---|---|"]
    lines += [f"| {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in paired]
    lines += ["", "```", json.dumps(verdicts, indent=2), "```", ""]
    (RES / "summary.md").write_text("\n".join(lines))
    secs = float(np.mean([by[("E2E", t, e)]["seconds"] for t, e in units]))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (RES / "config.json").write_text(json.dumps({
        "git_commit": commit, "vlia_checkpoint": VLIA, "token_table": TABLE, "encoder": ENCODER, "tau": plan["tau"],
        "n_calibration": plan["n_calibration"], "ask_rate": float(ask.mean()), "rotate180": meta["rotate180"],
        "demos_per_task": meta["demos_per_task"], "n_context_episodes": meta["n_context_episodes"],
        "E1_gate_max_abs_diff": meta["E1_max_abs_diff"], "seconds_per_episode": secs,
        "D1": {"tokens_equal_when_F_right": token_ok, "same_success_rate_when_F_right": agree, "n": int(right.sum())},
        "E2E_by_F_right": split, "verdicts": verdicts}, indent=2))
    for r in rec:
        print(f"{r['context']} {r['template']} p={r['p']}: {100 * r['accuracy']:.1f} [{100 * r['ci95_lo']:.1f}; {100 * r['ci95_hi']:.1f}]")
    print(np.array2string(conf))
    print((RES / "summary.md").read_text())
    print(json.dumps({"D1": {"tokens": token_ok, "same_success": agree, "n": int(right.sum())}, "split": split}))
    for r in tr:
        print(r)
    return 0


def cmd_selftest(_) -> int:
    assert prefix_indices(200, 0).tolist() == [0]
    i = prefix_indices(200, 25)
    assert len(i) == N_FRAMES and i[0] == 0 and i[-1] == 49 and (np.diff(i) >= 0).all()
    assert prefix_indices(200, 100)[-1] == 199 and prefix_indices(151, 10)[-1] == 15
    orig = {"0": "Put  the cup.", "1": "open box"}
    assert map_tasks({5: "put the cup", 7: "OPEN box"}, orig) == {5: 0, 7: 1}
    for bad in ({5: "put the cup", 7: "other"}, {5: "put the cup", 7: "put the cup."}):
        try:
            map_tasks(bad, orig)
            raise AssertionError("mapping must stop")
        except SystemExit:
            pass
    assert ask_threshold([0, 1, 2, 3, 4]) == 1.0
    assert e2e_ask(np.array([0.0, 1, 0]), np.array([1.0, 0, 0]), np.array([True, False, True])).tolist() == [1, 1, 0]
    v = np.r_[np.ones(10), np.zeros(10)]
    pt, lo, hi = cluster_bootstrap(np.r_[np.zeros(10), np.ones(10)], v)
    assert pt == 0.5 and lo == 0.0 and hi == 1.0, (pt, lo, hi)
    pt, lo, hi = cluster_bootstrap(np.arange(20) % 4, np.ones(20))
    assert (pt, lo, hi) == (1.0, 1.0, 1.0)
    assert top2(np.array([0.1, 0.7, 0.4]))[0] == 1 and np.isclose(top2(np.array([0.1, 0.7, 0.4]))[1], 0.3)
    assert not strict_correct(np.array([0.5, 0.5, 0.1]), 0) and strict_correct(np.array([0.6, 0.5, 0.1]), 0)
    e = normalize(np.array([[1.0, 0], [0, 1.0]]))
    assert np.allclose(query_vec(e, None), normalize(np.array([1.0, 1.0])))
    assert np.allclose(query_vec(e, np.array([1.0, 0])), normalize(np.array([-1.0, 1.0])))
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "extract", "score", "plan", "analyze"])
    ap.add_argument("--rotate180", type=int, default=0)
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "extract": cmd_extract, "score": cmd_score, "plan": cmd_plan, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
