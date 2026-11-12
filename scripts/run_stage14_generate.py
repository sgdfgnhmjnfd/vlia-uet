#!/usr/bin/env python3
"""Stage 14 part A: the VLM writes the intention itself (no list), and the robot acts on it.

Subcommands: selftest | gate-m1 | probe | gen-libero | gen-ego | human-check | plan | analyze.
Pre-registration: results/prereg_stage14_generate_early.md (A1, A2, A3 run now; B fixed for later).
Rollouts run through scripts/libero_goal_ablation.py (conditions E2E-gen, E2E-gen-blank, E2E-gen-p50,
--e2e-tokens results/stage14_generate_early/e2e_gen_tokens.json --task-embeddings <WORK>/gen_embeddings.npz).
"""
from __future__ import annotations

import argparse
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
import libero_goal_ablation as G  # noqa: E402

N_TASKS = 10
N_UNIT = 30
LIB_P = (0, 10, 25, 50, 75, 100)
EGO_K = (8, 2, 4)
BLANK_K = 8
PRIMARY_P = 25
PROMPT_EGO = ('These frames are from a first-person video of a person doing one step of a task. In one short sentence, '
              'say what the person wants to achieve in this step. Start with "I want to".')
PROMPT_LIB = ('These frames show the start of someone doing a household task with objects on a table. In one short '
              'sentence, say what they want the objects to end up like. Start with "I want".')
MAX_NEW = 40
TEMPLATE = "a first-person photo of a person doing: {why}"
ENCODER = "google/siglip-so400m-patch14-384"
WORK = Path("/media/dhqg/d3/vlia_outputs/stage14")
S12_FRAMES = Path("/media/dhqg/d3/vlia_outputs/stage12/frames")
RES = ROOT / "results/stage14_generate_early"
OUT = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1")
EVAL = OUT / "eval"
TABLE = OUT / "task_embeddings_v2.npz"
VLIA = OUT / "train/intention_only/seed_0/checkpoint_025000/pretrained_model"
QUOTES = "\"'“”‘’`"
GEN_CONDS = ("E2E-gen", "E2E-gen-blank", "E2E-gen-p50")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def originals() -> dict:
    return {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}


# ----------------------------------------------------------------------
# sentences, matching, statistics (numpy only)
# ----------------------------------------------------------------------

def clean_sentence(text: str) -> str:
    """First sentence (end mark kept), quotes and whitespace stripped."""
    t = text.strip()
    m = re.search("[.!?][" + QUOTES + "]*(?=\\s|$)", t)  # closing quotes may sit between the mark and the space
    if m:
        t = t[: m.end()]
    prev = None
    while prev != t:
        prev = t
        t = t.strip().strip(QUOTES)
    return t


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def strict_top(scores: np.ndarray) -> int:
    """Arg-max, or -1 when the top is tied."""
    o = int(np.argmax(scores))
    return -1 if (scores == scores[o]).sum() > 1 else o


def strict_hit(scores: np.ndarray, true: int) -> bool:
    return bool((scores[true] > np.delete(scores, true)).all())


def nearest(sent_emb: np.ndarray, desc_emb: np.ndarray) -> tuple[int, np.ndarray]:
    cos = desc_emb @ sent_emb
    return strict_top(cos), cos


def rank_of_true(cos: np.ndarray, true: int) -> int:
    """1 + number of other candidates with cosine >= the true one (ties count against)."""
    return 1 + int((np.delete(cos, true) >= cos[true]).sum())


def strat_boot(groups, values, B: int = 10000, seed: int = 0):
    groups, values = np.asarray(groups), np.asarray(values, float)
    rng = np.random.default_rng(seed)
    total = np.zeros(B)
    for g in np.unique(groups):
        v = values[groups == g]
        total += v[rng.integers(0, len(v), size=(B, len(v)))].sum(axis=1)
    lo, hi = np.percentile(total / len(values), [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def tertiles(x: np.ndarray) -> np.ndarray:
    """Tertile index 0/1/2 by rank (ties by order)."""
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return (3 * r) // len(x)


def write_csv(path: Path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------
# encoders
# ----------------------------------------------------------------------

class TextEncoder:
    """SigLIP so400m text side, exactly as v3_stage4_task_embeddings.py (X.text_features, max_length 64)."""

    def __init__(self, device=None):
        import torch

        import extract_ladder_embeddings as X

        self.X = X
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.processor = X.load_encoder(ENCODER, self.device)

    def __call__(self, texts, template: bool) -> np.ndarray:
        texts = [TEMPLATE.format(why=t) if template else t for t in texts]
        emb, _ = self.X.text_features(self.model, self.processor, ENCODER, texts, self.device, 32)
        return emb.astype(np.float64)


class Generator:
    def __init__(self):
        import v3_stage3_mllm as M3

        self.m = M3.MLLM()
        self.resize = M3.resize448

    def __call__(self, images, prompt: str) -> tuple[str, str, int]:
        """images: PIL list (448 px). Returns (clean sentence, raw text, n prompt tokens)."""
        torch = self.m.torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.m.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = self.m.processor(text=[text], images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.m.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW)
        n = inputs["input_ids"].shape[1]
        raw = self.m.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]
        return clean_sentence(raw), raw, int(n)


def grey(k: int):
    from PIL import Image

    return [Image.new("RGB", (448, 448), (128, 128, 128)) for _ in range(k)]


def write_json(path: Path, rec: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)


# ----------------------------------------------------------------------
# LIBERO
# ----------------------------------------------------------------------

def libero_units() -> list[tuple[int, int]]:
    n = json.loads((Path("/media/dhqg/d3/vlia_outputs/stage12/score_meta.json")).read_text())["demos_per_task"]
    return [(t, d) for t in range(N_TASKS) for d in range(N_UNIT) if d < int(n[str(t)])]


def extract_p75(units) -> int:
    """Stage 12 frames for p = 75 (same prefix_indices and decoder); existing files are never touched."""
    import av

    import run_stage12_e2e as S12

    todo = [(t, d) for t, d in units if not (S12_FRAMES / f"t{t}_d{d}_p75.npy").exists()]
    if not todo:
        return 0
    snap, eps, demos, _others, _ = S12.load_meta()
    by_file = {}
    for t, d in todo:
        ep = demos[t][d]
        chunk, fi, t0, length = S12.video_loc(eps, ep)
        by_file.setdefault((chunk, fi), []).append((S12_FRAMES / f"t{t}_d{d}_p75.npy", int(round(t0 * 10)) + S12.prefix_indices(length, 75)))
    for (chunk, fi), items in sorted(by_file.items()):
        want = {int(i) for _, ids in items for i in ids}
        got = {}
        with av.open(str(Path(snap) / f"videos/{S12.CAM}/chunk-{chunk:03d}/file-{fi:03d}.mp4")) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            for frame in c.decode(s):
                i = int(round(float(frame.pts * s.time_base) * 10))
                if i in want:
                    got[i] = frame.to_ndarray(format="rgb24")
                if i >= max(want):
                    break
        if want - set(got):
            raise SystemExit(f"STOP: p75 frames missing in file {fi}")
        for path, ids in items:
            np.save(path, np.stack([got[int(i)] for i in ids]).astype(np.uint8))
    log(f"p75 frames extracted for {len(todo)} demos")
    return len(todo)


def lib_path(t, d, p):
    return WORK / "gen_libero" / f"t{t}_d{d}_p{p}.json"


def blank_path(kind):
    return WORK / f"blank_{kind}.json"


def cmd_gen_libero(args) -> int:
    units = libero_units()
    ps = [int(x) for x in args.p.split(",")]
    if 75 in ps:
        extract_p75(units)
    (WORK / "gen_libero").mkdir(parents=True, exist_ok=True)
    gen = None
    todo = [(t, d, p) for p in ps for t, d in units if not lib_path(t, d, p).exists()]
    log(f"gen-libero: {len(units)} demos x p {ps}: {len(todo)} to generate")
    t0 = time.time()
    for n, (t, d, p) in enumerate(todo, 1):
        gen = gen or Generator()
        frames = np.load(S12_FRAMES / f"t{t}_d{d}_p{p}.npy")
        s, raw, ntok = gen([gen.resize(f) for f in frames], PROMPT_LIB)
        write_json(lib_path(t, d, p), {"task": t, "demo": d, "p": p, "k": len(frames), "sentence": s, "raw": raw, "n_tokens": ntok})
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"{n}/{len(todo)} {el / n:.2f}s/query; t{t} d{d} p{p}: {s}")
    if args.blank and not blank_path("libero").exists():
        gen = gen or Generator()
        s, raw, _ = gen(grey(BLANK_K), PROMPT_LIB)
        write_json(blank_path("libero"), {"k": BLANK_K, "sentence": s, "raw": raw})
        log(f"blank (LIBERO, k={BLANK_K}): {s}")
    if todo:
        (WORK / "gen_libero_info.json").write_text(json.dumps({"seconds_per_query": (time.time() - t0) / len(todo), "n": len(todo)}))
    return 0


# ----------------------------------------------------------------------
# EgoIntent
# ----------------------------------------------------------------------

def ego_rows():
    import v3_stage3_mllm as M3

    return M3.rows_and_sets()


def ego_path(sid, k):
    return WORK / "gen_ego" / f"{sid.replace('/', '__')}__k{k}.json"


def cmd_gen_ego(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from hoi_utils import sample_exact_video_frames

    rows, sid, _vids, _texts, _cands = ego_rows()
    ks = [int(x) for x in args.k.split(",")]
    (WORK / "gen_ego").mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(rows)), key=lambda i: sid[i])
    if args.limit:
        order = order[: args.limit]
    todo = [i for i in order if any(not ego_path(sid[i], k).exists() for k in ks)]
    log(f"gen-ego: {len(order)} steps x k {ks}: {len(todo)} steps to do")
    gen = Generator()

    def load(i):
        return [gen.resize(f) for f in sample_exact_video_frames(rows[i]["video_path"])]

    t0, nq = time.time(), 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {i: pool.submit(load, i) for i in todo[:6]}
        for n, i in enumerate(todo, 1):
            for j in todo[n:n + 5]:
                futs.setdefault(j, pool.submit(load, j))
            frames = futs.pop(i).result()
            for k in ks:
                if ego_path(sid[i], k).exists():
                    continue
                s, raw, ntok = gen(frames[:k], PROMPT_EGO)
                write_json(ego_path(sid[i], k), {"sample_id": sid[i], "k": k, "sentence": s, "raw": raw, "n_tokens": ntok})
                nq += 1
            if n % 20 == 0 or n == len(todo) or n <= args.show:
                el = time.time() - t0
                log(f"{n}/{len(todo)} steps {el / max(nq, 1):.2f}s/query eta {el / n * (len(todo) - n) / 3600:.2f}h; {sid[i]}: "
                    + " | ".join(json.loads(ego_path(sid[i], k).read_text())["sentence"] for k in ks)
                    + f"  [true WHY: {rows[i]['why']}]")
    if args.blank and not blank_path("ego").exists():
        s, raw, _ = gen(grey(BLANK_K), PROMPT_EGO)
        write_json(blank_path("ego"), {"k": BLANK_K, "sentence": s, "raw": raw})
        log(f"blank (EgoIntent, k={BLANK_K}): {s}")
    if nq:
        (WORK / f"gen_ego_info{'_limit' if args.limit else ''}.json").write_text(
            json.dumps({"seconds_per_query": (time.time() - t0) / nq, "n_queries": nq}))
    return 0


def cmd_probe(args) -> int:
    """20 EgoIntent steps at k = 8 and 10 LIBERO demos at p = 25, into a probe folder; prints the sentences."""
    from hoi_utils import sample_exact_video_frames

    gen = Generator()
    pdir = WORK / "probe"
    pdir.mkdir(parents=True, exist_ok=True)
    rows, sid, _, _, _ = ego_rows()
    order = sorted(range(len(rows)), key=lambda i: sid[i])[:20]
    out, t_ego = [], []
    for i in order:
        frames = [gen.resize(f) for f in sample_exact_video_frames(rows[i]["video_path"])]
        t0 = time.time()
        s, raw, _ = gen(frames[:8], PROMPT_EGO)
        t_ego.append(time.time() - t0)
        out.append({"set": "ego", "id": sid[i], "sentence": s, "true": rows[i]["why"]})
        log(f"EGO {sid[i]}: {s}  [true WHY: {rows[i]['why']}]")
    orig = originals()
    t_lib = []
    units = libero_units()
    for t, d in [units[i] for i in np.linspace(0, len(units) - 1, 10).astype(int)]:
        frames = np.load(S12_FRAMES / f"t{t}_d{d}_p25.npy")
        t0 = time.time()
        s, raw, _ = gen([gen.resize(f) for f in frames], PROMPT_LIB)
        t_lib.append(time.time() - t0)
        out.append({"set": "libero", "id": f"t{t}_d{d}_p25", "sentence": s, "true": orig[t]})
        log(f"LIBERO t{t} d{d} p25: {s}  [true: {orig[t]}]")
    n_ego = 3 * 2503
    info = {"seconds_per_query_ego": float(np.mean(t_ego[1:])), "seconds_per_query_libero": float(np.mean(t_lib)),
            "eta_ego_k248_hours": float(np.mean(t_ego[1:])) * n_ego / 3600,
            "eta_libero_6p_hours": float(np.mean(t_lib)) * 299 * 6 / 3600}
    (pdir / "probe.json").write_text(json.dumps({"info": info, "sentences": out}, indent=1))
    log(json.dumps(info))
    return 0


# ----------------------------------------------------------------------
# human checks (written before any score)
# ----------------------------------------------------------------------

def cmd_human_check(_) -> int:
    RES.mkdir(parents=True, exist_ok=True)
    orig = originals()
    if not (RES / "human_check_libero.csv").exists():
        units = libero_units()
        allk = [(t, d, p) for t, d in units for p in LIB_P]
        assert all(lib_path(*x).exists() for x in allk), "gen-libero not finished"
        pick = np.random.default_rng(0).choice(len(allk), size=60, replace=False)
        rows = []
        for i in sorted(pick):
            t, d, p = allk[i]
            rows.append({"task": t, "demo": d, "p": p, "true_description": orig[t],
                         "sentence": json.loads(lib_path(t, d, p).read_text())["sentence"], "rating": ""})
        write_csv(RES / "human_check_libero.csv", rows)
        print(f"wrote human_check_libero.csv ({len(rows)} rows)")
    rows_, sid, _, _, _ = ego_rows()
    order = sorted(range(len(rows_)), key=lambda i: sid[i])
    if not (RES / "human_check.csv").exists() and all(ego_path(sid[i], 8).exists() for i in order):
        pick = np.random.default_rng(0).choice(len(order), size=50, replace=False)
        rows = [{"sample_id": sid[order[i]], "true_why": rows_[order[i]]["why"],
                 "sentence_k8": json.loads(ego_path(sid[order[i]], 8).read_text())["sentence"], "rating": ""} for i in sorted(pick)]
        write_csv(RES / "human_check.csv", rows)
        print(f"wrote human_check.csv ({len(rows)} rows)")
    return 0


# ----------------------------------------------------------------------
# gate M1, plan
# ----------------------------------------------------------------------

def m1_check(phr_emb: dict, desc_emb: np.ndarray) -> tuple[int, int, list]:
    hits, rows = 0, []
    for (kind, t), e in sorted(phr_emb.items()):
        y, cos = nearest(e, desc_emb)
        hits += y == t
        rows.append({"phrasing": kind, "task": t, "nearest": y, "cos_own": float(cos[t])})
    return hits, len(phr_emb), rows


def cmd_gate_m1(_) -> int:
    orig = originals()
    ph = json.loads((ROOT / "results/stage11_libero_goal/phrasings.json").read_text())
    enc = TextEncoder()
    desc = enc([orig[t] for t in range(N_TASKS)], template=False)
    keys = [(k, t) for k in ("goal", "para") for t in range(N_TASKS)]
    em = enc([ph[k][str(t)] for k, t in keys], template=False)
    hits, n, rows = m1_check(dict(zip(keys, em)), desc)
    ok = hits >= 0.9 * n
    RES.mkdir(parents=True, exist_ok=True)
    write_csv(RES / "gate_m1.csv", rows)
    print(f"GATE M1 nearest-description matcher on Stage 11 phrasings: {hits}/{n}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("STOP: M1 failed")
    return 0


def plan_from(sent: dict, blank: str, units) -> dict:
    """sent[(p, t, d)] = sentence. Tokens per condition; p = 50 units with the p = 25 sentence reuse E2E-gen."""
    plan = {"tokens": {c: {} for c in GEN_CONDS}, "reuse": {"E2E-gen-p50": []}, "new": {}}
    for t, d in units:
        k = f"{t},{d}"
        plan["tokens"]["E2E-gen"][k] = sent[(25, t, d)]
        plan["tokens"]["E2E-gen-blank"][k] = blank
        plan["tokens"]["E2E-gen-p50"][k] = sent[(50, t, d)]
        if sent[(50, t, d)] == sent[(25, t, d)]:
            plan["reuse"]["E2E-gen-p50"].append(k)
    plan["new"]["E2E-gen-p50"] = [f"{t},{d}" for t, d in units if f"{t},{d}" not in set(plan["reuse"]["E2E-gen-p50"])]
    return plan


def cmd_plan(_) -> int:
    units = libero_units()
    orig = originals()
    sent = {(p, t, d): json.loads(lib_path(t, d, p).read_text())["sentence"] for p in (25, 50) for t, d in units}
    blank = json.loads(blank_path("libero").read_text())["sentence"]
    plan = plan_from(sent, blank, units)
    texts = sorted(set(orig.values()) | set(sent.values()) | {blank})
    emb = TextEncoder()(texts, template=True)
    ref = dict(zip(map(str, np.load(TABLE)["tasks"]), np.load(TABLE)["emb"]))
    e1 = max(float(np.abs(emb[texts.index(orig[t])] - ref[orig[t]]).max()) for t in range(N_TASKS))
    print(f"GATE E1 on-the-fly T1 embedding of the 10 descriptions vs token table: max abs diff {e1:.3g}: {'PASS' if e1 < 1e-6 else 'FAIL'}")
    if e1 >= 1e-6:
        raise SystemExit("STOP: E1 failed")
    np.savez(WORK / "gen_embeddings.npz", tasks=np.asarray(texts), emb=emb.astype(np.float32))
    RES.mkdir(parents=True, exist_ok=True)
    plan["E1_max_abs_diff"] = e1
    (RES / "e2e_gen_tokens.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(units)} units; blank sentence: {blank!r}; E2E-gen-p50 reuse {len(plan['reuse']['E2E-gen-p50'])}, new {len(plan['new']['E2E-gen-p50'])}")
    for t, e in ((0, 0), (4, 5), (9, 9)):
        r = json.loads((EVAL / f"episodes/I-only_t{t}_e{e}.json").read_text())
        assert r["z_int_task"] == orig[t] and r["seed"] == 1000 + e
    return 0


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------

def analyze_libero(enc, orig) -> tuple[list, dict, dict]:
    units = libero_units()
    desc = enc([orig[t] for t in range(N_TASKS)], template=False)
    sents = {(t, d, p): json.loads(lib_path(t, d, p).read_text())["sentence"] for t, d in units for p in LIB_P}
    uniq = sorted(set(sents.values()))
    E = dict(zip(uniq, enc(uniq, template=False)))
    rows, hit = [], {}
    for (t, d, p), s in sents.items():
        y, cos = nearest(E[s], desc) if s else (-1, np.zeros(N_TASKS))
        hit[(t, d, p)] = float(y == t)
        rows.append({"task": t, "demo": d, "p": p, "sentence": s, "nearest_task": y,
                     "nearest_description": orig[y] if y >= 0 else "", "cos_nearest": float(cos[y]) if y >= 0 else "",
                     "cos_true": float(cos[t]), "match": int(y == t),
                     "two_way_0v1": (int(cos[t] > cos[1 - t]) if t in (0, 1) else "")})
    tasks = np.array([t for t, d in units])
    acc = []
    for p in LIB_P:
        h = np.array([hit[(t, d, p)] for t, d in units])
        pt, lo, hi = strat_boot(tasks, h)
        r01 = [r for r in rows if r["p"] == p and r["task"] in (0, 1)]
        w = strat_boot([r["task"] for r in r01], [r["two_way_0v1"] for r in r01])
        acc.append({"p": p, "n": len(units), "open_ended_10way": pt, "ci95_lo": lo, "ci95_hi": hi,
                    "within_scene_0v1": w[0], "within_scene_ci95_lo": w[1], "within_scene_ci95_hi": w[2], "n_0v1": len(r01),
                    "n_distinct_sentences": len({sents[(t, d, p)] for t, d in units})})
    acc_p = {r["p"]: r for r in acc}
    d6 = strat_boot(tasks, [hit[(t, d, 100)] - hit[(t, d, 25)] for t, d in units])
    v = {"GA4": {"pass": acc_p[100]["ci95_lo"] > 0.5, "accuracy": acc_p[100]["open_ended_10way"], "ci95": [acc_p[100]["ci95_lo"], acc_p[100]["ci95_hi"]],
                 "rule": "open-ended 10-way at p=100 CI lower bound > 50%"},
         "GA5": {"pass": acc_p[50]["ci95_lo"] > 0.3, "accuracy": acc_p[50]["open_ended_10way"], "ci95": [acc_p[50]["ci95_lo"], acc_p[50]["ci95_hi"]],
                 "rule": "open-ended 10-way at p=50 CI lower bound > 30%"},
         "GA6": {"pass": d6[1] > 0, "delta": d6[0], "ci95": [d6[1], d6[2]], "rule": "hit(p=100) - hit(p=25) CI lower bound > 0"}}
    return rows, {"acc": acc, "verdicts": v, "hit": hit, "sents": sents}, {"desc": desc, "E": E}


def analyze_robot(lib, orig) -> dict | None:
    plan_path = RES / "e2e_gen_tokens.json"
    if not plan_path.exists():
        return None
    plan = json.loads(plan_path.read_text())
    units = libero_units()

    def ep(c, t, e):
        p = EVAL / f"episodes/{c}_t{t}_e{e}.json"
        return json.loads(p.read_text()) if p.exists() else None

    succ, src = {}, {}
    for c in ("E2E-gen", "E2E-gen-blank", "E2E-gen-p50"):
        vals, ok, nre = [], True, 0
        for t, e in units:
            k = f"{t},{e}"
            if c == "E2E-gen-p50" and k in set(plan["reuse"]["E2E-gen-p50"]):
                r = ep("E2E-gen", t, e)
                nre += 1
            else:
                r = ep(c, t, e)
                if r is not None:
                    assert r["z_int_task"] == plan["tokens"][c][k] and r["seed"] == 1000 + e and r["init_state_id"] == e, (c, k)
            if r is None:
                ok = False
                break
            vals.append(float(r["success"]))
        if ok:
            succ[c] = np.array(vals)
            src[c] = {"reused": nre, "new": len(units) - nre}
    if "E2E-gen" not in succ:
        return None
    for name, f in (("I-only", "I-only"), ("Stage12 E2E", "E2E"), ("I-wrong", "I-wrong")):
        succ[name] = np.array([float(ep(f, t, e)["success"]) for t, e in units])
    # snap variant: token-matrix lookup of the nearest description (pending until matrix_seed0.npz exists)
    snap_status = "pending (matrix_seed0.npz not built)"
    try:
        import token_matrix as TM

        m = TM.load_matrix(0)
        near = {f"{t},{e}": orig[r] for (t, e) in units for r in [int(np.argmax(lib["E_desc_cos"][(t, e)]))]}
        look = TM.lookup(m, near, orig)
        succ["snap (matrix)"] = np.array([look[f"{t},{e}"] for t, e in units])
        snap_status = "done"
    except (SystemExit, FileNotFoundError, OSError, ImportError) as exc:
        snap_status = f"pending ({exc})"
    summary = []
    for c, s in succ.items():
        k, n = int(s.sum()), len(units)
        lo, hi = G.wilson(k, n)
        summary.append({"condition": c, "successes": k, "n": n, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
                        **({f"episodes_{a}": b for a, b in src[c].items()} if c in src else {})})
    paired = []
    for a, b in (("E2E-gen", "I-wrong"), ("E2E-gen", "I-only"), ("E2E-gen", "Stage12 E2E"), ("E2E-gen", "E2E-gen-blank"),
                 ("E2E-gen-p50", "E2E-gen"), ("E2E-gen-p50", "I-wrong")):
        if a in succ and b in succ:
            x = G.paired_bootstrap(succ[a] - succ[b])
            paired.append({"rule": "GA3" if (a, b) == ("E2E-gen", "I-wrong") else "descriptive", "comparison": f"{a} - {b}",
                           "delta": x[0], "ci95_lo": x[1], "ci95_hi": x[2], "n": len(units)})
    match = np.array([lib["hit"][(t, e, PRIMARY_P)] for t, e in units])
    cos_true = np.array([lib["cos_true"][(t, e, PRIMARY_P)] for t, e in units])
    ter = tertiles(cos_true)
    by_match = {m_: {"n": int((match == m_).sum()), "E2E_gen_successes": int(succ["E2E-gen"][match == m_].sum())} for m_ in (0.0, 1.0)}
    by_ter = [{"tertile": i, "n": int((ter == i).sum()), "cos_true_min": float(cos_true[ter == i].min()), "cos_true_max": float(cos_true[ter == i].max()),
               "E2E_gen_successes": int(succ["E2E-gen"][ter == i].sum())} for i in range(3)]
    return {"summary": summary, "paired": paired, "episodes": src, "snap": snap_status, "by_text_match": by_match,
            "by_cos_tertile": by_ter, "text_match_p25": float(match.mean())}


def analyze_ego(enc) -> dict | None:
    import run_scene_cv as CV

    rows, sid, vids, texts, cands = ego_rows()
    order = sorted(range(len(rows)), key=lambda i: sid[i])
    have = {k: all(ego_path(sid[i], k).exists() for i in order) for k in EGO_K}
    if not any(have.values()):
        return None
    vids = np.asarray(vids)
    valid = np.array([cands[i][0] >= 0 for i in range(len(rows))])
    ev = np.array([i for i in order if valid[i]])
    uniq_why = sorted(set(texts))
    W = dict(zip(uniq_why, enc(uniq_why, template=False)))
    sent = {(i, k): json.loads(ego_path(sid[i], k).read_text())["sentence"] for k in EGO_K if have[k] for i in order}
    blank = json.loads(blank_path("ego").read_text())["sentence"] if blank_path("ego").exists() else None
    us = sorted(set(sent.values()) | ({blank} if blank is not None else set()))
    S = dict(zip(us, enc(us, template=False)))
    video_whys = {}
    for i in range(len(rows)):
        video_whys.setdefault(vids[i], set()).add(texts[i])

    def hits_mrr(i, s):
        e = S[s]
        c5 = np.array([W[texts[int(c)]] @ e for c in cands[i]])
        allw = sorted(video_whys[vids[i]])
        call = np.array([W[w] @ e for w in allw])
        return float(strict_hit(c5, 0)), 1.0 / rank_of_true(call, allw.index(texts[i]))

    out, per = [], {}
    import v3_common as V

    data = V.load_data()
    mf = V.f_metrics(data)
    fpos = {s: j for j, s in enumerate(data.sample_ids)}
    for k in EGO_K:
        if not have[k]:
            continue
        h, rr = zip(*[hits_mrr(i, sent[(i, k)]) for i in ev])
        h, rr = np.array(h), np.array(rr)
        per[k] = h
        a = CV.video_bootstrap(vids[ev], h, np.ones(len(ev)))
        m = CV.video_bootstrap(vids[ev], rr, np.ones(len(ev)))
        f5 = np.array([mf[k]["kway5"][fpos[sid[i]]] for i in ev]) if k in mf else None
        out.append({"k": k, "n_steps": len(ev), "open_ended_5way": a[0], "ci95_lo": a[1], "ci95_hi": a[2],
                    "mrr_same_video": m[0], "mrr_ci95_lo": m[1], "mrr_ci95_hi": m[2],
                    "F_5way_same_steps": float(f5.mean()) if f5 is not None else ""})
    verdicts = {}
    if 8 in per:
        a = next(r for r in out if r["k"] == 8)
        verdicts["GA1"] = {"pass": a["ci95_lo"] > 0.2, "accuracy": a["open_ended_5way"], "ci95": [a["ci95_lo"], a["ci95_hi"]],
                           "rule": "open-ended 5-way k=8 CI lower bound > 20%"}
        if blank is not None:
            hb = np.array([hits_mrr(i, blank)[0] for i in ev])
            d = CV.video_bootstrap(vids[ev], per[8] - hb, np.ones(len(ev)))
            out.append({"k": "8 (blank)", "n_steps": len(ev), "open_ended_5way": float(hb.mean())})
            verdicts["GA2"] = {"pass": d[1] > 0, "delta": d[0], "ci95": [d[1], d[2]], "rule": "video - blank, k=8, CI lower bound > 0"}
    # Stage 3 multiple-choice answers on the same steps (strict top-1 of the letter scores)
    s3 = Path("/media/dhqg/d3/vlia_outputs/v3_mllm/scores")
    for k in (2, 8):
        f = [s3 / f"{sid[i].replace('/', '__')}__k{k}.json" for i in ev]
        if all(p.exists() for p in f):
            h = np.array([float(strict_hit(np.asarray(json.loads(p.read_text())["score"]), 0)) for p in f])
            out.append({"k": f"{k} (Stage 3 Qwen multiple choice)", "n_steps": len(ev), "open_ended_5way": float(h.mean())})
    sent_rows = [{"sample_id": sid[i], "video_uid": vids[i], "true_why": texts[i], "sentence_k8": sent[(i, 8)]} for i in order] if have[8] else []
    return {"rows": out, "verdicts": verdicts, "sentences": sent_rows, "blank": blank}


def cmd_analyze(_) -> int:
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    cmd_human_check(None)  # the human-check files exist before any score is computed
    orig = originals()
    enc = TextEncoder()
    rows, lib, embs = analyze_libero(enc, orig)
    write_csv(RES / "libero_sentences.csv", rows)
    write_csv(RES / "open_ended_early.csv", lib["acc"])
    lib["cos_true"] = {(r["task"], r["demo"], r["p"]): r["cos_true"] for r in rows}
    lib["E_desc_cos"] = {(t, d): embs["desc"] @ embs["E"][lib["sents"][(t, d, PRIMARY_P)]] for t, d in libero_units()
                         if lib["sents"][(t, d, PRIMARY_P)]}
    robot = analyze_robot(lib, orig)
    ego = analyze_ego(enc)
    verdicts = dict(lib["verdicts"])
    lines = ["# Stage 14 part A: the model writes the intention", "", "## A3. Open-ended early intention (LIBERO-10, 299 demos, nearest of the 10 descriptions, scoring only)", "",
             "| p | open-ended 10-way [95% CI] | within-scene 0 vs 1 [95% CI] | distinct sentences |", "|---:|---|---|---:|"]
    lines += [f"| {r['p']} | {100 * r['open_ended_10way']:.1f} [{100 * r['ci95_lo']:.1f}; {100 * r['ci95_hi']:.1f}] | "
              f"{100 * r['within_scene_0v1']:.1f} [{100 * r['within_scene_ci95_lo']:.1f}; {100 * r['within_scene_ci95_hi']:.1f}] | {r['n_distinct_sentences']} |" for r in lib["acc"]]
    if robot:
        r3 = next(r for r in robot["paired"] if r["rule"] == "GA3")
        verdicts["GA3"] = {"pass": r3["ci95_lo"] > 0, "delta": r3["delta"], "ci95": [r3["ci95_lo"], r3["ci95_hi"]],
                           "rule": "success(E2E-gen) - success(I-wrong) CI lower bound > 0 (paired_bootstrap, 299 units)"}
        write_csv(RES / "summary.csv", robot["summary"])
        rl = ["## A2. Robot (299 units)", "", "| condition | successes / n | Wilson 95% | episodes reused / new |", "|---|---:|---|---|"]
        rl += [f"| {r['condition']} | {r['successes']} / {r['n']} | [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] | {r.get('episodes_reused', '')} / {r.get('episodes_new', '')} |"
               for r in robot["summary"]]
        rl += ["", "| rule | comparison | delta [95% CI] |", "|---|---|---|"]
        rl += [f"| {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in robot["paired"]]
        rl += ["", f"Text match at p=25 (nearest description = true task): {100 * robot['text_match_p25']:.1f}%.",
               f"E2E-gen successes by text match: {json.dumps(robot['by_text_match'])}.",
               f"By tertile of cos(s, true description): {json.dumps(robot['by_cos_tertile'])}.", f"Snap variant: {robot['snap']}.", ""]
        (RES / "libero_summary.md").write_text("\n".join(rl))
        lines += [""] + rl
    if ego:
        verdicts.update(ego["verdicts"])
        write_csv(RES / "ego_open_ended.csv", ego["rows"])
        if ego["sentences"]:
            write_csv(RES / "ego_sentences_k8.csv", ego["sentences"])
        lines += ["## A1. EgoIntent (development steps, open-ended, scoring vs the 5 F candidates)", "", "| k | open-ended 5-way | MRR (same video) | F 5-way, same steps |", "|---|---|---|---:|"]
        for r in ego["rows"]:
            a = f"{100 * r['open_ended_5way']:.1f}" + (f" [{100 * r['ci95_lo']:.1f}; {100 * r['ci95_hi']:.1f}]" if "ci95_lo" in r else "")
            m = f"{r['mrr_same_video']:.3f} [{r['mrr_ci95_lo']:.3f}; {r['mrr_ci95_hi']:.3f}]" if "mrr_same_video" in r else ""
            f = f"{100 * r['F_5way_same_steps']:.1f}" if r.get("F_5way_same_steps", "") != "" else ""
            lines.append(f"| {r['k']} | {a} | {m} | {f} |")
        lines += ["", f"Blank-control sentence (EgoIntent): {ego['blank']!r}", ""]
    lines += ["## Rules", "", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for k in sorted(verdicts):
        v = verdicts[k]
        lines.append(f"| {k}: {v['rule']} | {v.get('accuracy', v.get('delta')):+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {'PASS' if v['pass'] else 'FAIL'} |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    info = {f.stem: json.loads(f.read_text()) for f in WORK.glob("*_info*.json")}
    plan = json.loads((RES / "e2e_gen_tokens.json").read_text()) if (RES / "e2e_gen_tokens.json").exists() else {}
    secs = [json.loads(p.read_text())["seconds"] for c in GEN_CONDS for p in (EVAL / "episodes").glob(f"{c}_t*_e*.json")]
    (RES / "config.json").write_text(json.dumps({
        "git_commit": commit, "model": "Qwen/Qwen2.5-VL-3B-Instruct", "prompts": {"ego": PROMPT_EGO, "libero": PROMPT_LIB},
        "max_new_tokens": MAX_NEW, "template": TEMPLATE, "encoder": ENCODER, "vlia_checkpoint": str(VLIA),
        "gpu_queue_order": ["Stage 15 (finished)", "probe", "gen-libero p=0,10,25,50,75,100 + blank", "plan", "smoke",
                            "E2E-gen p=25", "E2E-gen-blank", "E2E-gen-p50 (changed units)", "gen-ego k=8,2,4 + blank", "analyze",
                            "then token matrix Part 1 / Part 2 and seed 1 (not restarted by this task)"],
        "generation_info": info, "seconds_per_episode": float(np.mean(secs)) if secs else None, "n_rollouts": len(secs),
        "E1_max_abs_diff": plan.get("E1_max_abs_diff"), "blank_libero": json.loads(blank_path("libero").read_text())["sentence"] if blank_path("libero").exists() else None,
        "verdicts": verdicts, "robot": robot and {k: v for k, v in robot.items() if k != "summary"}, "analyze_seconds": time.time() - t0}, indent=2, default=str))
    print((RES / "summary.md").read_text())
    return 0


# ----------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------

def cmd_selftest(_) -> int:
    assert clean_sentence('  "I want the mug in the microwave." Then I close it.') == "I want the mug in the microwave."
    assert clean_sentence("I want to cut the bread") == "I want to cut the bread"
    assert clean_sentence("“I want both pots on the stove!” ok") == "I want both pots on the stove!"
    assert clean_sentence("I want 3.5 cups. More") == "I want 3.5 cups."
    assert clean_sentence('""') == "" and clean_sentence("  ") == ""
    assert strict_top(np.array([0.1, 0.5, 0.5])) == -1 and strict_top(np.array([0.1, 0.6, 0.5])) == 1
    assert strict_hit(np.array([0.6, 0.5]), 0) and not strict_hit(np.array([0.5, 0.5]), 0)
    d = normalize(np.eye(3))
    y, cos = nearest(normalize(np.array([0.2, 0.9, 0.1])), d)
    assert y == 1 and np.isclose(cos[1], max(cos))
    assert rank_of_true(np.array([0.3, 0.5, 0.3, 0.1]), 0) == 3 and rank_of_true(np.array([0.9, 0.5]), 0) == 1
    pt, lo, hi = strat_boot(np.r_[np.zeros(10), np.ones(10)], np.r_[np.ones(10), np.zeros(10)])
    assert (pt, lo, hi) == (0.5, 0.5, 0.5)
    assert tertiles(np.array([5.0, 1, 3, 2, 4, 6])).tolist() == [2, 0, 1, 0, 1, 2]
    units = [(0, 0), (1, 3)]
    sent = {(25, 0, 0): "a", (50, 0, 0): "a", (25, 1, 3): "b", (50, 1, 3): "c"}
    plan = plan_from(sent, "blank", units)
    assert plan["reuse"]["E2E-gen-p50"] == ["0,0"] and plan["new"]["E2E-gen-p50"] == ["1,3"]
    assert plan["tokens"]["E2E-gen-blank"] == {"0,0": "blank", "1,3": "blank"} and plan["tokens"]["E2E-gen-p50"]["1,3"] == "c"
    hits, n, _ = m1_check({("goal", 0): normalize(np.array([1.0, 0.1, 0])), ("para", 1): normalize(np.array([0.9, 0.2, 0]))}, d)
    assert (hits, n) == (1, 2)
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "gate-m1", "probe", "gen-libero", "gen-ego", "human-check", "plan", "analyze"])
    ap.add_argument("--p", default="0,10,25,50,75,100")
    ap.add_argument("--k", default="8,2,4")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "gate-m1": cmd_gate_m1, "probe": cmd_probe, "gen-libero": cmd_gen_libero,
            "gen-ego": cmd_gen_ego, "human-check": cmd_human_check, "plan": cmd_plan, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
