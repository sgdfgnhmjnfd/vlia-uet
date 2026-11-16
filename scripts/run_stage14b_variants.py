#!/usr/bin/env python3
"""Stage 14b: better intention-writing prompts on LIBERO video (no training, no robot).

Subcommands: selftest | gen (--variant G0|G1|G2|O1 --split cal|test --p ...) | blank (--variant) | choose | analyze.
Pre-registration: results/prereg_stage14b_generate_variants.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage14_generate as S  # noqa: E402

N_TASKS = 10
CAL_D = (30, 50)  # Stage 12 calibration demos 30..49
CAL_P = (25, 50)
LIB_P = S.LIB_P
VARIANTS_G = ("G0", "G1", "G2")
VARIANTS = VARIANTS_G + ("O1",)
EXAMPLE_UNITS = ((2, 0), (5, 0), (9, 0))
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW = S.MAX_NEW
MAX_NEW_LIST = 60
PROMPT_G0 = S.PROMPT_LIB
PROMPT_G1 = ("These frames, in time order, show a robot arm starting a tabletop task. Look at which object the gripper moves "
             "towards or holds, and where things are being moved. In one short sentence, say what the final arrangement of the "
             "objects should be. Name the objects and the place. Start with 'I want'.")
G1_LAST = "Start with 'I want'."
PROMPT_G2_LIST = "List the objects on the table and the furniture or containers, as a short comma-separated list."
PROMPT_G2 = ("Objects in the scene: {list}. These frames, in time order, show a robot arm starting a tabletop task. In one short "
             "sentence, say what the final arrangement of the objects should be, using the names above. Start with 'I want'.")
BDDL_DIR = Path("/home/dhqg/lerobot/.venv/lib/python3.12/site-packages/libero/libero/bddl_files/libero_10")
WORK = Path("/media/dhqg/d3/vlia_outputs/stage14b")
RES = ROOT / "results/stage14b_generate_variants"
S12_REC = ROOT / "results/stage12_end_to_end/recognition.csv"
C_ACC = {0: 0.8865435356200527, 10: 0.9630606860158312, 25: 0.9973614775725593, 50: 1.0, 75: 1.0, 100: 1.0}  # Stage 15 agent primary
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
log = S.log


# ----------------------------------------------------------------------
# units, scenes, prompts
# ----------------------------------------------------------------------

def demos_per_task() -> dict:
    n = json.loads(Path("/media/dhqg/d3/vlia_outputs/stage12/score_meta.json").read_text())["demos_per_task"]
    return {int(k): int(v) for k, v in n.items()}


def cal_units(n=None) -> list[tuple[int, int]]:
    n = n or demos_per_task()
    return [(t, d) for t in range(N_TASKS) for d in range(*CAL_D) if d < n[t]]


def test_units() -> list[tuple[int, int]]:
    return S.libero_units()


def bddl_file(desc: str, bddl_dir: Path = BDDL_DIR) -> Path:
    hits = [f for f in sorted(bddl_dir.glob("*.bddl")) if f.stem.endswith("_" + desc.replace(" ", "_"))]
    if len(hits) != 1:
        raise SystemExit(f"STOP: {len(hits)} BDDL files for {desc!r}")
    return hits[0]


def bddl_instances(text: str) -> list[str]:
    """Instance names declared in (:fixtures ...) then (:objects ...), in file order."""
    out = []
    for block in ("fixtures", "objects"):
        m = re.search(r"\(:" + block + r"\b(.*?)\n\s*\)", text, re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            if " - " in line:
                out += line.split(" - ")[0].split()
    return out


def plural(w: str) -> str:
    return w + ("es" if re.search(r"(s|x|ch|sh)$", w) else "s")


def scene_names(instances: list[str]) -> str:
    names = [re.sub(r"_\d+$", "", i).replace("_", " ") for i in instances]
    order = list(dict.fromkeys(names))
    parts = []
    for nm in order:
        c = names.count(nm)
        parts.append(nm if c == 1 else f"{NUM[c]} {' '.join(nm.split()[:-1] + [plural(nm.split()[-1])])}")
    return ", ".join(parts)


def scene_of(t: int, orig: dict) -> tuple[str, str]:
    """(scene id, object-name list) for task t."""
    f = bddl_file(orig[t])
    scene = f.stem[: len(f.stem) - len(orig[t].replace(" ", "_")) - 1]
    return scene, scene_names(bddl_instances(f.read_text()))


def prompt_o1(names: str) -> str:
    assert PROMPT_G1.endswith(G1_LAST)
    return PROMPT_G1[: -len(G1_LAST)] + f"Objects in the scene: {names}. " + G1_LAST


def clean_list(raw: str) -> str:
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    return line.rstrip(".").strip()


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------

def out_path(v: str, t: int, d: int, p: int) -> Path:
    if v == "G0" and d < S.N_UNIT:
        return S.lib_path(t, d, p)  # Stage 14 outputs, reused on the 299 units
    return WORK / v / f"t{t}_d{d}_p{p}.json"


def list_path(t: int, d: int) -> Path:
    return WORK / "G2_lists" / f"t{t}_d{d}.json"


def blank_file(v: str, scene: str | None = None) -> Path:
    if v == "G0":
        return S.blank_path("libero")
    return WORK / "blank" / (f"{v}.json" if scene is None else f"{v}__{scene}.json")


class Writer:
    def __init__(self):
        self.g = S.Generator()

    def __call__(self, images, prompt, max_new=MAX_NEW):
        m = self.g.m
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = m.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
        inputs = m.processor(text=[text], images=images, return_tensors="pt").to("cuda")
        with m.torch.inference_mode():
            out = m.model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
        n = inputs["input_ids"].shape[1]
        return m.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0]


def frames(t, d, p):
    import v3_stage3_mllm as M3

    return [M3.resize448(f) for f in np.load(S.S12_FRAMES / f"t{t}_d{d}_p{p}.npy")]


def g2_list(w, t, d) -> str:
    path = list_path(t, d)
    if path.exists():
        return json.loads(path.read_text())["list"]
    raw = w(frames(t, d, 0)[:1], PROMPT_G2_LIST, MAX_NEW_LIST)
    S.write_json(path, {"task": t, "demo": d, "list": clean_list(raw), "raw": raw})
    return clean_list(raw)


def prompt_for(v, t, d, w, orig, scenes) -> tuple[str, dict]:
    if v == "G0":
        return PROMPT_G0, {}
    if v == "G1":
        return PROMPT_G1, {}
    if v == "G2":
        lst = g2_list(w, t, d)
        return PROMPT_G2.format(list=lst), {"list": lst}
    if v == "O1":
        return prompt_o1(scenes[t][1]), {"scene": scenes[t][0], "names": scenes[t][1]}
    raise ValueError(v)


def cmd_gen(args) -> int:
    v = args.variant
    units = cal_units() if args.split == "cal" else test_units()
    ps = [int(x) for x in args.p.split(",")]
    if v == "G0" and args.split == "test":
        missing = [u for u in units for p in ps if not out_path(v, *u, p).exists()]
        assert not missing, missing[:3]
        log("G0 test: Stage 14 outputs reused")
        return 0
    orig = S.originals()
    scenes = {t: scene_of(t, orig) for t in range(N_TASKS)}
    (WORK / v).mkdir(parents=True, exist_ok=True)
    (WORK / "G2_lists").mkdir(parents=True, exist_ok=True)
    todo = [(t, d, p) for p in ps for t, d in units if not out_path(v, t, d, p).exists()]
    log(f"gen {v} {args.split}: {len(units)} demos x p {ps}: {len(todo)} to generate")
    w, t0 = None, time.time()
    for n, (t, d, p) in enumerate(todo, 1):
        w = w or Writer()
        prompt, extra = prompt_for(v, t, d, w, orig, scenes)
        raw = w(frames(t, d, p), prompt)
        S.write_json(out_path(v, t, d, p), {"variant": v, "task": t, "demo": d, "p": p, "sentence": S.clean_sentence(raw), "raw": raw, **extra})
        if n % 25 == 0 or n == len(todo):
            log(f"{n}/{len(todo)} {(time.time() - t0) / n:.2f}s/query; t{t} d{d} p{p}: {S.clean_sentence(raw)}")
    if todo:
        info = WORK / "gpu_seconds.json"
        rec = json.loads(info.read_text()) if info.exists() else {}
        rec[f"{v}_{args.split}"] = rec.get(f"{v}_{args.split}", 0) + time.time() - t0
        info.write_text(json.dumps(rec, indent=1))
    return 0


def cmd_blank(args) -> int:
    v = args.variant
    (WORK / "blank").mkdir(parents=True, exist_ok=True)
    w = None
    orig = S.originals()
    if v == "O1":
        jobs = {sc: prompt_o1(names) for sc, names in {scene_of(t, orig) for t in range(N_TASKS)}}
    elif v == "G1":
        jobs = {None: PROMPT_G1}
    elif v == "G2":
        jobs = {None: None}
    else:
        assert blank_file("G0").exists()
        return 0
    for sc, prompt in jobs.items():
        if blank_file(v, sc).exists():
            continue
        w = w or Writer()
        extra = {}
        if v == "G2":
            lraw = w(S.grey(1), PROMPT_G2_LIST, MAX_NEW_LIST)
            extra = {"list": clean_list(lraw), "list_raw": lraw}
            prompt = PROMPT_G2.format(list=extra["list"])
        raw = w(S.grey(S.BLANK_K), prompt)
        S.write_json(blank_file(v, sc), {"variant": v, "scene": sc, "k": S.BLANK_K, "sentence": S.clean_sentence(raw), "raw": raw, **extra})
        log(f"blank {v} {sc or ''}: {S.clean_sentence(raw)}")
    return 0


# ----------------------------------------------------------------------
# scoring and selection
# ----------------------------------------------------------------------

def load_sents(v, units, ps) -> dict:
    return {(t, d, p): json.loads(out_path(v, t, d, p).read_text())["sentence"] for t, d in units for p in ps}


def score(sents: dict, enc, desc) -> dict:
    """(t, d, p) -> (nearest strict, cos vector)."""
    uniq = sorted(set(sents.values()))
    E = dict(zip(uniq, enc(uniq, template=False)))
    return {k: S.nearest(E[s], desc) if s else (-1, np.zeros(N_TASKS)) for k, s in sents.items()}


def select(cal_acc: dict) -> tuple[str, dict]:
    """cal_acc[v][p] -> accuracy. Highest mean over CAL_P; ties to the lower number."""
    means = {v: float(np.mean([cal_acc[v][p] for p in CAL_P])) for v in VARIANTS_G if v in cal_acc}
    best = max(means.values())
    return min(v for v, m in means.items() if m == best), means


def cmd_choose(_) -> int:
    orig = S.originals()
    enc = S.TextEncoder()
    desc = enc([orig[t] for t in range(N_TASKS)], template=False)
    units = cal_units()
    rows, acc = [], {}
    for v in VARIANTS:
        sc = score(load_sents(v, units, CAL_P), enc, desc)
        for p in CAL_P:
            a = float(np.mean([sc[(t, d, p)][0] == t for t, d in units]))
            acc.setdefault(v, {})[p] = a
            rows.append({"variant": v, "family": "G+obj" if v == "O1" else "G", "p": p, "accuracy": a, "n": len(units)})
    chosen, means = select(acc)
    RES.mkdir(parents=True, exist_ok=True)
    S.write_csv(RES / "calibration.csv", rows)
    rec = {"G": chosen, "G+obj": "O1", "mean_cal_accuracy_p25_p50": means, "O1_mean": float(np.mean([acc["O1"][p] for p in CAL_P])),
           "rule": "highest mean of the p=25 and p=50 calibration accuracies; ties to the lower number; G3 not run (does not fit on the GPU)",
           "n_calibration": len(units)}
    (RES / "chosen.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    return 0


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def snap_from(nearest: dict, cos: dict, units, v: str, orig: dict) -> dict:
    """e2e_sup_tokens.json format. nearest/cos keyed (t, d, p)."""
    out = {"tokens": {}, "pred": {}, "reuse": {}, "new": {}}
    for p in LIB_P:
        c = f"E2E-snap-{v}-p{p}"
        out["tokens"][c], out["pred"][c], out["reuse"][c], out["new"][c] = {}, {}, [], []
        for t, d in units:
            y = nearest[(t, d, p)]
            y = int(np.argmax(cos[(t, d, p)])) if y < 0 else y  # exact tie: lowest tied index
            k = f"{t},{d}"
            out["tokens"][c][k], out["pred"][c][k] = orig[y], y
            (out["reuse"] if y == t else out["new"])[c].append(k)
    return out


def agreement(nearest: dict, units, true) -> dict:
    ag = np.array([nearest[(t, d, 25)] == nearest[(t, d, 50)] for t, d in units])
    hit50 = np.array([nearest[(t, d, 50)] == t for t, d in units], float)
    return {"n_agree": int(ag.sum()), "n_disagree": int((~ag).sum()),
            "acc_p50_agree": float(hit50[ag].mean()) if ag.any() else float("nan"),
            "acc_p50_disagree": float(hit50[~ag].mean()) if (~ag).any() else float("nan")}


def robot_g0() -> dict:
    """Stage 14 successes per unit: E2E-gen (p = 25) and E2E-gen-p50 (reused units take E2E-gen)."""
    plan = json.loads((ROOT / "results/stage14_generate_early/e2e_gen_tokens.json").read_text())
    reuse = set(plan["reuse"]["E2E-gen-p50"])
    out = {}
    for t, e in test_units():
        k = f"{t},{e}"
        r25 = json.loads((S.EVAL / f"episodes/E2E-gen_t{t}_e{e}.json").read_text())
        r50 = r25 if k in reuse else json.loads((S.EVAL / f"episodes/E2E-gen-p50_t{t}_e{e}.json").read_text())
        assert r25["z_int_task"] == plan["tokens"]["E2E-gen"][k] and r50["z_int_task"] == plan["tokens"]["E2E-gen-p50"][k]
        out[(t, e)] = {25: bool(r25["success"]), 50: bool(r50["success"])}
    return out


def f_reference() -> dict:
    import csv

    return {int(r["p"]): float(r["accuracy"]) for r in csv.DictReader(S12_REC.open()) if r["context"] == "C2" and r["template"] == "T1"}


def fmt(x):
    return f"{100 * x:.1f}"


def cmd_analyze(_) -> int:
    orig = S.originals()
    chosen = json.loads((RES / "chosen.json").read_text())
    cg = chosen["G"]
    shown = list(dict.fromkeys(["G0", cg, "O1"]))
    enc = S.TextEncoder()
    desc = enc([orig[t] for t in range(N_TASKS)], template=False)
    units = test_units()
    tasks = np.array([t for t, d in units])
    scenes = {t: scene_of(t, orig) for t in range(N_TASKS)}
    res, hits = {}, {}
    for v in shown:
        sents = load_sents(v, units, LIB_P)
        sc = score(sents, enc, desc)
        res[v] = {"sents": sents, "near": {k: y for k, (y, _) in sc.items()}, "cos": {k: c for k, (_, c) in sc.items()}}
        hits[v] = {p: np.array([float(res[v]["near"][(t, d, p)] == t) for t, d in units]) for p in LIB_P}
    # blank controls: one sentence per unit (per scene for O1)
    blank_s = {}
    for v in shown:
        blank_s[v] = {(t, d): json.loads(blank_file(v, scenes[t][0] if v == "O1" else None).read_text())["sentence"] for t, d in units}
    bsc = {v: score({(t, d, 0): s for (t, d), s in blank_s[v].items()}, enc, desc) for v in shown}
    bhit = {v: np.array([float(bsc[v][(t, d, 0)][0] == t) for t, d in units]) for v in shown}
    F, rows, md = f_reference(), [], []
    for v in shown:
        for p in LIB_P:
            a = S.strat_boot(tasks, hits[v][p])
            rows.append({"variant": v, "family": "G+obj" if v == "O1" else "G", "p": p, "accuracy": a[0], "ci95_lo": a[1], "ci95_hi": a[2], "n": len(units),
                         "starts_I_want": float(np.mean([res[v]["sents"][(t, d, p)].startswith("I want") for t, d in units])),
                         "n_distinct": len({res[v]["sents"][(t, d, p)] for t, d in units}),
                         "mean_cos_true": float(np.mean([res[v]["cos"][(t, d, p)][t] for t, d in units]))})
        b = S.strat_boot(tasks, bhit[v])
        rows.append({"variant": f"{v}-blank", "family": "G+obj" if v == "O1" else "G", "p": "", "accuracy": b[0], "ci95_lo": b[1], "ci95_hi": b[2], "n": len(units)})
    for p in LIB_P:
        rows.append({"variant": "F (Stage 12 C2/T1)", "family": "reference", "p": p, "accuracy": F.get(p, ""), "n": 299})
        rows.append({"variant": "C (Stage 15 agent primary)", "family": "reference", "p": p, "accuracy": C_ACC[p], "n": 379})
    RES.mkdir(parents=True, exist_ok=True)
    S.write_csv(RES / "accuracy_by_p.csv", rows)
    acc = {(r["variant"], r["p"]): r for r in rows}
    # rules
    pb = __import__("libero_goal_ablation").paired_bootstrap
    gb1 = pb(hits[cg][25] - hits["G0"][25])
    gb2 = acc[(cg, 50)]
    gb3 = pb(hits[cg][50] - bhit[cg])
    rules = {"GB-1": {"rule": f"hit({cg}) - hit(G0) at p=25, paired CI lower bound > 0", "delta": gb1[0], "ci95": [gb1[1], gb1[2]], "pass": gb1[1] > 0},
             "GB-2": {"rule": f"accuracy({cg}) at p=50, CI lower bound > 30%", "accuracy": gb2["accuracy"], "ci95": [gb2["ci95_lo"], gb2["ci95_hi"]], "pass": gb2["ci95_lo"] > 0.3},
             "GB-3": {"rule": f"hit({cg}) - hit({cg}-blank) at p=50, paired CI lower bound > 0", "delta": gb3[0], "ci95": [gb3[1], gb3[2]], "pass": gb3[1] > 0}}
    desc_rows = [{"comparison": f"O1 - {cg}", "p": p, **dict(zip(("delta", "ci95_lo", "ci95_hi"), pb(hits["O1"][p] - hits[cg][p])))} for p in LIB_P]
    # per task, confusion, sentences
    pt_rows, sent_rows = [], {v: [] for v in shown}
    for v in shown:
        conf = []
        for p in LIB_P:
            for t in range(N_TASKS):
                ys = [res[v]["near"][(tt, d, p)] for tt, d in units if tt == t]
                pt_rows.append({"variant": v, "p": p, "task": t, "n": len(ys), "accuracy": float(np.mean([y == t for y in ys]))})
                conf.append({"p": p, "true_task": t, **{f"pred_{j}": ys.count(j) for j in range(N_TASKS)}, "tie": ys.count(-1)})
        S.write_csv(RES / f"confusion_{v}.csv", conf)
        for t, d in units:
            for p in LIB_P:
                sent_rows[v].append({"task": t, "demo": d, "p": p, "sentence": res[v]["sents"][(t, d, p)], "nearest_task": res[v]["near"][(t, d, p)],
                                     "true_task": t, "cos_true": float(res[v]["cos"][(t, d, p)][t])})
        S.write_csv(RES / f"sentences_{v}.csv", sent_rows[v])
    S.write_csv(RES / "per_task.csv", pt_rows)
    # snap, agreement, similarity
    for v in dict.fromkeys([cg, "O1"]):
        (RES / f"snap_tokens_{v}.json").write_text(json.dumps(snap_from(res[v]["near"], res[v]["cos"], units, v, orig), indent=1))
    agr = [{"variant": v, **agreement(res[v]["near"], units, tasks)} for v in shown]
    S.write_csv(RES / "agreement.csv", agr)
    rob = robot_g0()
    sim = []
    for p in (25, 50):
        c = np.array([res["G0"]["cos"][(t, d, p)][t] for t, d in units])
        ter = S.tertiles(c)
        for k in range(3):
            m = ter == k
            sim.append({"variant": "G0", "p": p, "tertile": k, "n": int(m.sum()), "cos_min": float(c[m].min()), "cos_max": float(c[m].max()),
                        "robot_successes": int(sum(rob[u][p] for u, mm in zip(units, m) if mm))})
    S.write_csv(RES / "similarity.csv", sim)
    # examples
    ex = ["# Stage 14b examples (units fixed in the pre-registration)", ""]
    for t, d in EXAMPLE_UNITS:
        ex += [f"## task {t}, demo {d}: {orig[t]}", "", "| p | " + " | ".join(shown) + " |", "|---:|" + "---|" * len(shown)]
        for p in LIB_P:
            ex.append(f"| {p} | " + " | ".join(f"{res[v]['sents'][(t, d, p)]} (→{res[v]['near'][(t, d, p)]})" for v in shown) + " |")
        ex.append("")
    (RES / "examples.md").write_text("\n".join(ex))
    # summary markdown
    hdr = "| variant | " + " | ".join(f"p={p}" for p in LIB_P) + " |"
    md = ["# Stage 14b: accuracy by prefix (299 units, nearest of the 10 descriptions, strict top-1)", "",
          f"Chosen on calibration: G = {cg}, G+obj = O1. G+obj is given the scene's object names and is not open-ended.", "", hdr, "|---|" + "---|" * len(LIB_P)]
    for v in shown:
        lab = f"{v} (G+obj)" if v == "O1" else (f"{v} (Stage 14)" if v == "G0" else f"{v} (chosen G)")
        md.append(f"| {lab} | " + " | ".join(f"{fmt(acc[(v, p)]['accuracy'])} [{fmt(acc[(v, p)]['ci95_lo'])}; {fmt(acc[(v, p)]['ci95_hi'])}]" for p in LIB_P) + " |")
    for v in shown:
        md.append(f"| {v} blank | " + " | ".join(fmt(acc[(f'{v}-blank', '')]['accuracy']) for _ in LIB_P) + " |")
    md.append("| F (Stage 12) | " + " | ".join(fmt(F[p]) if p in F else "–" for p in LIB_P) + " |")
    md.append("| C (Stage 15, 379 demos) | " + " | ".join(fmt(C_ACC[p]) for p in LIB_P) + " |")
    md += ["", "| rule | value | 95% CI | verdict |", "|---|---:|---|---|"]
    for k, r in rules.items():
        val = r.get("delta", r.get("accuracy"))
        md.append(f"| {k}: {r['rule']} | {val:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {'PASS' if r['pass'] else 'FAIL'} |")
    md += ["", "| descriptive | p | delta [95% CI] |", "|---|---:|---|"]
    md += [f"| {r['comparison']} | {r['p']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |" for r in desc_rows]
    md += ["", "Blank sentences: " + "; ".join(f"{v}: {sorted(set(blank_s[v].values()))}" for v in shown)]
    (RES / "accuracy_by_p.md").write_text("\n".join(md) + "\n")
    gpu = json.loads((WORK / "gpu_seconds.json").read_text()) if (WORK / "gpu_seconds.json").exists() else {}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg = {"git_commit": commit, "writer": MODEL_ID, "encoder": S.ENCODER, "scorer": "Stage 14 analyze_libero (plain text, no template), strict top-1",
           "prompts": {"G0": PROMPT_G0, "G1": PROMPT_G1, "G2_list": PROMPT_G2_LIST, "G2": PROMPT_G2, "O1": prompt_o1("{names}")},
           "scene_names": {t: scenes[t] for t in range(N_TASKS)}, "max_new_tokens": MAX_NEW, "max_new_tokens_list": MAX_NEW_LIST,
           "chosen": chosen, "rules": rules, "descriptive": desc_rows, "gpu_minutes": {k: v / 60 for k, v in gpu.items()},
           "gpu_minutes_total": sum(gpu.values()) / 60, "n_test_units": len(units), "n_calibration": len(cal_units())}
    (RES / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print("\n".join(md))
    return 0


# ----------------------------------------------------------------------

def cmd_selftest(_) -> int:
    assert scene_names(["kitchen_table", "flat_stove_1", "moka_pot_1", "moka_pot_2"]) == "kitchen table, flat stove, two moka pots"
    assert scene_names(["plate_1", "box_1", "plate_2", "box_2", "box_3"]) == "two plates, three boxes"
    assert bddl_instances("(:fixtures\n  t - table\n)\n(:objects\n  a_1 a_2 - a\n  b_1 - b\n)\n(:obj_of_interest\n a_1\n)") == ["t", "a_1", "a_2", "b_1"]
    assert prompt_o1("x, y").endswith("Name the objects and the place. Objects in the scene: x, y. Start with 'I want'.")
    assert clean_list("bowl, plate, basket.\nmore") == "bowl, plate, basket"
    assert select({"G0": {25: .3, 50: .4}, "G1": {25: .4, 50: .3}, "G2": {25: .2, 50: .2}})[0] == "G0"
    assert select({"G0": {25: .3, 50: .3}, "G1": {25: .4, 50: .3}, "G2": {25: .4, 50: .3}})[0] == "G1"
    n = {t: 30 for t in range(10)} | {1: 49, 8: 29}
    assert len(cal_units(n)) == 19 and not set(cal_units(n)) & {(t, d) for t in range(10) for d in range(30)}
    print("selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["selftest", "gen", "blank", "choose", "analyze"])
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--split", choices=["cal", "test"], default="cal")
    ap.add_argument("--p", default="25,50")
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "gen": cmd_gen, "blank": cmd_blank, "choose": cmd_choose, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
