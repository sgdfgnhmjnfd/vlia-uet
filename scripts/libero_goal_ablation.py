#!/usr/bin/env python3
"""LIBERO-10 goal-information ablation (prereg v2, Stage 4).

Same SmolVLA policy, same per-episode seed and init state, three instruction
conditions:
  correct : the task's original instruction
  empty   : ""
  wrong   : instruction of task (i + 5) mod 10
Episode e of task t uses LIBERO init state e and seed 1000 + e in every
condition. Run with ~/lerobot/.venv (the env of the 24% reference eval), cwd
~/lerobot; remaining CLI args go to lerobot's eval config parser, e.g.

  run:     python libero_goal_ablation.py --out DIR --episodes-per-task N [--probe 3] \
             --policy.path=CKPT --env.type=libero --env.task=libero_10 \
             --eval.batch_size=1 --eval.n_episodes=1 --rename_map '{...}'
  analyze: python libero_goal_ablation.py --analyze --out DIR --results RESULTS_DIR

Resumable: one JSON per (condition, task, episode) under DIR/episodes/.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

CONDITIONS = ("correct", "empty", "wrong")
# prereg v3 Stage 4 (VLIA intention-only checkpoint): empty language + z_int
# Stage 11: I-goal / I-para use the goal / para phrasing of the task as z_int text
INTENTION_CONDITIONS = ("I-only", "I-wrong", "I-goal", "I-para")
STAGE11_VLIA = ("I-only", "I-wrong", "I-goal", "I-para")
STAGE11_BASE = ("correct", "empty")
STAGE4_E0_9 = {"I-only": 20, "I-wrong": 0, "correct": 19, "empty": 0}  # gate R2
ROOT = Path(__file__).resolve().parents[1]
N_TASKS = 10
SEED0 = 1000


def wrong_task(t: int) -> int:
    return (t + 5) % N_TASKS


def intention_text(c: str, t: int, original: dict, phrasings: dict | None) -> str:
    """Text whose T1 embedding is the intention vector of condition c on task t."""
    if c == "I-only":
        return original[t]
    if c == "I-wrong":
        return original[wrong_task(t)]
    return phrasings[{"I-goal": "goal", "I-para": "para"}[c]][str(t)]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def paired_bootstrap(diff: np.ndarray, B: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------
# rollouts (lerobot venv)
# ----------------------------------------------------------------------

def run(own, rest) -> int:
    import torch

    sys.argv = [sys.argv[0], *rest]
    sys.path.insert(0, str(ROOT))
    import policies.smolvla.configuration_smolvla  # noqa: F401  (registers vlia_smolvla)
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.random_utils import set_seed

    out = Path(own.out)
    ep_dir = out / ("probe" if own.probe else "episodes")
    vid_dir = out / ("probe_videos" if own.probe else "videos")
    ep_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    @parser.wrap()
    def main(cfg: EvalPipelineConfig):
        cfg_json = json.loads(Path(cfg.policy.pretrained_path, "config.json").read_text())
        assert cfg_json["output_features"]["action"]["shape"] == [7], "action dim must be 7"
        assert cfg_json["empty_cameras"] == 1, "empty_cameras must be 1"
        set_seed(cfg.seed)
        envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
        suite = next(iter(envs))
        task_envs = envs[suite]
        assert sorted(task_envs) == list(range(N_TASKS)), sorted(task_envs)
        conditions = tuple(own.conditions.split(","))
        is_vlia = cfg_json.get("type") == "vlia_smolvla"
        if is_vlia:
            from policies.smolvla.modeling_smolvla import VLIASmolVLAPolicy

            assert set(conditions) <= set(INTENTION_CONDITIONS), conditions
            policy = VLIASmolVLAPolicy.from_pretrained(cfg.policy.pretrained_path)
            policy.to(cfg.policy.device)
            emb = np.load(own.task_embeddings)
            table = {str(t): torch.tensor(e, dtype=torch.float32) for t, e in zip(emb["tasks"], emb["emb"])}
            current_z = {"z": None}
            phrasings = json.loads(Path(own.phrasings).read_text()) if own.phrasings else None
            inner_select = policy.select_action

            def select_with_intention(batch, *a, **kw):
                batch = dict(batch)
                batch["intention"] = current_z["z"].to(policy.config.device)[None, :]
                return inner_select(batch, *a, **kw)

            policy.select_action = select_with_intention
        else:
            assert set(conditions) <= set(CONDITIONS), conditions
            policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                    "rename_observations_processor": {"rename_map": cfg.rename_map}})
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        original = {t: task_envs[t].envs[0].task_description for t in range(N_TASKS)}
        (out / "instructions.json").write_text(json.dumps(
            {"original": original, "wrong_map": {t: wrong_task(t) for t in range(N_TASKS)}}, indent=2))

        jobs = []
        if own.probe:
            jobs = [(conditions[0], 0, e) for e in range(own.probe)]
        else:
            # condition-interleaved order so a partial run stays paired
            for t in range(N_TASKS):
                for e in range(own.episodes_per_task):
                    for c in conditions:
                        jobs.append((c, t, e))
        t_start = time.time()
        n_done = 0
        for c, t, e in jobs:
            path = ep_dir / f"{c}_t{t}_e{e}.json"
            if path.exists():
                continue
            instr = {"correct": original[t], "empty": "", "wrong": original[wrong_task(t)],
                     "I-only": "", "I-wrong": "", "I-goal": "", "I-para": ""}[c]
            if is_vlia:
                z_task = intention_text(c, t, original, phrasings)
                current_z["z"] = table[z_task]
            env = task_envs[t]
            sub = env.envs[0]
            sub.init_state_id = e
            sub.task_description = instr
            assert list(env.call("task_description")) == [instr]
            frames = []

            def render(vec_env):
                frames.append(vec_env.envs[0].render())

            set_seed(SEED0 + e)
            t0 = time.time()
            with torch.no_grad():
                ret = rollout(env, policy, env_pre, env_post, pre, post, seeds=[SEED0 + e], render_callback=render)
            secs = time.time() - t0
            success = bool(ret["success"][0].any())
            done = ret["done"][0].numpy()
            n_steps = int(np.argmax(done) + 1) if done.any() else int(len(done))
            video = vid_dir / f"{c}_t{t}_e{e}.mp4"
            if frames:
                write_video(str(video), np.stack(frames), fps=sub.metadata.get("render_fps", 30))
            rec = {"condition": c, "task": t, "episode": e, "seed": SEED0 + e, "init_state_id": e,
                   "instruction": instr, "z_int_task": (z_task if is_vlia else None), "success": success, "n_steps": n_steps, "seconds": secs,
                   "video": str(video) if frames else None}
            path.write_text(json.dumps(rec))
            n_done += 1
            el = time.time() - t_start
            print(f"[{n_done}] {c} t{t} e{e} success={success} {secs:.1f}s "
                  f"(mean {el / n_done:.1f}s/ep)", flush=True)
        for t in task_envs.values():
            t.close()

    main()
    return 0


# ----------------------------------------------------------------------
# analysis (numpy only)
# ----------------------------------------------------------------------

def analyze(own) -> int:
    import csv

    out = Path(own.out)
    res = Path(own.results)
    res.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("*.json"))]
    by = {(r["condition"], r["task"], r["episode"]): r for r in recs}
    pairs = sorted({(r["task"], r["episode"]) for r in recs if all((c, r["task"], r["episode"]) in by for c in CONDITIONS)})
    succ = {c: np.array([float(by[(c, t, e)]["success"]) for t, e in pairs]) for c in CONDITIONS}
    rows = []
    for c in CONDITIONS:
        k, n = int(succ[c].sum()), len(pairs)
        lo, hi = wilson(k, n)
        rows.append({"condition": c, "n_episodes": n, "successes": k, "success_rate": k / n,
                     "wilson_lo": lo, "wilson_hi": hi})
    l1 = paired_bootstrap(succ["correct"] - succ["empty"])
    l2 = paired_bootstrap(succ["empty"] - succ["wrong"])
    verdict = {
        "L1": {"pass": l1[1] > 0, "delta": l1[0], "ci95": [l1[1], l1[2]],
               "rule": "success(correct) - success(empty) CI lower bound > 0"},
        "L2": {"pass": l2[1] > 0, "delta": l2[0], "ci95": [l2[1], l2[2]],
               "rule": "success(empty) - success(wrong) CI lower bound > 0"},
    }
    task_rows = []
    for t in range(N_TASKS):
        row = {"task": t}
        for c in CONDITIONS:
            v = [by[(c, t, e)]["success"] for tt, e in pairs if tt == t]
            row[f"{c}_successes"] = int(sum(v))
            row[f"{c}_n"] = len(v)
        task_rows.append(row)
    videos = {}
    for c in CONDITIONS:
        ok = [by[(c, t, e)]["video"] for t, e in pairs if by[(c, t, e)]["success"] and by[(c, t, e)]["video"]]
        bad = [by[(c, t, e)]["video"] for t, e in pairs if not by[(c, t, e)]["success"] and by[(c, t, e)]["video"]]
        videos[c] = {"success": ok[0] if ok else None, "failure": bad[0] if bad else None}
    for name, table in (("summary.csv", rows), ("per_task.csv", task_rows),
                        ("per_episode.csv", sorted(recs, key=lambda r: (r["task"], r["episode"], r["condition"])))):
        with (res / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    instr = json.loads((out / "instructions.json").read_text())
    lines = ["# LIBERO-10 goal-information ablation (prereg v2, Stage 4)", "",
             f"{len(pairs)} paired (task, episode) units per condition; same seed and init state across conditions.", "",
             "| condition | successes / n | success rate | Wilson 95% |", "|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['successes']} / {r['n_episodes']} | {r['success_rate']:.3f} | "
                     f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |")
    lines += ["", "| task | instruction | correct | empty | wrong (uses task) |", "|---:|---|---:|---:|---|"]
    for r in task_rows:
        t = r["task"]
        lines.append(f"| {t} | {instr['original'][str(t)]} | {r['correct_successes']}/{r['correct_n']} | "
                     f"{r['empty_successes']}/{r['empty_n']} | {r['wrong_successes']}/{r['wrong_n']} (task {wrong_task(t)}) |")
    lines += ["", "## Verdicts (paired bootstrap over (task, episode), B=10000, seed 0)", "",
              f"- L1 correct − empty: {l1[0]:+.3f} [{l1[1]:+.3f}, {l1[2]:+.3f}] → {'PASS' if verdict['L1']['pass'] else 'FAIL'}",
              f"- L2 empty − wrong: {l2[0]:+.3f} [{l2[1]:+.3f}, {l2[2]:+.3f}] → {'PASS' if verdict['L2']['pass'] else 'FAIL'}",
              "", "## Example videos (on d3)", "", "```", json.dumps(videos, indent=2), "```", ""]
    (res / "summary.md").write_text("\n".join(lines))
    meta = json.loads((out / "run_meta.json").read_text()) if (out / "run_meta.json").exists() else {}
    (res / "config.json").write_text(json.dumps({
        "episodes_dir": str(out / "episodes"), "n_pairs": len(pairs), "conditions": CONDITIONS,
        "seed_rule": "seed = 1000 + episode, init_state_id = episode", "wrong_map": "t -> (t+5) mod 10",
        "bootstrap": {"B": 10000, "seed": 0, "unit": "(task, episode) pair"}, "verdicts": verdict,
        "videos": videos, **meta}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def analyze_intention(own) -> int:
    """B1: I-only - empty(v1); B2: I-only - I-wrong; paired over (task, episode)."""
    import csv

    out = Path(own.out)
    res = Path(own.results)
    res.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("*.json"))]
    v1 = [json.loads(p.read_text()) for p in sorted(Path(own.v1_episodes).glob("*.json"))]
    by = {(r["condition"], r["task"], r["episode"]): r for r in recs + v1}
    need = ("I-only", "I-wrong", "empty", "correct")
    pairs = sorted({(r["task"], r["episode"]) for r in recs if all((c, r["task"], r["episode"]) in by for c in need)})
    for t, e in pairs:  # pairing must be on identical seeds / init states
        assert len({(by[(c, t, e)]["seed"], by[(c, t, e)]["init_state_id"]) for c in need}) == 1
    succ = {c: np.array([float(by[(c, t, e)]["success"]) for t, e in pairs]) for c in need}
    rows = []
    for c in need:
        k, n = int(succ[c].sum()), len(pairs)
        lo, hi = wilson(k, n)
        rows.append({"condition": c, "source": "this run" if c.startswith("I-") else "libero_goal_ablation_v1",
                     "n_episodes": n, "successes": k, "success_rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
    b1 = paired_bootstrap(succ["I-only"] - succ["empty"])
    b2 = paired_bootstrap(succ["I-only"] - succ["I-wrong"])
    dc = paired_bootstrap(succ["I-only"] - succ["correct"])
    verdict = {
        "B1": {"pass": b1[1] > 0, "delta": b1[0], "ci95": [b1[1], b1[2]], "rule": "success(I-only) - success(empty, v1) CI lower bound > 0"},
        "B2": {"pass": b2[1] > 0, "delta": b2[0], "ci95": [b2[1], b2[2]], "rule": "success(I-only) - success(I-wrong) CI lower bound > 0"},
        "descriptive_I-only_minus_correct_language": {"delta": dc[0], "ci95": [dc[1], dc[2]], "note": "no superiority claim"},
    }
    task_rows = []
    for t in range(N_TASKS):
        row = {"task": t}
        for c in need:
            v = [by[(c, tt, e)]["success"] for tt, e in pairs if tt == t]
            row[c] = f"{int(sum(v))}/{len(v)}"
        task_rows.append(row)
    for name, table in (("summary.csv", rows), ("per_task.csv", task_rows),
                        ("per_episode.csv", sorted(recs, key=lambda r: (r["task"], r["episode"], r["condition"])))):
        with (res / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    videos = {}
    for c in ("I-only", "I-wrong"):
        ok = [by[(c, t, e)]["video"] for t, e in pairs if by[(c, t, e)]["success"]]
        bad = [by[(c, t, e)]["video"] for t, e in pairs if not by[(c, t, e)]["success"]]
        videos[c] = {"success": ok[0] if ok else None, "failure": bad[0] if bad else None}
    lines = ["# LIBERO-10 intention-only (prereg v3, Stage 4)", "",
             f"{len(pairs)} paired (task, episode) units; seeds and init states identical to libero_goal_ablation_v1.", "",
             "| condition | source | successes / n | rate | Wilson 95% |", "|---|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['source']} | {r['successes']} / {r['n_episodes']} | {r['success_rate']:.3f} | "
                     f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |")
    lines += ["", "| task | I-only | I-wrong | empty (v1) | correct (v1) |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['task']} | {r['I-only']} | {r['I-wrong']} | {r['empty']} | {r['correct']} |" for r in task_rows]
    lines += ["", "## Verdicts (paired bootstrap over (task, episode), B=10000, seed 0)", "",
              f"- B1 I-only − empty: {b1[0]:+.3f} [{b1[1]:+.3f}, {b1[2]:+.3f}] → {'PASS' if verdict['B1']['pass'] else 'FAIL'}",
              f"- B2 I-only − I-wrong: {b2[0]:+.3f} [{b2[1]:+.3f}, {b2[2]:+.3f}] → {'PASS' if verdict['B2']['pass'] else 'FAIL'}",
              f"- descriptive I-only − correct language: {dc[0]:+.3f} [{dc[1]:+.3f}, {dc[2]:+.3f}]",
              "", "```", json.dumps(videos, indent=2), "```", ""]
    (res / "summary.md").write_text("\n".join(lines))
    (res / "config.json").write_text(json.dumps({"episodes_dir": str(out / "episodes"), "v1_episodes": own.v1_episodes,
                                                 "n_pairs": len(pairs), "verdicts": verdict, "videos": videos}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def analyze_stage11(own) -> int:
    """Stage 11: I-goal / I-para vs I-wrong (G1, G2) and descriptive deltas, e = 0..29; gate R2 on e = 0..9."""
    import csv

    res = Path(own.results)
    res.mkdir(parents=True, exist_ok=True)
    vlia = [json.loads(p.read_text()) for p in sorted((Path(own.out) / "episodes").glob("*.json"))]
    base = [json.loads(p.read_text()) for p in sorted(Path(own.v1_episodes).glob("*.json"))]
    by = {(r["condition"], r["task"], r["episode"]): r for r in vlia if r["condition"] in STAGE11_VLIA}
    by.update({(r["condition"], r["task"], r["episode"]): r for r in base if r["condition"] in STAGE11_BASE})
    conds = STAGE11_VLIA + STAGE11_BASE
    units = sorted({(t, e) for (_, t, e) in by if all((c, t, e) in by for c in conds)})
    for t, e in units:  # pairing must be on identical seeds / init states
        assert len({(by[(c, t, e)]["seed"], by[(c, t, e)]["init_state_id"]) for c in conds}) == 1, (t, e)
    phrasings = json.loads(Path(own.phrasings).read_text())
    original = {int(k): v for k, v in json.loads((Path(own.out) / "instructions.json").read_text())["original"].items()}
    for c in STAGE11_VLIA:  # the recorded intention text must be the pre-registered one
        for t, e in units:
            assert by[(c, t, e)]["z_int_task"] == intention_text(c, t, original, phrasings), (c, t, e)
    sets = {"e0-9": [u for u in units if u[1] < 10], "e0-29": units}
    succ = {k: {c: np.array([float(by[(c, t, e)]["success"]) for t, e in u]) for c in conds} for k, u in sets.items()}
    r2 = {c: int(succ["e0-9"][c].sum()) for c in STAGE4_E0_9}
    r2_pass = len(sets["e0-9"]) == 100 and r2 == STAGE4_E0_9
    print(f"GATE R2 e=0..9 counts {r2} (n={len(sets['e0-9'])}) expected {STAGE4_E0_9}: {'PASS' if r2_pass else 'FAIL'}")
    if not r2_pass:
        raise SystemExit("STOP: R2 failed")
    rows = []
    for k, u in sets.items():
        for c in conds:
            n, x = len(u), int(succ[k][c].sum())
            lo, hi = wilson(x, n)
            rows.append({"episodes": k, "condition": c, "successes": x, "n": n, "success_rate": x / n,
                         "wilson_lo": lo, "wilson_hi": hi})
    comps = [("G1", "I-goal", "I-wrong"), ("G2", "I-para", "I-wrong"), ("descriptive", "I-goal", "I-only"),
             ("descriptive", "I-para", "I-only"), ("descriptive", "I-only", "correct"), ("descriptive", "I-only", "empty")]
    paired, verdict = [], {}
    for k in sets:
        for rule, a, b in comps:
            d, lo, hi = paired_bootstrap(succ[k][a] - succ[k][b])
            paired.append({"episodes": k, "rule": rule, "comparison": f"{a} - {b}", "delta": d, "ci95_lo": lo,
                           "ci95_hi": hi, "n_pairs": len(sets[k])})
            if k == "e0-29" and rule != "descriptive":
                verdict[rule] = {"pass": lo > 0, "delta": d, "ci95": [lo, hi],
                                 "rule": f"success({a}) - success({b}) CI lower bound > 0"}
    task_rows = []
    for t in range(N_TASKS):
        row = {"task": t}
        for c in conds:
            v = [by[(c, tt, e)]["success"] for tt, e in units if tt == t]
            row[f"{c}_successes"], row[f"{c}_n"] = int(sum(v)), len(v)
        task_rows.append(row)
    emb = np.load(own.task_embeddings)
    table = dict(zip(map(str, emb["tasks"]), emb["emb"].astype(np.float64)))
    orig = np.stack([table[original[t]] for t in range(N_TASKS)])
    diag = []
    for key, c in (("goal", "I-goal"), ("para", "I-para")):
        for t in range(N_TASKS):
            sims = orig @ table[phrasings[key][str(t)]]
            diag.append({"phrasing": key, "task": t, "text": phrasings[key][str(t)], "cos_to_own_original": float(sims[t]),
                         "rank_of_own_task": int(1 + (sims > sims[t]).sum()),
                         "nearest_task": int(np.argmax(sims)), "successes": task_rows[t][f"{c}_successes"],
                         "I-only_successes": task_rows[t]["I-only_successes"], "n": task_rows[t][f"{c}_n"]})
    for name, tab in (("summary.csv", rows), ("paired.csv", paired), ("per_task.csv", task_rows),
                      ("text_diagnostic.csv", diag)):
        with (res / name).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(tab[0]))
            w.writeheader()
            w.writerows(tab)
    secs = {c: float(np.mean([by[(c, t, e)]["seconds"] for t, e in units])) for c in conds}
    lines = ["# LIBERO-10: intention phrased as a goal (Stage 11)", "",
             f"Paired (task, episode) units: {len(sets['e0-9'])} (e = 0..9), {len(units)} (e = 0..29).", "",
             "| episodes | condition | successes / n | rate | Wilson 95% |", "|---|---|---:|---:|---|"]
    lines += [f"| {r['episodes']} | {r['condition']} | {r['successes']} / {r['n']} | {r['success_rate']:.3f} | "
              f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}] |" for r in rows]
    lines += ["", "| episodes | rule | comparison | delta [95% CI] |", "|---|---|---|---|"]
    lines += [f"| {r['episodes']} | {r['rule']} | {r['comparison']} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] |"
              for r in paired]
    lines += ["", "| task | I-only | I-goal | I-para | I-wrong | correct | empty | goal cos / rank | para cos / rank |",
              "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for t, r in enumerate(task_rows):
        g, p = diag[t], diag[N_TASKS + t]
        lines.append(f"| {t} | " + " | ".join(f"{r[c + '_successes']}/{r[c + '_n']}" for c in conds) +
                     f" | {g['cos_to_own_original']:.3f} / {g['rank_of_own_task']} | {p['cos_to_own_original']:.3f} / {p['rank_of_own_task']} |")
    lines += ["", "## Verdicts (e = 0..29, paired bootstrap over (task, episode), B=10000, seed 0)", ""]
    lines += [f"- {k} {v['rule']}: {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] → {'PASS' if v['pass'] else 'FAIL'}"
              for k, v in verdict.items()]
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    (res / "config.json").write_text(json.dumps({
        "vlia_episodes": str(Path(own.out) / "episodes"), "baseline_episodes": own.v1_episodes,
        "vlia_checkpoint": own.vlia_checkpoint, "baseline_checkpoint": own.baseline_checkpoint,
        "task_embeddings": own.task_embeddings, "phrasings": own.phrasings, "n_pairs": {k: len(u) for k, u in sets.items()},
        "seconds_per_episode_mean": secs, "seed_rule": "seed = 1000 + episode, init_state_id = episode",
        "bootstrap": {"B": 10000, "seed": 0, "unit": "(task, episode) pair"}, "gate_R2": {"counts": r2, "pass": r2_pass},
        "verdicts": verdict, "git_commit": own.git_commit}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=10)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--task-embeddings", default="/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings.npz")
    ap.add_argument("--analyze-intention", action="store_true")
    ap.add_argument("--analyze-stage11", action="store_true")
    ap.add_argument("--phrasings", default=str(ROOT / "results/stage11_libero_goal/phrasings.json"))
    ap.add_argument("--vlia-checkpoint", default="/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/train/intention_only/seed_0/checkpoint_025000/pretrained_model")
    ap.add_argument("--baseline-checkpoint", default="/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000/pretrained_model")
    ap.add_argument("--git-commit", default="")
    ap.add_argument("--v1-episodes", default="/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/episodes")
    own, rest = ap.parse_known_args()
    if own.analyze:
        return analyze(own)
    if own.analyze_intention:
        return analyze_intention(own)
    if own.analyze_stage11:
        return analyze_stage11(own)
    return run(own, rest)


if __name__ == "__main__":
    raise SystemExit(main())
