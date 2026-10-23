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
N_TASKS = 10
SEED0 = 1000


def wrong_task(t: int) -> int:
    return (t + 5) % N_TASKS


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
            jobs = [("correct", 0, e) for e in range(own.probe)]
        else:
            # condition-interleaved order so a partial run stays paired
            for t in range(N_TASKS):
                for e in range(own.episodes_per_task):
                    for c in CONDITIONS:
                        jobs.append((c, t, e))
        t_start = time.time()
        n_done = 0
        for c, t, e in jobs:
            path = ep_dir / f"{c}_t{t}_e{e}.json"
            if path.exists():
                continue
            instr = {"correct": original[t], "empty": "", "wrong": original[wrong_task(t)]}[c]
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
                   "instruction": instr, "success": success, "n_steps": n_steps, "seconds": secs,
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=10)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--results")
    own, rest = ap.parse_known_args()
    if own.analyze:
        return analyze(own)
    return run(own, rest)


if __name__ == "__main__":
    raise SystemExit(main())
