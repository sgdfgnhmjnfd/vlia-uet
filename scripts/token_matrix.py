#!/usr/bin/env python3
"""Token matrix (Stage 13 amendment 1): VLIA success for every scene task t, token task k and episode e.

Entry (t, k, e): k = t -> I-only, k = (t+5) mod 10 -> I-wrong, else I-tok{k}. Every rollout: empty instruction,
init_state_id = e, seed 1000 + e.

  build   --seed S [--eval-dir DIR]   -> /media/dhqg/d3/vlia_outputs/token_matrix/matrix_seed{S}.npz
  analyze                             -> results/token_matrix/
  lookup  --tokens JSON [--seed S]    -> per-unit successes of a (t, e) -> description map
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

N_TASKS, N_EP = 10, 30
OUT = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1")
EVAL_DIRS = {0: OUT / "eval", 1: OUT / "eval_seed1"}
CKPTS = {s: str(OUT / f"train/intention_only/seed_{s}/checkpoint_025000/pretrained_model") for s in (0, 1)}
WORK = Path("/media/dhqg/d3/vlia_outputs/token_matrix")
RES = ROOT / "results/token_matrix"
PAIRS = ((0, 1), (1, 0), (7, 5), (5, 7), (5, 9), (9, 5))
STAGE12_TOKENS = ROOT / "results/stage12_end_to_end/e2e_tokens.json"


def original_tasks() -> dict:
    d = json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"]
    return {int(k): v for k, v in d.items()}


def entry_file(t: int, k: int, e: int) -> str:
    if k == t:
        return f"I-only_t{t}_e{e}.json"
    if k == G.wrong_task(t):
        return f"I-wrong_t{t}_e{e}.json"
    return f"I-tok{k}_t{t}_e{e}.json"


def build(ep_dir: Path, original: dict):
    """(success (10, 10, 30) float with NaN where missing, seconds of I-tok entries, missing list)."""
    m = np.full((N_TASKS, N_TASKS, N_EP), np.nan)
    secs, missing = [], []
    for t in range(N_TASKS):
        for k in range(N_TASKS):
            for e in range(N_EP):
                p = Path(ep_dir) / entry_file(t, k, e)
                if not p.exists():
                    missing.append((t, k, e))
                    continue
                r = json.loads(p.read_text())
                assert r["task"] == t and r["episode"] == e and r["seed"] == 1000 + e and r["init_state_id"] == e, p
                assert r["z_int_task"] == original[k], (p, r["z_int_task"], original[k])
                assert r.get("instruction", "") == "", p
                if "token_task" in r:
                    assert r["token_task"] == k, p
                m[t, k, e] = float(r["success"])
                if k not in (t, G.wrong_task(t)):
                    secs.append(r["seconds"])
    return m, secs, missing


def matrix_path(seed: int) -> Path:
    return WORK / f"matrix_seed{seed}.npz"


def load_matrix(seed: int) -> np.ndarray:
    d = np.load(matrix_path(seed))
    m = d["success"]
    if np.isnan(m).any():
        raise SystemExit(f"STOP: matrix seed {seed} has {int(np.isnan(m).sum())} missing entries")
    return m


def lookup(m: np.ndarray, tokens: dict, original: dict) -> dict:
    """(t, e) -> description map ('t,e' keys) -> {'t,e': success of entry (t, k(description), e)}."""
    inv = {v: k for k, v in original.items()}
    out = {}
    for key, desc in tokens.items():
        t, e = map(int, key.split(","))
        out[key] = float(m[t, inv[desc], e])
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(dict.fromkeys(f for r in rows for f in r)), restval="")
        w.writeheader()
        w.writerows(rows)


def matrix_rows(m: np.ndarray) -> list[dict]:
    return [{"scene_task": t, **{f"token_{k}": int(m[t, k].sum()) for k in range(N_TASKS)}} for t in range(N_TASKS)]


def seed_analysis(m: np.ndarray, seed: int, ep_dir: Path, original: dict) -> tuple[list, dict]:
    eye = np.eye(N_TASKS, dtype=bool)
    lines, out = [], {}
    dk, dn = int(m[eye].sum()), int(m[eye].size)
    ok, on = int(m[~eye].sum()), int(m[~eye].size)
    out["diagonal"] = {"successes": dk, "n": dn, "rate": dk / dn, "wilson": G.wilson(dk, dn)}
    out["off_diagonal"] = {"successes": ok, "n": on, "rate": ok / on, "wilson": G.wilson(ok, on)}
    per_task = []
    for t in range(N_TASKS):
        wrong = np.delete(m[t], t, axis=0).mean(axis=0)  # mean over the 9 wrong tokens, per episode
        d, lo, hi = G.paired_bootstrap(m[t, t] - wrong)
        per_task.append({"task": t, "right": int(m[t, t].sum()), "mean_wrong": float(wrong.sum()), "delta": d, "ci95_lo": lo, "ci95_hi": hi,
                         "wrong_tokens_with_successes": {int(k): int(m[t, k].sum()) for k in range(N_TASKS) if k != t and m[t, k].sum() > 0}})
    out["per_task"] = per_task
    out["pairs"] = [{"scene_task": a, "token_task": b, "successes": int(m[a, b].sum()), "right_token": int(m[a, a].sum())} for a, b in PAIRS]
    # gate L1: Stage 12 E2E rollouts vs matrix entries for the same (t, k, e)
    plan = json.loads(STAGE12_TOKENS.read_text())["tokens"]
    inv = {v: k for k, v in original.items()}
    same, n = 0, 0
    for key, desc in plan.items():
        t, e = map(int, key.split(","))
        k = inv[desc]
        p = Path(ep_dir) / f"E2E_t{t}_e{e}.json"
        if k in (t, G.wrong_task(t)) or not p.exists():
            continue
        r = json.loads(p.read_text())
        assert r["z_int_task"] == desc
        same += int(float(r["success"]) == m[t, k, e])
        n += 1
    out["gate_L1"] = {"agreement": same / n if n else None, "n_units": n, "below_85pct": (same / n < 0.85) if n else None}
    lines += [f"## Seed {seed}", "", "Successes out of 30; rows = scene task t, columns = token task k (diagonal = I-only).", "",
              "| t \\ k | " + " | ".join(str(k) for k in range(N_TASKS)) + " |", "|---:|" + "---:|" * N_TASKS]
    for t in range(N_TASKS):
        lines.append(f"| {t} | " + " | ".join((f"**{int(m[t, k].sum())}**" if k == t else str(int(m[t, k].sum()))) for k in range(N_TASKS)) + " |")
    d, o = out["diagonal"], out["off_diagonal"]
    lines += ["", f"- Diagonal (right token): {d['successes']}/{d['n']} = {d['rate']:.3f} [{d['wilson'][0]:.3f}, {d['wilson'][1]:.3f}]",
              f"- Off-diagonal (wrong tokens): {o['successes']}/{o['n']} = {o['rate']:.3f} [{o['wilson'][0]:.3f}, {o['wilson'][1]:.3f}]", "",
              "| task | right token / 30 | mean over 9 wrong tokens / 30 | right − mean wrong [95% CI] | wrong tokens with successes |", "|---:|---:|---:|---|---|"]
    lines += [f"| {r['task']} | {r['right']} | {r['mean_wrong']:.2f} | {r['delta']:+.3f} [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | "
              f"{', '.join(f'k={k}: {v}' for k, v in r['wrong_tokens_with_successes'].items()) or '–'} |" for r in per_task]
    lines += ["", "| scene task | token task | successes / 30 | with the right token |", "|---:|---:|---:|---:|"]
    lines += [f"| {p['scene_task']} | {p['token_task']} | {p['successes']} | {p['right_token']} |" for p in out["pairs"]]
    g = out["gate_L1"]
    lines += ["", f"- Gate L1 (descriptive): Stage 12 `E2E` vs matrix entry, same (t, k, e) with k ≠ t, k ≠ (t+5) mod 10: "
              f"{g['agreement'] if g['agreement'] is None else f'{100 * g['agreement']:.1f}%'} of {g['n_units']} units agree"
              + (" — **below 85%**" if g["below_85pct"] else ""), ""]
    return lines, out


def cmd_build(args) -> int:
    ep_dir = Path(args.eval_dir) if args.eval_dir else EVAL_DIRS[args.seed] / "episodes"
    m, secs, missing = build(ep_dir, original_tasks())
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(matrix_path(args.seed), success=m, seconds=np.asarray(secs), eval_dir=str(ep_dir))
    print(f"seed {args.seed}: {int((~np.isnan(m)).sum())}/{m.size} entries, missing {len(missing)}"
          + (f" e.g. {missing[:5]}" if missing else ""))
    return 0 if not missing else 1


def cmd_analyze(_) -> int:
    original = original_tasks()
    RES.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in (0, 1) if matrix_path(s).exists()]
    lines = ["# Token matrix (Stage 13 amendment 1)", ""]
    cfg = {"git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "bootstrap": {"B": 10000, "seed": 0, "unit": "episode"}, "seeds": {}}
    mats = {}
    for s in seeds:
        d = np.load(matrix_path(s))
        m = d["success"]
        n_missing = int(np.isnan(m).sum())
        if n_missing:
            raise SystemExit(f"STOP: matrix seed {s} has {n_missing} missing entries")
        mats[s] = m
        write_csv(RES / f"matrix_seed{s}.csv", matrix_rows(m))
        sl, out = seed_analysis(m, s, Path(str(d["eval_dir"])), original)
        lines += sl
        cfg["seeds"][str(s)] = {"checkpoint": CKPTS[s], "eval_dir": str(d["eval_dir"]), "missing_entries": n_missing,
                                "seconds_per_episode_I-tok": float(np.mean(d["seconds"])) if len(d["seconds"]) else None, **out}
    if 0 in mats and 1 in mats:
        diff = mats[1].sum(axis=2) - mats[0].sum(axis=2)
        write_csv(RES / "matrix_seed1_minus_seed0.csv", [{"scene_task": t, **{f"token_{k}": int(diff[t, k]) for k in range(N_TASKS)}} for t in range(N_TASKS)])
        lines += ["## Seed 1 − seed 0 per cell (descriptive)", "", "| t \\ k | " + " | ".join(str(k) for k in range(N_TASKS)) + " |", "|---:|" + "---:|" * N_TASKS]
        lines += [f"| {t} | " + " | ".join(f"{int(diff[t, k]):+d}" for k in range(N_TASKS)) + " |" for t in range(N_TASKS)]
        lines.append("")
    (RES / "summary.md").write_text("\n".join(lines))
    (RES / "config.json").write_text(json.dumps(cfg, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else str(x)))
    print("\n".join(lines))
    return 0


def cmd_lookup(args) -> int:
    tokens = json.loads(Path(args.tokens).read_text())["tokens"]
    res = lookup(load_matrix(args.seed), tokens, original_tasks())
    print(json.dumps({"n_units": len(res), "successes": int(sum(res.values())), "per_unit": res}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "analyze", "lookup"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-dir", help="episode dir (default: the seed's eval dir)")
    ap.add_argument("--tokens")
    args = ap.parse_args()
    return {"build": cmd_build, "analyze": cmd_analyze, "lookup": cmd_lookup}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
