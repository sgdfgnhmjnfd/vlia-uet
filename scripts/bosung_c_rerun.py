#!/usr/bin/env python3
"""Bosung 20/11, step 2: run classifier C (Stage 15) again, from the raw demo videos, for the closed-loop row.

Uses the Stage 15 code unchanged (worktree /media/dhqg/d3/vlia-sup, run_stage15_supervised.py at 2e865a6): SigLIP
features are extracted again from the raw LIBERO demo videos and the nested CV (5 outer x 4 inner folds, same seeds) is
trained again, so every demo is predicted by a model that never saw it. Only the work folder differs (C_WORK below), so
nothing of Stage 15 is read back or overwritten. Then the p = 25 prediction per unit is compared with the old one.

  python scripts/bosung_c_rerun.py extract     # GPU if available (about 55 min on GPU)
  python scripts/bosung_c_rerun.py cv          # about 10 min
  python scripts/bosung_c_rerun.py predict     # CPU: c_predictions_p25.csv + e2e_sup_cl_tokens.json
  python scripts/bosung_c_rerun.py provisional # CPU: tokens from the OLD predictions (only for tests before predict)
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUP = Path("/media/dhqg/d3/vlia-sup")
C_WORK = Path("/media/dhqg/d3/vlia_outputs/bosung_2026-11-20/c_rerun")
RES = ROOT / "results/bosung_2026-11-20"
OLD_REF = "origin/claude/stage15-supervised-early:results/stage15_supervised_early/e2e_sup_tokens.json"
ROW = "E2E-sup-p25-CL"
P = 25


def s15():
    sys.path.insert(0, str(SUP / "scripts"))
    import run_stage15_supervised as S15  # noqa: E402

    S15.WORK = C_WORK  # the only change: a fresh work folder
    return S15


def old_plan() -> dict:
    return json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", OLD_REF]))


def stage12_units() -> list[str]:
    us = json.loads((ROOT / "results/train_100k/unit_sentences.json").read_text())
    return list(us["units_299"])


def original() -> dict:
    return {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}


def write_tokens(pred: dict, source: str) -> Path:
    """pred {"t,e": task}. Token text = description of the predicted task (row of that task in task_embeddings_v2)."""
    orig = original()
    units = stage12_units()
    assert set(pred) == set(units), "predictions must cover exactly the 299 Stage 12 units"
    tok = {k: orig[pred[k]] for k in units if pred[k] >= 0}
    out = {"tokens": {ROW: tok}, "pred": {ROW: {k: int(pred[k]) for k in units}},
           "true": {k: int(k.split(",")[0]) for k in units}, "reuse": {}, "source": source,
           "note": "closed-loop C: every unit is rolled out, no reuse of I-only"}
    RES.mkdir(parents=True, exist_ok=True)
    path = RES / "e2e_sup_cl_tokens.json"
    path.write_text(json.dumps(out, indent=1))
    return path


def main() -> int:
    cmd = sys.argv[1]
    C_WORK.mkdir(parents=True, exist_ok=True)
    if cmd == "extract":
        # a feature file cut by a power cut would be skipped as "done": move unreadable ones aside (never deleted)
        fdir = C_WORK / "feat/agent"
        for f in sorted(fdir.glob("*.npz")) if fdir.exists() else []:
            try:
                z = np.load(f)
                _ = z["emb"].shape, z["times"].shape
            except Exception:
                bad = C_WORK / "broken"
                bad.mkdir(exist_ok=True)
                f.rename(bad / f"{f.name}.{int(f.stat().st_mtime)}")
                print(f"moved unreadable {f.name} to {bad}", flush=True)
        return s15().cmd_extract(SimpleNamespace(camera="agent"))
    if cmd == "cv":
        return s15().cmd_cv(SimpleNamespace(camera="agent"))
    old = old_plan()
    old_pred = old["pred"]["E2E-sup-p25"]
    if cmd == "provisional":
        print(write_tokens({k: int(v) for k, v in old_pred.items()}, "PROVISIONAL: old Stage 15 predictions (tests only)"))
        return 0
    assert cmd == "predict", cmd
    S15 = s15()
    oof = S15.load_oof(C_WORK / "oof_agent.npz")
    cv = json.loads((C_WORK / "cv_agent.json").read_text())
    pp = S15.primary_pred(oof, cv["info"], P)
    feat_dir = C_WORK / "feat/agent"
    rows, pred = [], {}
    for k in stage12_units():
        t, e = map(int, k.split(","))
        key = S15.demo_key(t, e)
        s = oof[("primary", key, P)]
        f = np.load(feat_dir / f"{key}.npz")
        times = f["times"]
        n_used = int((times <= S15.libero_end(times, P) + 1e-9).sum())
        pred[k] = pp[(t, e)]
        rows.append({"t": t, "e": e, "demo_episode": int(f["episode"]), "video": f"LIBERO-10 dataset episode {int(f['episode'])} (agent camera)",
                     "frames_in_prefix": n_used, "pred": pred[k], "top_score": float(np.max(s)),
                     "old_pred": int(old_pred[k]), "match_old": int(pred[k] == int(old_pred[k])), "correct": int(pred[k] == t)})
    with (RES / "c_predictions_p25.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    path = write_tokens(pred, f"bosung C rerun ({C_WORK})")
    match = sum(r["match_old"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    # every token must equal the old token of the same unit when the prediction matches
    new_tok = json.loads(path.read_text())["tokens"][ROW]
    tok_same = sum(new_tok.get(k) == old["tokens"]["E2E-sup-p25"].get(k) for k in new_tok)
    summary = {"units": len(rows), "match_old": match, "correct": correct, "token_equal_old": tok_same,
               "mismatch_units": [f"{r['t']},{r['e']}" for r in rows if not r["match_old"]],
               "tokens_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (RES / "c_rerun_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    if match != len(rows):
        (C_WORK / "MISMATCH").write_text(json.dumps(summary))
        print("MISMATCH: new C predictions differ from Stage 15 on", len(rows) - match, "units")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
