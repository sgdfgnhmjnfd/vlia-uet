#!/usr/bin/env python3
"""prereg v3 Stage 4: so400m T1 text embeddings of LIBERO task instructions.

z_int(task) = L2-normalised so400m text embedding of
"a first-person photo of a person doing: {task}" (ladder tokenisation,
max_length 64), for every task string of the `lerobot/libero` dataset and the
10 LIBERO-10 eval instructions. Output: task_embeddings.npz (tasks, emb).

Stage 11: --extra-json PHRASINGS --out V2.npz also embeds every `goal` / `para` string with the same
function and template, and checks gate E1 (original strings equal to the v1 table, max abs diff < 1e-6).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_ladder_embeddings as X  # noqa: E402

OUT = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings.npz")
ENCODER = "google/siglip-so400m-patch14-384"
TEMPLATE = X.TEMPLATES["T1"]


def main() -> int:
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra-json")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    if args.extra_json and out.resolve() == OUT.resolve():
        raise SystemExit("--extra-json must not overwrite the v1 table")

    snap = sorted(glob.glob(str(Path.home() / ".cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/*/")))[-1]
    tasks = list(pd.read_parquet(Path(snap) / "meta/tasks.parquet").index)
    eval_tasks = list(json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].values())
    missing = [t for t in eval_tasks if t not in tasks]
    extra = []
    if args.extra_json:
        ph = json.loads(Path(args.extra_json).read_text())
        extra = [ph[key][str(t)] for key in ("goal", "para") for t in range(len(eval_tasks))]
    all_tasks = sorted(set(tasks) | set(eval_tasks) | set(extra))
    model, processor = X.load_encoder(ENCODER, torch.device("cpu"))
    emb, truncated = X.text_features(model, processor, ENCODER, [TEMPLATE.format(why=t) for t in all_tasks],
                                     torch.device("cpu"), 32)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    info = {"n_tasks": len(all_tasks), "dataset_tasks": len(tasks), "eval_tasks_missing_from_dataset": missing,
            "truncated": truncated, "encoder": ENCODER, "template": TEMPLATE, "dim": int(emb.shape[1])}
    if args.extra_json:
        v1 = np.load(OUT)
        ref = dict(zip(map(str, v1["tasks"]), v1["emb"]))
        new = dict(zip(all_tasks, emb.astype(np.float32)))
        assert set(ref) <= set(new), "v1 strings missing from v2"
        diff = max(float(np.abs(new[t] - ref[t]).max()) for t in ref)
        info.update({"extra_json": args.extra_json, "n_extra": len(set(extra) - set(ref)),
                     "E1_max_abs_diff_vs_v1": diff, "E1_pass": diff < 1e-6})
        print(f"GATE E1 max abs diff vs v1 = {diff:.3g}: {'PASS' if diff < 1e-6 else 'FAIL'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tasks=np.asarray(all_tasks), emb=emb.astype(np.float32))
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    if args.extra_json and not info["E1_pass"]:
        raise SystemExit("STOP: E1 failed")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
