#!/usr/bin/env python3
"""prereg v3 Stage 4: so400m T1 text embeddings of LIBERO task instructions.

z_int(task) = L2-normalised so400m text embedding of
"a first-person photo of a person doing: {task}" (ladder tokenisation,
max_length 64), for every task string of the `lerobot/libero` dataset and the
10 LIBERO-10 eval instructions. Output: task_embeddings.npz (tasks, emb).
"""
from __future__ import annotations

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

    snap = sorted(glob.glob(str(Path.home() / ".cache/huggingface/lerobot/hub/datasets--lerobot--libero/snapshots/*/")))[-1]
    tasks = list(pd.read_parquet(Path(snap) / "meta/tasks.parquet").index)
    eval_tasks = list(json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].values())
    missing = [t for t in eval_tasks if t not in tasks]
    all_tasks = sorted(set(tasks) | set(eval_tasks))
    model, processor = X.load_encoder(ENCODER, torch.device("cpu"))
    emb, truncated = X.text_features(model, processor, ENCODER, [TEMPLATE.format(why=t) for t in all_tasks],
                                     torch.device("cpu"), 32)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, tasks=np.asarray(all_tasks), emb=emb.astype(np.float32))
    info = {"n_tasks": len(all_tasks), "dataset_tasks": len(tasks), "eval_tasks_missing_from_dataset": missing,
            "truncated": truncated, "encoder": ENCODER, "template": TEMPLATE, "dim": int(emb.shape[1])}
    OUT.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
