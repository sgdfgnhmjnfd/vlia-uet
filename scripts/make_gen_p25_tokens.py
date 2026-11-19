#!/usr/bin/env python3
"""Tokens for the Stage 14b G2 p = 25 sentences (row E2E-gen-p25), on CPU, with the same function as Stage 14
(run_stage14_generate.TextEncoder, T1 template, SigLIP so400m text tower).

Check first: the 10 true descriptions re-encoded on CPU must have cosine >= 0.9999 with the token tables used at 25k
(task_embeddings.npz and task_embeddings_v2.npz). Otherwise stop.
Outputs:
  results/train_100k/e2e_gen_p25_tokens.json   {"tokens": {"E2E-gen-p25": {"t,e": sentence}}, "reuse": {...}, "new": {...}}
  /media/dhqg/d3/vlia_outputs/train_100k/eval_token_table.npz   task_embeddings_v2 + the G2 sentences (the VLIA
      eval table: originals, para/goal phrasings, F descriptions, G2 sentences)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
V1 = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings.npz")
V2 = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/task_embeddings_v2.npz")
TABLE_OUT = Path("/media/dhqg/d3/vlia_outputs/train_100k/eval_token_table.npz")
TOK_OUT = ROOT / "results/train_100k/e2e_gen_p25_tokens.json"


def main() -> int:
    import run_stage14_generate as G14

    us = json.loads((ROOT / "results/train_100k/unit_sentences.json").read_text())
    rows = us["rows"]["E2E-gen-p25"]
    original = [us["original"][str(t)] for t in range(10)]
    enc = G14.TextEncoder(device="cpu")
    chk = enc(original, template=True)
    worst = 1.0
    for f in (V1, V2):
        tab = dict(zip(map(str, np.load(f)["tasks"]), np.load(f)["emb"]))
        for i, s in enumerate(original):
            a, b = chk[i].astype(np.float64), tab[s].astype(np.float64)
            worst = min(worst, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    print(f"CHECK: CPU re-encoding of the 10 descriptions vs 25k token tables, min cosine {worst:.6f}")
    if worst < 0.9999:
        raise SystemExit("STOP: cosine < 0.9999")
    sents = sorted({r["sentence"] for r in rows.values()})
    emb = enc(sents, template=True)
    v2 = np.load(V2)
    tasks = list(map(str, v2["tasks"]))
    add = [s for s in sents if s not in tasks]
    E = np.concatenate([v2["emb"], emb[[sents.index(s) for s in add]]]).astype(np.float32)
    T = np.asarray(tasks + add)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABLE_OUT.with_suffix(".tmp.npz"), tasks=T, emb=E)
    TABLE_OUT.with_suffix(".tmp.npz").replace(TABLE_OUT)
    reuse = [k for k, r in rows.items() if r["reuse"]]
    plan = {"tokens": {"E2E-gen-p25": {k: r["sentence"] for k, r in rows.items()}}, "reuse": {"E2E-gen-p25": reuse},
            "new": {"E2E-gen-p25": [k for k in rows if k not in set(reuse)]}, "cpu_check_min_cosine": worst,
            "table": str(TABLE_OUT), "n_new_sentences": len(add)}
    TOK_OUT.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    print(f"{len(rows)} units, {len(sents)} distinct G2 sentences ({len(add)} added to the table); table {TABLE_OUT} {E.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
