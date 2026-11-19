#!/usr/bin/env python3
"""Final 100k evaluation: one sentence file for every row and every unit (CPU, no model).

Writes results/train_100k/unit_sentences.json:
  {"original": {t: description}, "units_300": [...], "units_299": [...],
   "rows": {row: {"t,e": {"true": ..., "sentence": ..., "instruction": ..., "token_text": ..., "source": ..., "key": ...,
                          "reuse": null | "I-only" | "correct"}}},
   "pairs": {...}, "counts": {...}}

- instruction = text in the policy's language slot (baseline rows); token_text = text whose T1 embedding is the VLIA
  intention token (VLIA rows). Each L-* row uses byte-for-byte the sentence of its token partner.
- reuse: a unit whose sentence equals the true description is not rolled out; it reuses I-only (VLIA E2E-sup-p25,
  E2E-gen-p25) or correct (baseline L-*), as in the plan. E2E (F) is run on all its 299 units, as at 25k.
Sources (read only): results/libero_goal_ablation_v1/instructions.json, results/stage11_libero_goal/phrasings.json,
results/stage12_end_to_end/e2e_tokens.json, results/stage15_supervised_early/e2e_sup_tokens.json (branch
origin/claude/stage15-supervised-early), Stage 14b G2 sentences (git show a469873:...sentences_G2.csv, p = 25).
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/train_100k/unit_sentences.json"
SRC = {
    "original": "results/libero_goal_ablation_v1/instructions.json",
    "phrasings": "results/stage11_libero_goal/phrasings.json",
    "e2e": "results/stage12_end_to_end/e2e_tokens.json",
    "sup": "origin/claude/stage15-supervised-early:results/stage15_supervised_early/e2e_sup_tokens.json",
    "g2": "a469873:results/stage14b_generate_variants/sentences_G2.csv",
}
VLIA_ROWS = ("I-only", "I-wrong", "I-para", "I-goal", "E2E", "E2E-sup-p25", "E2E-gen-p25")
BASE_ROWS = ("correct", "empty", "L-para", "L-goal", "L-E2E", "L-sup-p25", "L-gen-p25")
PAIRS = {"L-para": "I-para", "L-goal": "I-goal", "L-E2E": "E2E", "L-sup-p25": "E2E-sup-p25", "L-gen-p25": "E2E-gen-p25"}
ORDER = ("I-only", "I-wrong", "correct", "empty", "L-para", "I-para", "L-goal", "I-goal", "E2E", "L-E2E",
         "E2E-sup-p25", "L-sup-p25", "E2E-gen-p25", "L-gen-p25")
N_TASKS = 10


def git_show(spec: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", spec], capture_output=True, text=True, check=True).stdout


def wrong_task(t: int) -> int:
    return (t + 5) % N_TASKS


def build() -> dict:
    original = {int(k): v for k, v in json.loads((ROOT / SRC["original"]).read_text())["original"].items()}
    ph = json.loads((ROOT / SRC["phrasings"]).read_text())
    e2e = json.loads((ROOT / SRC["e2e"]).read_text())["tokens"]
    sup = json.loads(git_show(SRC["sup"]))["tokens"]["E2E-sup-p25"]
    g2 = {f"{r['task']},{r['demo']}": r["sentence"] for r in csv.DictReader(io.StringIO(git_show(SRC["g2"]))) if r["p"] == "25"}
    units_300 = [f"{t},{e}" for t in range(N_TASKS) for e in range(30)]
    units_299 = [k for k in units_300 if k in e2e]
    assert len(units_299) == 299 and set(sup) == set(units_299) and set(g2) == set(units_299), (len(sup), len(g2))

    def entry(t, sentence, source, key, slot, reuse_row=None):
        true = original[t]
        return {"true": true, "sentence": sentence, "source": source, "key": key,
                "instruction": sentence if slot == "instruction" else "",
                "token_text": sentence if slot == "token" else None,
                "reuse": reuse_row if (reuse_row and sentence == true) else None}

    rows = {r: {} for r in ORDER}
    for k in units_300:
        t, e = map(int, k.split(","))
        rows["I-only"][k] = entry(t, original[t], SRC["original"], f"original[{t}]", "token")
        rows["I-wrong"][k] = entry(t, original[wrong_task(t)], SRC["original"], f"original[{wrong_task(t)}]", "token")
        rows["I-para"][k] = entry(t, ph["para"][str(t)], SRC["phrasings"], f"para[{t}]", "token")
        rows["I-goal"][k] = entry(t, ph["goal"][str(t)], SRC["phrasings"], f"goal[{t}]", "token")
        rows["correct"][k] = entry(t, original[t], SRC["original"], f"original[{t}]", "instruction")
        rows["empty"][k] = entry(t, "", "-", "-", "instruction")
        rows["L-para"][k] = entry(t, ph["para"][str(t)], SRC["phrasings"], f"para[{t}]", "instruction", "correct")
        rows["L-goal"][k] = entry(t, ph["goal"][str(t)], SRC["phrasings"], f"goal[{t}]", "instruction", "correct")
    for k in units_299:
        t, e = map(int, k.split(","))
        rows["E2E"][k] = entry(t, e2e[k], SRC["e2e"], f"tokens[{k}]", "token")  # all 299 run, as at 25k
        rows["E2E-sup-p25"][k] = entry(t, sup[k], SRC["sup"], f"tokens[E2E-sup-p25][{k}]", "token", "I-only")
        rows["E2E-gen-p25"][k] = entry(t, g2[k], SRC["g2"], f"p=25 task={t} demo={e}", "token", "I-only")
        rows["L-E2E"][k] = entry(t, e2e[k], SRC["e2e"], f"tokens[{k}]", "instruction", "correct")
        rows["L-sup-p25"][k] = entry(t, sup[k], SRC["sup"], f"tokens[E2E-sup-p25][{k}]", "instruction", "correct")
        rows["L-gen-p25"][k] = entry(t, g2[k], SRC["g2"], f"p=25 task={t} demo={e}", "instruction", "correct")
    counts = {r: {"units": len(rows[r]), "reuse": sum(1 for x in rows[r].values() if x["reuse"]),
                  "same_as_true": sum(1 for x in rows[r].values() if x["sentence"] == x["true"])} for r in ORDER}
    return {"original": original, "units_300": units_300, "units_299": units_299, "rows": rows,
            "pairs": PAIRS, "order": list(ORDER), "vlia_rows": list(VLIA_ROWS), "base_rows": list(BASE_ROWS),
            "sources": SRC, "counts": counts}


def main() -> int:
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(OUT)
    for r in d["order"]:
        c = d["counts"][r]
        print(f"{r:12s} units {c['units']:3d}  sentence == true description: {c['same_as_true']:3d}  reuse: {c['reuse']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
