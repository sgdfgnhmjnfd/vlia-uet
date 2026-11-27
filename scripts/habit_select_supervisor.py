#!/usr/bin/env python3
"""Stage 22 Step 0: find HABIT Supervisor pointing tasks from the metadata only (no videos).

  python scripts/habit_select_supervisor.py   # reads /media/dhqg/d3/datasets/habit/full/meta, writes select_supervisor.csv

A task is a pointing task if one of its robot low-level instructions matches point/indicat (subtasks.jsonl). Per task
(sid): episodes, task_status, unit_name, recording blocks taken from input_key (date / id / block), and whether the
episode order of k (when k is in the instruction) is blocked in time.
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

META = Path("/media/dhqg/d3/datasets/habit/full/meta")
OUT = Path(__file__).resolve().parents[1] / "results/stage22_habit_supervisor"
R = re.compile(r"point|indicat", re.I)


def main() -> int:
    sub = {s["task"]: s["task_index"] for s in map(json.loads, open(META / "subtasks.jsonl"))}
    eps = [json.loads(line) for line in open(META / "episodes.jsonl")]
    by = collections.defaultdict(list)
    for e in eps:
        if any(R.search(t) for t in e["low_level_tasks"]):
            by[e["sid"]].append(e)
    rows = []
    for sid, es in sorted(by.items(), key=lambda x: -len(x[1])):
        es.sort(key=lambda e: e["input_key"][1])
        blocks = collections.Counter("/".join(e["input_key"][1].split("/")[-5:-2]) for e in es)
        point_steps = sorted({t for e in es for t in e["low_level_tasks"] if R.search(t)})
        rows.append({"sid": sid, "task": es[0]["tasks"][0], "episodes": len(es),
                     "success": sum(e["task_status"] == "success" for e in es),
                     "units": ";".join(sorted({e["unit_name"] for e in es})),
                     "dates": len({b.split("/")[0] for b in blocks}), "blocks": len(blocks),
                     "block_sizes": ";".join(f"{b}:{n}" for b, n in sorted(blocks.items())),
                     "robot_point_subtasks": ";".join(str(sub[t]) for t in point_steps),
                     "pointings_per_episode": max(sum(bool(R.search(t)) for t in e["low_level_tasks"]) for e in es)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "select_supervisor.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
