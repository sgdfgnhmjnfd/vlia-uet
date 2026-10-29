#!/usr/bin/env python3
"""Synthetic end-to-end test of scripts/run_time_horizon.py `run` (fake cache, no video, no model)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402
import run_time_horizon as TH  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    D, rate = 16, 4.0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows, tv = [], {}
        cache = tmp / "cache"
        cache.mkdir()
        for v in range(3):
            members, emb, row_of = [], [], []
            for st in range(1, 31):
                what, why = f"what {v} {st % 12}", f"why {v} {st % 7}"
                for t in (what, why):
                    tv.setdefault(t, rng.normal(size=D))
                i = len(rows)
                rows.append({"sample_id": f"v{v}/s{st}", "split": "train", "video_uid": f"u{v}",
                             "step_id": str(st), "video_path": "none", "what": what, "why": why})
                n = int(rng.integers(6, 14))
                emb.append(tv[what] + 0.5 * rng.normal(size=(n, D)))
                row_of += [i] * n
                members.append(i)
            emb = np.concatenate(emb)
            np.savez(cache / f"u{v}.npz", emb=emb.astype(np.float16), row=np.asarray(row_of),
                     t=np.arange(len(emb)) / rate, rate=rate, sample_ids=np.asarray([rows[i]["sample_id"] for i in members]))
        (tmp / "m.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        TH.MANIFEST, TH.ROOT, TH.B = tmp / "m.jsonl", tmp, 100
        AH.encode_texts = lambda enc, texts, device=None: TH.normalize(np.asarray([tv[t] for t in texts], float))

        class A:
            overwrite = True
            device = "cpu"
            contexts = [1, 2, 4]
            horizons = [0.5, 2, 8]
        A.cache = cache
        assert TH.cmd_selftest(A) == 0
        assert TH.cmd_run(A) == 0
        out = tmp / "results/time_horizon_v1"
        for f in ("summary.csv", "paired_horizon.csv", "paired_context.csv", "time_horizon.png"):
            assert (out / f).exists(), f
    print("time-horizon test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
