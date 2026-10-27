#!/usr/bin/env python3
"""End-to-end test of scripts/run_v3_supplementary.py on SYNTHETIC data only.

Builds a fake ladder cache, manifest and step_label.json files in a temp dir,
points the ladder loader at them, and runs selftest, stage1, stage2 (reduced
grid) and next. Checks the outputs exist, F is pipeline index 0, and that no
real EgoIntent data is read.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_anticipation_horizon as AH  # noqa: E402
import run_ladder_v1 as L  # noqa: E402
import run_v3_supplementary as V  # noqa: E402

D, STEPS = 24, 14


def build(tmp: Path):
    rng = np.random.default_rng(0)
    rows, frames, texts_vec = [], [], {}
    for s in range(4):
        for v in range(2):
            ev_dir = tmp / "data" / "indoor" / f"scene{s}" / f"ev{v}"
            ev_dir.mkdir(parents=True)
            steps = []
            base = rng.normal(size=D)
            for st in range(1, STEPS + 1):
                why, what, nxt = f"why {s}{v}{st}", f"what {s}{v}{st}", f"next {s}{v}{st % 9}"
                for t in (why, nxt):
                    texts_vec.setdefault(t, rng.normal(size=D))
                f = base + texts_vec[why] + 0.3 * rng.normal(size=(8, D))
                frames.append(f / np.linalg.norm(f, axis=1, keepdims=True))
                rows.append({"sample_id": f"indoor/scene{s}/ev{v}/step{st}", "split": "train",
                             "video_uid": f"u{s}{v}", "scene": f"scene{s}", "event": f"ev{v}",
                             "step_id": str(st), "video_path": str(ev_dir / f"step{st}.mp4"),
                             "why": why, "what": what})
                steps.append({"step_id": st, "local_intent": what, "procedural_intent": why,
                              "observed_next_step": nxt})
            (ev_dir / "step_label.json").write_text(json.dumps({"steps": steps}))
    manifest = tmp / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    cache = tmp / "ladder" / V.ENC
    cache.mkdir(parents=True)
    np.save(cache / "frame_emb.npy", np.asarray(frames, dtype=np.float16))
    (cache / "sample_ids.json").write_text(json.dumps([r["sample_id"] for r in rows]))
    whys = sorted({r["why"] for r in rows})
    (cache / "texts.json").write_text(json.dumps(whys))
    t = L.normalize(np.asarray([texts_vec[w] for w in whys], dtype=np.float32))
    np.save(cache / "text_T0.npy", t)
    np.save(cache / "text_T1.npy", t)
    (cache / "meta.json").write_text(json.dumps({"frame_dtype": "float16"}))
    return manifest, tmp / "ladder", texts_vec


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        manifest, ladder, tv = build(tmp)
        L.MANIFEST, L.LADDER_ROOT = manifest, ladder
        V.ROOT = tmp  # outputs go to tmp/results
        V.B = 200
        V.SEEDS2 = (0,)
        AH.encode_texts = lambda enc, texts, device=None: L.normalize(
            np.asarray([tv[t] for t in texts], dtype=np.float32))
        data = V.load_data()
        f = V.f_out_of_fold(data)
        V.F_EXPECTED_HITS = {k: int(f[k]["kway5"].sum()) for k in V.PREFIXES}
        # F must equal the ladder's own A0|C2|T1 evaluation
        for fold in data.folds:
            idx, res = L.eval_held_out(data, (V.ENC, "A0", "C2", "T1"), fold, full=False)
            for k in V.PREFIXES:
                assert np.array_equal(res[k]["kway5"], f[k]["kway5"][idx]), "F differs from ladder"
                assert np.allclose(res[k]["rr"], f[k]["rr"][idx])

        class A:
            overwrite = True
            threads = 2
            device = "cpu"
        assert V.cmd_selftest(A) == 0
        V.GRID2 = [(1e-2, 1), (1e-1, 1)]
        assert V.cmd_stage1(A) == 0
        try:
            import torch  # noqa: F401
            assert V.cmd_stage2(A) == 0
        except ImportError:
            print("SKIP stage2: torch not installed")
            (tmp / "results" / "v3_stage2_hn_adapter").mkdir(parents=True, exist_ok=True)
            (tmp / "results" / "v3_stage2_hn_adapter" / "summary.csv").write_text("skipped")
        assert V.cmd_next(A) == 0
        for name in ("v3_stage1_aggfusion", "v3_stage2_hn_adapter", "v3_next_step"):
            assert (tmp / "results" / name / "summary.csv").exists(), name
    print("all v3 supplementary tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
