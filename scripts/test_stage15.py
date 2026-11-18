#!/usr/bin/env python3
"""Synthetic checks for Stage 15 beyond `run_stage15_supervised.py selftest`: the E2E-sup condition of
libero_goal_ablation.py, the temporal CNN, nested CV with all four models, the LIBERO robot tables on fake
episodes, and the whole `human` pipeline on fake clips with a fake encoder (no SigLIP, no robot)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import libero_goal_ablation as G  # noqa: E402
import run_stage13_human as H  # noqa: E402
import run_stage15_supervised as S  # noqa: E402


def fake_episode(d: Path, c: str, t: int, e: int, z: str, success: bool):
    (d / f"{c}_t{t}_e{e}.json").write_text(json.dumps({"condition": c, "task": t, "episode": e, "seed": 1000 + e,
                                                       "init_state_id": e, "z_int_task": z, "success": success, "seconds": 9.0}))


def test_condition():
    original = {t: f"orig {t}" for t in range(10)}
    tok = {"E2E-sup-p25": {"3,7": "orig 4"}, "E2E-sup-human-p50": {"3,7": "orig 3"}}
    assert G.is_sup("E2E-sup-p25") and G.is_sup("E2E-sup-human-p50") and not G.is_sup("E2E") and not G.is_sup("E2E-sup")
    assert G.base_condition("E2E-sup-p50") == "E2E-sup" and G.base_condition("I-only") == "I-only"
    assert G.intention_text("E2E-sup-p25", 3, original, None, 7, tok) == "orig 4"
    assert G.intention_text("E2E-sup-human-p50", 3, original, None, 7, tok) == "orig 3"
    assert G.intention_text("I-wrong", 3, original, None) == "orig 8", "existing conditions unchanged"
    assert G.skip_sup("E2E-sup-p25", 3, 7, {"E2E-sup-p25": ["3,7"]}) and not G.skip_sup("E2E-sup-p50", 3, 7, {"E2E-sup-p25": ["3,7"]})
    src = (ROOT / "scripts/libero_goal_ablation.py").read_text()
    assert 'ep_dir / f"{c}_t{t}_e{e}.json"' in src  # -> E2E-sup-p25_t3_e7.json, never an E2E or I-only file
    assert "not own.probe and skip_sup" in src


def test_tcnn_and_cv():
    seqs, label = S._toy(n_per=8, k=3, d=12, sep=6.0, seed=1)
    tr = S.build(seqs, sorted(label), label, S.TRAIN_P)
    m = S.TCNN(3, epochs=30, val_seed=0).fit(tr["seq"], tr["y"], tr["vid"], label)
    s = m.scores(tr["seq"])
    assert s.shape == (len(tr["y"]), 3) and np.allclose(s.sum(1), 1) and len(m.nets) == 3
    assert S.strict_hits(s, tr["y"])[tr["p"] == 100].mean() > 0.9
    outer = dict(zip(sorted(label), S.strat_folds([label[v] for v in sorted(label)], 3, 0).tolist()))
    inner = lambda train: dict(zip(sorted(train), S.strat_folds([label[v] for v in sorted(train)], 2, 1).tolist()))  # noqa: E731
    grids = {"svm-lin": [{"C": 1}], "svm-rbf": [{"C": 1, "g": 1}], "logreg": [{"C": 1}], "tcnn": [{}]}
    oof, ch = S.run_cv(seqs, label, outer, inner, k=3, grids=grids, verbose=False)
    assert not S.leak_check(outer, ch) and all(c["primary"] in S.MODELS for c in ch)
    assert all(len(c["models"]["tcnn"]["best_epochs"]) == 3 for c in ch)
    rows = S.acc_rows(oof, label, ("primary",) + S.MODELS, S.EVAL_P, {})
    assert len(rows) == 5 * len(S.EVAL_P) and all(0 <= r["ci95_lo"] <= r["accuracy"] <= r["ci95_hi"] <= 1 for r in rows)
    acc = {(r["model"], r["p"]): r["accuracy"] for r in rows}
    assert acc[("primary", 100)] > acc[("primary", 0)], "toy task is easier with more of the video"


def test_robot_tables():
    original = {t: f"orig {t}" for t in range(10)}
    units = S.libero_units({t: (29 if t == 8 else 40) for t in range(10)})
    assert len(units) == 299 and (8, 29) not in units
    pred = {(p, t, e): (t if (e % 4) else (t + 1) % 10) for p in S.ROBOT_P for t, e in units}
    plan = S.plan_tokens(pred, units, original)
    with tempfile.TemporaryDirectory() as tmp:
        ev = Path(tmp)
        (ev / "episodes").mkdir()
        for t, e in units:
            fake_episode(ev / "episodes", "I-only", t, e, original[t], e < 20)
            fake_episode(ev / "episodes", "I-wrong", t, e, original[(t + 5) % 10], False)
            fake_episode(ev / "episodes", "E2E", t, e, original[t], e < 10)
        assert not S.check_reuse(plan, original, ev)
        try:
            S.robot_tables(plan, units, "E2E-sup", eval_dir=ev)
            raise AssertionError("missing new rollouts must stop")
        except SystemExit:
            pass
        for c, keys in plan["new"].items():
            for k in keys:
                t, e = map(int, k.split(","))
                fake_episode(ev / "episodes", c, t, e, plan["tokens"][c][k], False)
        succ, summary, paired, src = S.robot_tables(plan, units, "E2E-sup", eval_dir=ev)
        n_new = sum(1 for t, e in units if e % 4 == 0)
        assert src["E2E-sup-p25"] == {"reused": 299 - n_new, "new": n_new}
        # reused units carry the I-only outcome, new ones fail here: successes = I-only successes on reused units
        expect = sum(1 for t, e in units if e % 4 and e < 20)
        row = {r["condition"]: r for r in summary}
        assert row["E2E-sup-p25"]["successes"] == expect and row["I-wrong"]["successes"] == 0
        r4 = next(r for r in paired if r["rule"] == "R4")
        assert np.isclose(r4["delta"], expect / 299) and r4["ci95_lo"] > 0
        # a mismatching I-only file is caught
        fake_episode(ev / "episodes", "I-only", 0, 1, "orig 9", True)
        assert S.check_reuse(plan, original, ev)


def make_clips(videos: Path):
    import cv2

    rows = []
    rng = np.random.default_rng(0)
    for t in range(10):
        for r in H.REPS:
            v = f"t{t}_r{r}"
            fps, n, start, end = 10.0, 60, 1.5, 5.5
            w = cv2.VideoWriter(str(videos / f"{v}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
            colour = np.array([25 * t, 255 - 25 * t, 120 + 10 * (t % 3)], float)
            for i in range(n):
                s = i / fps
                frac = 0.0 if s < start else min(1.0, (s - start) / (end - start))
                img = (1 - frac) * 128 + frac * colour + rng.normal(0, 4, size=(48, 64, 3))
                w.write(np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1])
            w.release()
            rows.append({"video_id": v, "task": t, "scene": H.SCENE_OF[t], "rep": r, "person": "p1", "start_s": start, "end_s": end})
    with (videos / "labels.csv").open("w", newline="") as h:
        wr = csv.DictWriter(h, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def fake_embed(frames: np.ndarray) -> np.ndarray:
    f = frames.reshape(len(frames), -1, 3).astype(float).mean(axis=1) / 255 - 0.5
    return S12_normalize(np.c_[f, 0.05 * np.ones((len(f), 1))])


def S12_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_human():
    original = {int(k): v for k, v in json.loads((ROOT / "results/libero_goal_ablation_v1/instructions.json").read_text())["original"].items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        videos, work, res, ev = tmp / "clips", tmp / "work", tmp / "res", tmp / "eval"
        videos.mkdir()
        (ev / "episodes").mkdir(parents=True)
        make_clips(videos)
        for t, e in H.units():
            fake_episode(ev / "episodes", "I-only", t, e, original[t], True)
            fake_episode(ev / "episodes", "I-wrong", t, e, original[(t + 5) % 10], False)
        args = argparse.Namespace(videos=str(videos), work=str(work), results=str(res), labels=None, eval_dir=str(ev),
                                  rotate=0, allow_partial=False, cross_domain=0, step="extract")
        assert S.cmd_human(args, embed=fake_embed) == 0
        z = np.load(work / "feat/t3_r1.npz")
        assert z["times"][0] == 0 and np.isclose(z["times"][-1], 5.9) and np.allclose(z["times"][:3], [0, 0.2, 0.5])
        seqs = S.human_seqs(H.read_labels(videos / "labels.csv"), work / "feat")
        idle_end = np.sum(z["times"] <= 1.5 + 1e-9) - 1
        assert np.allclose(seqs["t3_r1"][0][-1], z["emb"][idle_end]), "p = 0 ends at the last idle frame"
        assert np.allclose(seqs["t3_r1"][100][-1], z["emb"][np.sum(z["times"] <= 5.5 + 1e-9) - 1])
        args.step = "cv"
        assert S.cmd_human(args) == 0
        cvh = json.loads((work / "cv_human.json").read_text())
        assert [c["fold"] for c in cvh["choices"]] == [0, 1, 2]
        for c in cvh["choices"]:
            assert all(v.endswith(f"_r{c['fold'] + 1}") for v in c["test"]) and len(c["test"]) == 10
        args.step = "plan"
        assert S.cmd_human(args) == 0
        plan = json.loads((res / "e2e_sup_human_tokens.json").read_text())
        assert plan["clip"]["4,5"] == "t4_r3" and len(plan["pred"]["E2E-sup-human-p25"]) == 300
        args.step = "analyze"
        assert S.cmd_human(args) == 0  # robot part not ready when some units need rollouts
        for c, keys in plan["new"].items():
            for k in keys:
                t, e = map(int, k.split(","))
                fake_episode(ev / "episodes", c, t, e, plan["tokens"][c][k], False)
        assert S.cmd_human(args) == 0
        cfg = json.loads((res / "config.json").read_text())
        v = cfg["verdicts"]
        assert v["R1"]["accuracy"] > 0.9 and v["R3"]["n"] == 27 and np.isclose(v["R3"]["chance"], 12 / 27)
        assert "R4" in v and cfg["robot"] is not None
        acc = {(r["model"], int(r["p"])): float(r["accuracy"]) for r in csv.DictReader((res / "accuracy.csv").open())}
        assert acc[("primary", 100)] > 0.9 and acc[("primary", 0)] < 0.5, "the idle part carries no task colour"
        assert (res / "curve.png").exists() and (res / "confusion_p25.csv").exists()


def main() -> int:
    test_condition()
    test_tcnn_and_cv()
    test_robot_tables()
    test_human()
    print("test_stage15 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
