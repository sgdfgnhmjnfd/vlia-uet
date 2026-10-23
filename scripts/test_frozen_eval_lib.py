#!/usr/bin/env python3
"""Unit tests for scripts/frozen_eval_lib.py on SYNTHETIC data only.

Writes a fake manifest, fake protocol and fake c1 means under
/media/dhqg/d3/tmp; frames and encoders are fakes. No EgoIntent test data is
read. Checks: the loader refuses train/val ids and non-candidate UIDs; C2 uses
only strictly earlier steps of the same video; gallery chance equals
`gallery_chance`; refuses an existing output dir; end-to-end outputs and
verdict fields exist.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frozen_eval_lib as F  # noqa: E402
import run_scene_cv as CV  # noqa: E402

TMP_ROOT = Path("/media/dhqg/d3/tmp")
D = 16
UIDS = ["u-a", "u-b", "u-c", "u-d"]
SCENES = {"u-a": "s1", "u-b": "s1", "u-c": "s2", "u-d": "s2"}


def synth_rows(n_steps=12):
    rows = []
    for v, uid in enumerate(UIDS):
        for s in range(1, n_steps + 1):
            why = f"goal {v}-{s % 7}"  # repeats inside a video -> multi-positive gallery
            rows.append({"sample_id": f"indoor/{SCENES[uid]}/ev{v}/step{s}", "video_uid": uid,
                         "scene": SCENES[uid], "event": f"ev{v}", "setting": "indoor", "why": why})
    return rows


def write_manifest(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def protocol(root):
    return {"train_video_uids": ["t-1"], "validation_video_uids": ["v-1"], "excluded_test_video_uids": ["x-1"],
            "candidate_test_video_uids": UIDS, "dataset_root": str(root)}


def fake_frames(path):
    seed = zlib.crc32(str(path).encode()) & 0xFFFFFFFF
    return np.random.RandomState(seed).randint(0, 255, size=(8, 6, 6, 3), dtype=np.uint8)


def fake_encoder_factory(name):
    rng = np.random.RandomState(zlib.crc32(name.encode()) & 0xFFFFFFFF)
    W = rng.normal(size=(6 * 6 * 3, D))

    def image_fn(frames):
        x = np.stack([f.reshape(-1).astype(np.float64) / 255.0 for f in frames]) @ W
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    def text_fn(texts):
        out = np.stack([np.random.RandomState(zlib.crc32(t.encode()) & 0xFFFFFFFF).normal(size=D) for t in texts])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32), 0

    return image_fn, text_fn


def frozen_cfg(means_path):
    return {"encoder_id": "fake/frozen", "aggregator": "A0", "centering": "C2",
            "template": "T1", "template_text": "a first-person photo of a person doing: {why}", "beta": 0.0,
            "frame_dtype": "float16", "c1_means_file": str(means_path),
            "baseline": {"encoder_id": "fake/base", "aggregator": "A0", "centering": "C0", "template": "T0",
                         "template_text": "{why}", "frame_dtype": "float32"}}


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def main() -> int:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
        tmp = Path(tmp)
        rows = synth_rows()
        man = tmp / "manifest.csv"
        write_manifest(man, rows)
        prot = protocol(tmp / "videos")
        n = len(rows)

        # 1. loader refuses train/val ids and non-candidate UIDs
        ok = F.load_test_rows(man, prot, set(), expected_n=n)
        assert len(ok) == n and all(r["video_path"].endswith(".mp4") for r in ok)
        expect_raises(ValueError, lambda: F.load_test_rows(man, prot, {rows[3]["sample_id"]}, expected_n=n))
        bad = [dict(r) for r in rows]
        bad[0]["video_uid"] = "t-1"
        man_bad = tmp / "bad.csv"
        write_manifest(man_bad, bad)
        expect_raises(ValueError, lambda: F.load_test_rows(man_bad, prot, set(), expected_n=n))
        expect_raises(ValueError, lambda: F.load_test_rows(man, prot, set(), expected_n=n + 1))

        # 2. C2 context: strictly earlier steps of the same video only
        vids = np.array([r["video_uid"] for r in ok])
        steps = np.array([r["step"] for r in ok])
        rng = np.random.default_rng(0)
        fr = rng.normal(size=(n, 8, D)).astype(np.float32)
        tx = rng.normal(size=(n, D)).astype(np.float32)
        mu_f, mu_t = np.zeros(D, np.float32), np.zeros(D, np.float32)
        out, _, _, use_c1 = F.frozen_process(fr, tx, vids, steps, mu_f + 0.1, mu_t + 0.1)
        for i in range(n):
            earlier = np.flatnonzero((vids == vids[i]) & (steps < steps[i]))
            assert use_c1[i] == (len(earlier) == 0)
            if len(earlier):
                ctx = fr[earlier].reshape(-1, D).mean(0)
                exp = (fr[i] - ctx) / np.linalg.norm(fr[i] - ctx, axis=-1, keepdims=True)
                assert np.allclose(out[i], exp, atol=1e-5)
        # perturbing later steps or other videos never changes a query's processed frames
        target = int(np.flatnonzero((vids == "u-a") & (steps == 5))[0])
        fr2 = fr.copy()
        fr2[(vids != "u-a") | (steps >= 5)] += 50.0
        fr2[target] = fr[target]
        out2, *_ = F.frozen_process(fr2, tx, vids, steps, mu_f + 0.1, mu_t + 0.1)
        assert np.allclose(out2[target], out[target])

        # 3. end-to-end on fake frames + fake encoders
        means = tmp / "c1.npz"
        np.savez(means, mu_frame=np.full(D, 0.01, np.float32), mu_text=np.full(D, 0.01, np.float32))
        outdir = tmp / "eval_out"
        res = F.evaluate_frozen_test(frozen_cfg(means), man, outdir, frame_loader=fake_frames,
                                     encoder_factory=fake_encoder_factory, protocol=prot, train_val_ids=set(),
                                     expected_n=n, bootstrap_B=200)
        for f in ("summary.csv", "summary.md", "per_example.csv", "config.json", "paired.csv", "control.csv"):
            assert (outdir / f).exists(), f
        assert set(res["verdicts"]) >= {"T1", "T2"}
        # chance values match gallery_chance over the whole test gallery
        vocab = {t: i for i, t in enumerate(sorted({r["why"] for r in rows}))}
        tid = np.array([vocab[r["why"]] for r in rows])
        ch = CV.gallery_chance(tid, np.zeros(n, int))
        for r in res["summary"]:
            assert abs(r["MRR_chance"] - ch["MRR"].mean()) < 1e-12
            assert abs(r["Top1_chance"] - ch["Top1"].mean()) < 1e-12
            assert abs(r["Top3_chance"] - ch["Top3"].mean()) < 1e-12
        assert json.loads((outdir / "config.json").read_text())["bootstrap"]["label"] == F.CLUSTER_LABEL
        # control donors come from another scene
        ctrl, donors, fb = F.control_donors([r["sample_id"] for r in rows], [r["scene"] for r in rows], vids)
        assert fb == 0 and all(rows[d]["scene"] != rows[i]["scene"] for i, d in zip(ctrl, donors))

        # 4. refuses an existing output dir
        expect_raises(FileExistsError, lambda: F.evaluate_frozen_test(
            frozen_cfg(means), man, outdir, frame_loader=fake_frames, encoder_factory=fake_encoder_factory,
            protocol=prot, train_val_ids=set(), expected_n=n, bootstrap_B=50))
    print("frozen_eval_lib tests: PASS (loader refusals, C2 strictly-earlier same-video context, "
          "gallery chance, end-to-end outputs, control donors, output-dir refusal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
