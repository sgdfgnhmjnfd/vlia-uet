#!/usr/bin/env python3
"""Train the supervised intention classifier (module C) on ALL LIBERO-10 demos, for the demo app.

The thesis numbers come from cross-validation in run_stage15_supervised.py. This script fits one final
model on every demo, with the hyper-parameter chosen most often by the inner cross-validation of that run,
and saves it as a small .npz that the app loads without the training data.

Needs the features extracted by `run_stage15_supervised.py extract` (folder $VLIA_DATA_ROOT/vlia_outputs/stage15/feat).

  python scripts/train_app_classifier.py --out app/classifier_svm_lin.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage15_supervised as S15  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "app/classifier_svm_lin.npz"))
    ap.add_argument("--camera", default="agent", choices=["agent", "wrist"])
    args = ap.parse_args()

    cfg = json.loads((S15.RES / "config.json").read_text())
    hp = S15.most_frequent_hp(cfg["folds"][args.camera], "svm-lin")
    seqs, label, _info, n_demos, original = S15.load_libero(args.camera)
    tr = S15.build(seqs, sorted(label), label, S15.TRAIN_P)
    model = S15.make_model("svm-lin", hp, S15.N_TASKS)
    model.fit(tr["mean"], tr["y"])
    acc = float((model.scores(tr["mean"]).argmax(1) == tr["y"]).mean())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=model.W.numpy(), b=model.b.numpy(), C=hp["C"],
             tasks=np.array([original[t] for t in range(S15.N_TASKS)]))
    print(f"saved {out}: linear SVM C={hp['C']}, {len(label)} demos, training accuracy {acc:.3f}")
    print("demos per task:", n_demos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
