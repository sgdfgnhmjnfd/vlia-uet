#!/usr/bin/env python3
"""Turn one intention sentence into the token files that libero_goal_ablation.py reads (condition E2E-gen).

  python app/make_token.py OUT_DIR TASK "I want the bowl in the drawer"
writes OUT_DIR/token.npz (SigLIP text embedding with the thesis template) and OUT_DIR/tokens.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage14_generate as G  # noqa: E402


def main() -> int:
    out, task, sentence = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3].strip()
    out.mkdir(parents=True, exist_ok=True)
    emb = G.TextEncoder()([sentence], template=True)
    np.savez(out / "token.npz", tasks=np.asarray([sentence]), emb=emb.astype(np.float32))
    (out / "tokens.json").write_text(json.dumps({"tokens": {"E2E-gen": {f"{task},0": sentence}}}, indent=1))
    print(f"token for task scene {task}: {sentence!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
