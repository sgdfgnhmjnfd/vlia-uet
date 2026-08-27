"""Công cụ debug LR scheduler khi resume checkpoint (xem scheduler_state.json,
training_step.json, optimizer_param_groups.json).

Ví dụ:
    python scripts/tools/inspect_scheduler.py \\
        --checkpoint-dir outputs/smolvla_libero/checkpoints/025000
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    args = p.parse_args()
    state_dir = Path(args.checkpoint_dir) / "training_state"
    for name in ["scheduler_state.json", "training_step.json", "optimizer_param_groups.json"]:
        f = state_dir / name
        if f.exists():
            print(f"--- {name} ---")
            print(json.dumps(json.loads(f.read_text()), indent=2, ensure_ascii=False))
        else:
            print(f"--- {name} : KHÔNG TỒN TẠI ---")


if __name__ == "__main__":
    main()
