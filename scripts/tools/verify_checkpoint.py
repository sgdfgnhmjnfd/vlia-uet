"""Kiểm tra checkpoint tồn tại và hợp lệ trước khi eval/resume — tránh lỗi
FileNotFoundError / HFValidationError do path sai hoặc ổ đĩa chưa mount.

Ví dụ:
    python scripts/tools/verify_checkpoint.py --path outputs/smolvla_libero/checkpoints/050000
"""

import argparse
import sys
from pathlib import Path

REQUIRED = ["pretrained_model/config.json", "pretrained_model/model.safetensors"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    args = p.parse_args()
    base = Path(args.path)
    if not base.exists():
        print(f"❌ Không tồn tại: {base}")
        sys.exit(1)
    ok = True
    for rel in REQUIRED:
        f = base / rel
        status = "✅" if f.exists() else "❌"
        if not f.exists():
            ok = False
        print(f"{status} {f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
