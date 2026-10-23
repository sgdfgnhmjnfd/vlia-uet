#!/usr/bin/env python3
"""One-shot locked-test evaluator; intentionally not executed during preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / "results/frozen_pipeline_v1/FROZEN.json"
APPROVAL_PATH = ROOT / "results/frozen_pipeline_v1/USER_APPROVED_TEST_EVAL"
OUTPUT_DIR = ROOT / "results/locked_test_final_v1"
TEST_MANIFEST = ROOT / "results/test_protocol_full_v1/candidate_test_manifest.csv"
REQUIRED_FIELDS = (
    "encoder_id",
    "aggregator",
    "centering",
    "template",
    "template_text",
    "beta",
    "frame_dtype",
    "c1_means_file",
    "c1_means_sha256",
    "baseline",
    "preregistration_sha256",
    "git_commit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_freeze(
    frozen_path: Path,
    approval_path: Path,
    output_dir: Path,
    confirm_freeze: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Frozen pipeline file does not exist: {frozen_path}")
    actual_hash = sha256_file(frozen_path)
    if actual_hash != confirm_freeze:
        raise ValueError(
            f"Freeze hash confirmation mismatch: expected {actual_hash}, "
            f"received {confirm_freeze}"
        )
    if not approval_path.is_file():
        raise PermissionError(
            f"Explicit user approval marker is required: {approval_path}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite one-shot evaluation output: {output_dir}"
        )
    with frozen_path.open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    missing = [field for field in REQUIRED_FIELDS if field not in frozen]
    if missing:
        raise ValueError(f"FROZEN.json is missing required fields: {missing}")
    if not isinstance(frozen["git_commit"], str) or not frozen["git_commit"]:
        raise ValueError("git_commit must be a non-empty commit identifier.")
    hashes = frozen["preregistration_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("preregistration_sha256 must be a non-empty path-to-hash mapping.")
    for relative_path, expected_hash in hashes.items():
        prereg_path = root / relative_path
        if not prereg_path.is_file():
            raise FileNotFoundError(f"Frozen preregistration file is missing: {prereg_path}")
        actual = sha256_file(prereg_path)
        if actual != expected_hash:
            raise ValueError(
                f"Frozen preregistration hash mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual}"
            )
    means_path = root / frozen["c1_means_file"]
    if sha256_file(means_path) != frozen["c1_means_sha256"]:
        raise ValueError(f"C1 means hash mismatch: {means_path}")
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-freeze", required=True)
    args = parser.parse_args()
    frozen = validate_freeze(
        FROZEN_PATH, APPROVAL_PATH, OUTPUT_DIR, args.confirm_freeze
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from frozen_eval_lib import evaluate_frozen_test

    # evaluate_frozen_test creates OUTPUT_DIR itself and refuses if it exists.
    result = evaluate_frozen_test(
        frozen=frozen,
        test_manifest=TEST_MANIFEST,
        output_dir=OUTPUT_DIR,
    )
    print(json.dumps(result["verdicts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
