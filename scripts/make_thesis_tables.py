#!/usr/bin/env python3
"""Generate thesis table snippets from permitted experiment summary CSVs."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX_RESULTS = {
    "prefix_length_ablation_v2": ROOT / "results/prefix_length_ablation_v2/summary.csv",
    "prefix_length_full_v1": ROOT / "results/prefix_length_full_v1/summary.csv",
}
NUMERIC = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/thesis_tables",
        help="Directory for the generated result-table snippets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace this script's known generated snippets in an existing output directory.",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        help="Also write the scene-CV table to this path (never overwrites an existing file).",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="MAIN_TEX",
        help="Report generated numeric table values absent from this LaTeX file.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric(value: str) -> str:
    return f"{float(value):.4f}"


def format_prefix_table(name: str, path: Path) -> tuple[str, list[str]]:
    header = (
        f"% Source: {path.relative_to(ROOT)}; validation summary only.\n"
        "\\begin{tabular}{rrrrrrr}\n"
        "Frames & Seeds & Examples/seed & MRR & Top-1 & Top-3 & Top-5 \\\\\n"
        "\\hline\n"
    )
    rows = read_csv(path)
    selected = [
        row
        for row in rows
        if row.get("setting") == "train_k_eval_k"
        and row.get("selection") == "best_val_epoch"
    ]
    selected.sort(key=lambda row: int(row["prefix_frames"]))
    tokens: list[str] = []
    lines: list[str] = []
    for row in selected:
        values = [
            row["prefix_frames"],
            row["n_seeds"],
            row["n_examples_per_seed"],
            f"{metric(row['MRR_mean'])} $\\pm$ {metric(row['MRR_std'])}",
            f"{metric(row['Top1_mean'])} $\\pm$ {metric(row['Top1_std'])}",
            f"{metric(row['Top3_mean'])} $\\pm$ {metric(row['Top3_std'])}",
            f"{metric(row['Top5_mean'])} $\\pm$ {metric(row['Top5_std'])}",
        ]
        tokens.extend(
            token
            for cell in values
            for token in NUMERIC.findall(cell.replace("\\pm", " "))
        )
        lines.append(" & ".join(values) + " \\\\\n")
    if not selected:
        lines.append("% No train_k_eval_k / best_val_epoch rows found.\n")
    return (
        f"% {name}: best validation-selected checkpoint; not an independent test.\n"
        + header
        + "".join(lines)
        + "\\end{tabular}\n",
        tokens,
    )


def format_scene_cv(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        return (
            "% scene_cv_v1 unavailable: results/scene_cv_v1/summary.csv not found.\n",
            [],
        )
    rows = read_csv(path)
    if not rows:
        return ("% scene_cv_v1 summary.csv is empty.\n", [])
    # The CV run has not been produced in this checkout. Avoid guessing its
    # schema or silently translating unrecognized columns into thesis values.
    required = {
        "target",
        "prefix_frames",
        "MRR_mean",
        "MRR_sd_seeds",
        "MRR_lift",
        "MRR_lift_ci_lo",
        "MRR_lift_ci_hi",
        "kway5_acc_mean",
        "kway5_acc_sd_seeds",
        "kway5_lift",
        "kway5_lift_ci_lo",
        "kway5_lift_ci_hi",
        "kway10_acc_mean",
        "kway10_acc_sd_seeds",
    }
    if not required.issubset(rows[0]):
        columns = ", ".join(rows[0].keys())
        return (
            "% scene_cv_v1 summary schema requires review; columns found: "
            f"{columns}\n",
            [],
        )
    selected = sorted(
        rows,
        key=lambda row: (row["target"], int(row["prefix_frames"])),
    )
    lines = [
        "% Generated from results/scene_cv_v1/summary.csv; out-of-fold, train+val only.\n",
        "\\begin{tabular}{lrrrrrr}\n",
        "Target & $k$ & MRR & MRR lift [95\\% CI] & 5-way acc. & 5-way lift [95\\% CI] & 10-way acc. \\\\\n",
        "\\hline\n",
    ]
    tokens: list[str] = []
    for row in selected:
        target = row["target"].replace("_", r"\_")
        mrr = f"{metric(row['MRR_mean'])} $\\pm$ {metric(row['MRR_sd_seeds'])}"
        mrr_lift = (
            f"{float(row['MRR_lift']):.2f} "
            f"[{float(row['MRR_lift_ci_lo']):.2f}, {float(row['MRR_lift_ci_hi']):.2f}]"
        )
        kway5 = (
            f"{metric(row['kway5_acc_mean'])} "
            f"$\\pm$ {metric(row['kway5_acc_sd_seeds'])}"
        )
        kway5_lift = (
            f"{float(row['kway5_lift']):.2f} "
            f"[{float(row['kway5_lift_ci_lo']):.2f}, "
            f"{float(row['kway5_lift_ci_hi']):.2f}]"
        )
        kway10 = (
            f"{metric(row['kway10_acc_mean'])} "
            f"$\\pm$ {metric(row['kway10_acc_sd_seeds'])}"
        )
        tokens.extend(
            token
            for cell in (
                row["prefix_frames"],
                mrr,
                mrr_lift,
                kway5,
                kway5_lift,
                kway10,
            )
            for token in NUMERIC.findall(cell.replace("\\pm", " "))
        )
        lines.append(
            " & ".join(
                (
                    target,
                    row["prefix_frames"],
                    mrr,
                    mrr_lift,
                    kway5,
                    kway5_lift,
                    kway10,
                )
            )
            + " \\\\\n"
        )
    lines.append("\\end{tabular}\n")
    return "".join(lines), tokens


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists() and not args.overwrite:
        print(
            f"Output directory exists: {output_dir}; pass --overwrite to replace "
            "only this script's known generated snippets.",
            file=sys.stderr,
        )
        return 2

    scene_summary = ROOT / "results/scene_cv_v1/summary.csv"
    generated: dict[str, tuple[str, list[str]]] = {}
    missing: list[str] = []
    for name, path in PREFIX_RESULTS.items():
        if path.is_file():
            generated[f"{name}.tex"] = format_prefix_table(name, path)
        else:
            missing.append(str(path))
            generated[f"{name}.tex"] = (
                f"% Source summary missing: {path}\n",
                [],
            )
    generated["scene_cv_v1.tex"] = format_scene_cv(scene_summary)
    generated["locked_test_full_v1.tex"] = (
        "% Omitted by the active instruction not to read or use test metrics.\n",
        [],
    )

    if args.scene_output is not None and args.scene_output.exists():
        print(f"Refusing to overwrite scene table: {args.scene_output}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (content, _) in generated.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    if args.scene_output is not None:
        scene_content, scene_tokens = generated["scene_cv_v1.tex"]
        args.scene_output.parent.mkdir(parents=True, exist_ok=True)
        args.scene_output.write_text(scene_content, encoding="utf-8")
        generated["scene_cv_v1.tex"] = (scene_content, scene_tokens)
        print(f"Wrote scene-CV table to {args.scene_output}")

    print(f"Wrote {len(generated)} snippets to {output_dir}")
    for path in missing:
        print(f"Missing input: {path}")
    print(
        "Excluded locked_test_full_v1 metrics: the active run instruction "
        "prohibits reading or using test metrics."
    )

    if args.check is not None:
        if not args.check.is_file():
            print(f"Cannot check missing LaTeX file: {args.check}", file=sys.stderr)
            return 2
        thesis = args.check.read_text(encoding="utf-8")
        absent = sorted(
            {
                token
                for _, tokens in generated.values()
                for token in tokens
                if token not in thesis
            },
            key=lambda value: (float(value), value),
        )
        if absent:
            print("Generated numeric strings absent from thesis:")
            for token in absent:
                print(f"  {token}")
        else:
            print("All generated numeric strings occur in the thesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
