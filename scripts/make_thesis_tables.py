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
E1_RESULTS = ROOT / "results/e1_zero_shot_v1"
LOCKED_TEST_SUMMARY = ROOT / "results/locked_test_full_v1/summary.csv"
DAILY_TABLES = {
    "e1b_smolvlm2_v1.tex": (
        "E1b SmolVLM2",
        ROOT / "results/e1b_smolvlm2_v1/summary.csv",
        ("prefix_frames", "n_queries", "accuracy", "chance", "lift", "acc_minus_chance"),
    ),
    "selective_e1_v1.tex": (
        "Selective prediction",
        ROOT / "results/selective_e1_v1/summary.csv",
        ("k", "score", "coverage", "selective_accuracy", "risk", "n_accepted"),
    ),
    "ladder_v1.tex": (
        "Nested pipeline ladder",
        ROOT / "results/ladder_v1/summary.csv",
        ("prefix_frames", "baseline_mrr", "nested_mrr", "baseline_kway5", "nested_kway5"),
    ),
    "ead_v1.tex": (
        "Evidence-accumulation early decision",
        ROOT / "results/ead_v1/summary.csv",
        ("budget", "realized_mean_frames", "early_accuracy", "fixed_interpolated_accuracy", "delta"),
    ),
    "knn_cache_v1.tex": (
        "Cache-kNN correction",
        ROOT / "results/knn_cache_v1/summary.csv",
        ("prefix_frames", "baseline_mrr", "knn_mrr", "baseline_kway5", "knn_kway5"),
    ),
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
    parser.add_argument(
        "--daily-only",
        action="store_true",
        help="Generate only the E1b, selective, ladder, early-decision, and kNN tables.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def format_daily_table(
    title: str, path: Path, columns: tuple[str, ...]
) -> tuple[str, list[str]]:
    if not path.is_file():
        try:
            source_label = path.relative_to(ROOT)
        except ValueError:
            source_label = path
        return f"% {title} unavailable: {source_label} not found.\n", []
    rows = read_csv(path)
    if not rows:
        return f"% {title} unavailable: summary is empty.\n", []
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        return (
            f"% {title} schema requires review; missing columns: "
            + ", ".join(missing)
            + "\n",
            [],
        )
    try:
        source_label = path.relative_to(ROOT)
    except ValueError:
        source_label = path
    lines = [
        f"% Generated from {source_label}; validation/train-val evidence only.\n",
        f"\\begin{{tabular}}{{{'l' + 'r' * (len(columns) - 1)}}}\n",
        " & ".join(tex_escape(column.replace("_", " ")) for column in columns)
        + " \\\\\n",
        "\\hline\n",
    ]
    tokens: list[str] = []
    for row in rows:
        values = [row[column] for column in columns]
        escaped = [tex_escape(value) for value in values]
        lines.append(" & ".join(escaped) + " \\\\\n")
        tokens.extend(
            token for value in escaped for token in NUMERIC.findall(value)
        )
    lines.append("\\end{tabular}\n")
    return "".join(lines), tokens


def metric(value: str) -> str:
    return f"{float(value):.4f}"


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


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
    # Require the exact scene-CV summary schema rather than guessing mappings.
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
        target = tex_escape(row["target"])
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


def format_e1_tables(summary_path: Path, paired_path: Path) -> tuple[str, list[str]]:
    if not summary_path.is_file() or not paired_path.is_file():
        missing = [
            str(path) for path in (summary_path, paired_path) if not path.is_file()
        ]
        return ("% E1 inputs missing: " + "; ".join(missing) + "\n", [])
    summary_rows = read_csv(summary_path)
    paired_rows = read_csv(paired_path)
    required = {
        "branch",
        "prefix_frames",
        "MRR",
        "MRR_chance",
        "MRR_lift_boot",
        "MRR_lift_ci_lo",
        "MRR_lift_ci_hi",
        "Top1",
        "Top3",
        "5way_acc",
        "5way_lift_boot",
        "5way_lift_ci_lo",
        "5way_lift_ci_hi",
        "10way_acc",
    }
    if not summary_rows or not required.issubset(summary_rows[0]):
        columns = ", ".join(summary_rows[0].keys()) if summary_rows else "(empty)"
        return f"% E1 summary schema needs review; columns: {columns}\n", []

    tokens: list[str] = []
    lines = [
        "% E1 zero-shot SigLIP alignment; train/validation only.\n",
        "\\begin{tabular}{lrrrrrrrr}\n",
        "Branch & $k$ & MRR & Chance & MRR lift [95\\% CI] & Top-1 & Top-3 & 5-way & 10-way \\\\\n",
        "\\hline\n",
    ]
    for row in sorted(
        summary_rows, key=lambda item: (item["branch"], int(item["prefix_frames"]))
    ):
        values = [
            row["branch"].replace("_", r"\_"),
            row["prefix_frames"],
            metric(row["MRR"]),
            f"{float(row['MRR_chance']):.5f}",
            f"{float(row['MRR_lift_boot']):.2f} "
            f"[{float(row['MRR_lift_ci_lo']):.2f}, {float(row['MRR_lift_ci_hi']):.2f}]",
            metric(row["Top1"]),
            metric(row["Top3"]),
            metric(row["5way_acc"]),
            metric(row["10way_acc"]),
        ]
        tokens.extend(
            token
            for cell in values
            for token in NUMERIC.findall(cell.replace("\\pm", " "))
        )
        lines.append(" & ".join(tex_escape(value) for value in values) + " \\\\\n")
    lines.extend(
        [
            "\\end{tabular}\n",
            "",
            "% Paired differences; video-bootstrap 95% intervals.\n",
            "\\begin{tabular}{llrrrrr}\n",
            "Comparison & Metric & A & B & $\\Delta$ & CI low & CI high \\\\\n",
            "\\hline\n",
        ]
    )
    for row in paired_rows:
        values = [
            row["comparison"],
            row["metric"],
            f"{row['branch_a']}@{row['from_k']}",
            f"{row['branch_b']}@{row['to_k']}",
            f"{float(row['delta']):+.5f}",
            f"{float(row['ci95_lo']):+.5f}",
            f"{float(row['ci95_hi']):+.5f}",
        ]
        tokens.extend(
            token
            for cell in values
            for token in NUMERIC.findall(cell.replace("\\pm", " "))
        )
        lines.append(" & ".join(tex_escape(value) for value in values) + " \\\\\n")
    lines.append("\\end{tabular}\n")
    return "".join(lines), tokens


def format_locked_test(path: Path) -> tuple[str, list[str]]:
    if not path.is_file():
        return (f"% Previously reported locked test summary missing: {path}\n", [])
    rows = read_csv(path)
    if not rows:
        return ("% Previously reported locked test summary is empty.\n", [])
    fields = list(rows[0])
    numeric_fields = []
    for field in fields:
        try:
            for row in rows:
                float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        numeric_fields.append(field)
    label_fields = [field for field in fields if field not in numeric_fields][:2]
    values = label_fields + numeric_fields
    if not values:
        return ("% No numeric fields in previously reported locked test summary.\n", [])
    columns = "l" * len(label_fields) + "r" * len(numeric_fields)
    lines = [
        "% locked candidate test, evaluated once, already reported.\n",
        f"\\begin{{tabular}}{{{columns}}}\n",
        " & ".join(tex_escape(value) for value in values) + " \\\\\n",
        "\\hline\n",
    ]
    tokens: list[str] = []
    for row in rows:
        cells = [tex_escape(row.get(field, "")) for field in values]
        lines.append(" & ".join(cells) + " \\\\\n")
        tokens.extend(
            token for cell in cells for token in NUMERIC.findall(cell)
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

    generated: dict[str, tuple[str, list[str]]] = {}
    missing: list[str] = []
    if not args.daily_only:
        scene_summary = ROOT / "results/scene_cv_v1/summary.csv"
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
        generated["e1_zero_shot_v1.tex"] = format_e1_tables(
            E1_RESULTS / "summary.csv", E1_RESULTS / "paired_differences.csv"
        )
        generated["locked_test_full_v1.tex"] = format_locked_test(
            LOCKED_TEST_SUMMARY
        )
    for filename, (title, path, columns) in DAILY_TABLES.items():
        generated[filename] = format_daily_table(title, path, columns)

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
    if not args.daily_only:
        print("Locked-test snippet copies only the previously reported summary CSV.")

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
