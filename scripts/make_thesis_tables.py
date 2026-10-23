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
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Write the v4 booktabs tables (ladder, selective, EAD, kNN, E1b v2, selective frozen, "
        "adapter dims, LIBERO goal ablation) into --output-dir (must not exist).",
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
    if args.v4:
        return main_v4(args.output_dir)
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


# ----------------------------------------------------------------------
# v4 tables (booktabs; numbers read from result files only)
# ----------------------------------------------------------------------

import json  # noqa: E402


def _ci(point, lo, hi, nd=4, sign=True):
    f = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return f"{f.format(float(point))} [{f.format(float(lo))}, {f.format(float(hi))}]"


def _verdict_text(verdicts: dict, keys) -> str:
    parts = []
    for key in keys:
        v = verdicts.get(key, {})
        parts.append(f"{key}: {'PASS' if v.get('pass') else 'FAIL'}")
    return "; ".join(parts)


def _booktabs(source: str, header: list[str], rows: list[list[str]], note: str) -> str:
    cols = "l" + "r" * (len(header) - 1)
    out = [f"% Generated by scripts/make_thesis_tables.py --v4 from {source}; train/val evidence unless stated.\n",
           f"\\begin{{tabular}}{{{cols}}}\n", "\\toprule\n",
           " & ".join(tex_escape(h) for h in header) + " \\\\\n", "\\midrule\n"]
    out += [" & ".join(tex_escape(c) for c in r) + " \\\\\n" for r in rows]
    out += ["\\midrule\n", f"\\multicolumn{{{len(header)}}}{{l}}{{{tex_escape(note)}}} \\\\\n",
            "\\bottomrule\n", "\\end{tabular}\n"]
    return "".join(out)


def _paired_lookup(rows, metric_key, metric, k):
    return next(r for r in rows if r[metric_key] == metric and int(r["k"]) == k)


def v4_ladder() -> str:
    d = ROOT / "results/ladder_v1"
    summ = read_csv(d / "summary.csv")
    pair = read_csv(d / "paired_vs_baseline.csv")
    ver = json.loads((d / "config.json").read_text())["verdicts"]
    rows = []
    for k in (2, 4, 6, 8):
        b = next(r for r in summ if r["pipeline"] == "baseline" and int(r["k"]) == k)
        n = next(r for r in summ if r["pipeline"] == "nested_selected" and int(r["k"]) == k)
        pm = _paired_lookup(pair, "metric", "MRR", k)
        p5 = _paired_lookup(pair, "metric", "kway5", k)
        rows.append([str(k), metric(b["MRR"]), metric(n["MRR"]), _ci(pm["delta"], pm["ci95_lo"], pm["ci95_hi"]),
                     metric(b["5way_acc"]), metric(n["5way_acc"]), _ci(p5["delta"], p5["ci95_lo"], p5["ci95_hi"])])
    return _booktabs("results/ladder_v1", ["k", "MRR base", "MRR nested", "Delta MRR [95% CI]",
                                            "5-way base", "5-way nested", "Delta 5-way [95% CI]"],
                     rows, _verdict_text(ver, ("N1", "N2")))


def _selective(d: Path, verdict_key: str) -> str:
    summ = [r for r in read_csv(d / "summary.csv") if r["K"] == "5" and r["score"] == "margin"]
    ver = json.loads((d / "config.json").read_text())["verdicts"]
    rows = [[r["k"], metric(r["acc@100"]), metric(r["acc@50"]), metric(r["acc@20"]),
             _ci(r["gain50"], r["gain50_ci_lo"], r["gain50_ci_hi"]), metric(r["AURC"]),
             metric(r["random_rejection_AURC"])] for r in summ]
    return _booktabs(str(d.relative_to(ROOT)), ["k", "acc@100", "acc@50", "acc@20", "acc@50 - acc@100 [95% CI]",
                                                "AURC", "random AURC"], rows, _verdict_text(ver, (verdict_key,)))


def v4_ead() -> str:
    d = ROOT / "results/ead_v1"
    ver = json.loads((d / "config.json").read_text())["verdicts"]
    rows = [[r["budget"], f"{float(r['realized_mean_frames']):.3f}", metric(r["early_decision_acc"]),
             metric(r["fixed_k_interpolated_acc"]), _ci(r["delta"], r["delta_ci_lo"], r["delta_ci_hi"])]
            for r in read_csv(d / "summary.csv")]
    return _booktabs("results/ead_v1", ["budget", "mean frames", "early-decision acc", "fixed-k interp. acc",
                                         "delta [95% CI]"], rows, _verdict_text(ver, ("S2",)))


def v4_knn() -> str:
    d = ROOT / "results/knn_cache_v1"
    summ = read_csv(d / "summary.csv")
    pair = read_csv(d / "paired_vs_baseline.csv")
    ver = json.loads((d / "config.json").read_text())["verdicts"]
    rows = []
    for k in (2, 4, 6, 8):
        b = next(r for r in summ if r["pipeline"] == "baseline" and int(r["k"]) == k)
        n = next(r for r in summ if r["pipeline"] == "knn_beta_selected" and int(r["k"]) == k)
        pm = _paired_lookup(pair, "metric", "MRR", k)
        p5 = _paired_lookup(pair, "metric", "kway5", k)
        rows.append([str(k), metric(b["MRR"]), metric(n["MRR"]), _ci(pm["delta"], pm["ci95_lo"], pm["ci95_hi"]),
                     metric(b["5way_acc"]), metric(n["5way_acc"]), _ci(p5["delta"], p5["ci95_lo"], p5["ci95_hi"])])
    return _booktabs("results/knn_cache_v1", ["k", "MRR base", "MRR kNN", "Delta MRR [95% CI]",
                                               "5-way base", "5-way kNN", "Delta 5-way [95% CI]"],
                     rows, _verdict_text(ver, ("K1",)))


def v4_e1b() -> str:
    d = ROOT / "results/e1b_smolvlm2_v2"
    summ = [r for r in read_csv(d / "summary.csv") if r["score"] == "pmi_mean"]
    pair = [r for r in read_csv(d / "paired_vs_siglip.csv") if r["smolvlm2_score"] == "pmi_mean"]
    ver = json.loads((d / "analysis_config.json").read_text())["verdicts"]
    rows = []
    for r in summ:
        p = next(x for x in pair if x["k"] == r["k"])
        rows.append([r["k"], r["n_queries"], metric(r["5way_acc"]),
                     _ci(r["5way_lift"], r["5way_lift_ci_lo"], r["5way_lift_ci_hi"], 3, sign=False),
                     metric(p["siglip_acc"]), _ci(p["delta"], p["ci95_lo"], p["ci95_hi"])])
    return _booktabs("results/e1b_smolvlm2_v2", ["k", "n", "SmolVLM2 5-way (PMI)", "lift [95% CI]",
                                                  "SigLIP 5-way", "E1b-B delta [95% CI]"],
                     rows, _verdict_text(ver, ("E1b-A",)) + "; E1b-B descriptive")


def v4_adapter() -> str:
    d = ROOT / "results/adapter_dims_v1"
    res = json.loads((d / "results.json").read_text())["rows"]
    tests = list(res[0]["results"])
    rows = [[t.replace("_", " ")] + ["PASS" if r["results"][t] else "FAIL" for r in res] for t in tests]
    return _booktabs("results/adapter_dims_v1", ["test"] + [f"d={r['d']}" for r in res], rows,
                     "implementation correctness only")


def v4_libero() -> str:
    d = ROOT / "results/libero_goal_ablation_v1"
    if not (d / "summary.csv").is_file():
        return "% LIBERO goal ablation unavailable: results/libero_goal_ablation_v1/summary.csv not found.\n"
    ver = json.loads((d / "config.json").read_text())["verdicts"]
    rows = [[r["condition"], f"{r['successes']}/{r['n_episodes']}", f"{float(r['success_rate']):.3f}",
             f"[{float(r['wilson_lo']):.3f}, {float(r['wilson_hi']):.3f}]"] for r in read_csv(d / "summary.csv")]
    note = "; ".join(f"{k}: {'PASS' if v['pass'] else 'FAIL'} ({v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}])"
                     for k, v in ver.items())
    return _booktabs("results/libero_goal_ablation_v1", ["condition", "success/n", "rate", "Wilson 95%"], rows, note)


def main_v4(output_dir: Path) -> int:
    if output_dir.exists():
        print(f"Refusing to overwrite {output_dir}", file=sys.stderr)
        return 2
    builders = {
        "ladder_v1.tex": v4_ladder,
        "selective_e1_v1.tex": lambda: _selective(ROOT / "results/selective_e1_v1", "S1"),
        "ead_v1.tex": v4_ead,
        "knn_cache_v1.tex": v4_knn,
        "e1b_smolvlm2_v2.tex": v4_e1b,
        "selective_frozen_v1.tex": lambda: _selective(ROOT / "results/selective_frozen_v1", "S1b"),
        "adapter_dims_v1.tex": v4_adapter,
        "libero_goal_ablation_v1.tex": v4_libero,
    }
    output_dir.mkdir(parents=True)
    for name, build in builders.items():
        (output_dir / name).write_text(build(), encoding="utf-8")
    print(f"Wrote {len(builders)} v4 snippets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
