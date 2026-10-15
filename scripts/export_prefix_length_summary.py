#!/usr/bin/env python3
"""Export the existing temporal-progression summary into CSV/MD/SVG artifacts.

This script packages the already-executed results from the repo's temporal
progression experiment into a compact readable artifact set without rerunning
heavy training.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


DEFAULT_SUMMARY = Path('/media/dhqg/d3/vlia_outputs/temporal_progression_v1/summary.json')
DEFAULT_OUT = Path('/home/dhqg/vlia-uet/results/prefix_length_ablation_v1')


def mean_std(values):
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_float(x):
    return f'{x:.6f}' if isinstance(x, float) else x


def main():
    summary_path = DEFAULT_SUMMARY
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    summary = payload.get('summary', {})
    rows = []
    for condition, stats in sorted(summary.items()):
        prefix_rows = stats.get('prefix_mrr', {})
        if prefix_rows:
            for prefix_key in sorted(prefix_rows, key=lambda s: int(s)):
                entry = prefix_rows[prefix_key]
                vals = entry.get('values', [])
                mean_mrr, std_mrr = mean_std(vals)
                rows.append({
                    'condition': condition,
                    'prefix_frames': int(prefix_key),
                    'mrr_mean': mean_mrr,
                    'mrr_std': std_mrr,
                    'top1_mean': float('nan'),
                    'top1_std': float('nan'),
                    'top3_mean': float('nan'),
                    'top3_std': float('nan'),
                    'n_examples': len(vals),
                })
        else:
            rows.append({
                'condition': condition,
                'prefix_frames': 'full',
                'mrr_mean': stats.get('mrr_mean', float('nan')),
                'mrr_std': stats.get('mrr_sample_std', float('nan')),
                'top1_mean': stats.get('top1_mean', float('nan')),
                'top1_std': stats.get('top1_sample_std', float('nan')),
                'top3_mean': stats.get('top3_mean', float('nan')),
                'top3_std': stats.get('top3_sample_std', float('nan')),
                'n_examples': len(stats.get('mrr_values', [])),
            })

    # Full summary JSON in repo
    out_json = out_dir / 'prefix_length_summary.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'source_summary': str(summary_path),
            'rows': rows,
            'final_metrics_by_condition': {
                cond: {
                    'mrr_mean': stats.get('mrr_mean'),
                    'mrr_sample_std': stats.get('mrr_sample_std'),
                    'top1_mean': stats.get('top1_mean'),
                    'top1_sample_std': stats.get('top1_sample_std'),
                    'top3_mean': stats.get('top3_mean'),
                    'top3_sample_std': stats.get('top3_sample_std'),
                    'n_seeds': len(stats.get('mrr_values', [])),
                }
                for cond, stats in summary.items()
            },
        }, f, indent=2)

    CSV_PATH = out_dir / 'prefix_length_summary.csv'
    fieldnames = ['condition', 'prefix_frames', 'mrr_mean', 'mrr_std', 'top1_mean', 'top1_std', 'top3_mean', 'top3_std', 'n_examples']
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ['mrr_mean', 'mrr_std', 'top1_mean', 'top1_std', 'top3_mean', 'top3_std']:
                if isinstance(out.get(key), float) and out[key] != out[key]:
                    out[key] = 'NA'
            writer.writerow(out)

    def human_value(value, digits=6):
        if value is None or (isinstance(value, float) and value != value):
            return 'NA'
        return f'{float(value):.{digits}f}'

    md_lines = [
        '# Prefix-length goal disambiguation summary',
        '',
        'Source: ' + str(summary_path),
        '',
        '| Condition | Prefix frames | MRR mean | MRR std | Top-1 mean | Top-3 mean | Eval examples |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        pf = row['prefix_frames']
        md_lines.append(
            f"| {row['condition']} | {pf} | {human_value(row['mrr_mean'])} | {human_value(row['mrr_std'])} | "
            f"{human_value(row['top1_mean'])} | {human_value(row['top3_mean'])} | {row['n_examples']} |"
        )
    md_lines.extend([
        '',
        '## Interpretation',
        '',
        '- The early prefix curves increase from 2 to 6 frames and are near their best at 8 frames.',
        '- Gains are monotonic but modest on the current split; the dataset is small and there is no untouched test set.',
        '- Same-task versus cross-task splits are not available in the current summary artifact, so selective-risk analysis was not computed here.',
        '',
        '## Notes',
        '',
        '- Reported metrics are validation-only under the existing source-video split.',
        '- The original temporal progression output remains under /media/dhqg/d3/vlia_outputs/temporal_progression_v1.',
    ])
    (out_dir / 'prefix_length_summary.md').write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    # Build a minimal SVG plot of MRR vs observed frames for the conditions with prefix data.
    prefix_conditions = {c: {} for c, stats in summary.items() if stats.get('prefix_mrr')}
    plot_svg = out_dir / 'prefix_length_summary.svg'
    x_values = [2, 4, 6, 8]
    y_series = {}
    for cond, stats in summary.items():
        prefix_data = stats.get('prefix_mrr', {})
        if not prefix_data:
            continue
        y_series[cond] = [prefix_data[str(x)]['mean'] for x in x_values if str(x) in prefix_data]

    # Keep only the common prefix lengths present in each series.
    if y_series:
        width = 760
        height = 420
        left, top, right, bottom = 60, 30, 700, 340
        plot_w = right - left
        plot_h = bottom - top
        min_y = min(min(vals) for vals in y_series.values()) * 0.95
        max_y = max(max(vals) for vals in y_series.values()) * 1.05
        if max_y == min_y:
            max_y = min_y + 0.01

        def map_x(ix):
            return left + (ix / (max(x_values) - min(x_values) if (max(x_values) - min(x_values)) else 1)) * plot_w

        def map_y(v):
            return bottom - ((v - min_y) / (max_y - min_y if (max_y - min_y) else 1.0)) * plot_h

        lines = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        ]
        # y ticks
        for i in range(5):
            v = min_y + (max_y - min_y) * (i / 4)
            y = map_y(v)
            lines.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#ddd"/>')
            lines.append(f'<text x="10" y="{y+4}" font-size="11">{v:.3f}</text>')
        # x ticks
        for x in x_values:
            px = map_x(x)
            lines.append(f'<line x1="{px}" y1="{bottom}" x2="{px}" y2="{bottom+6}" stroke="black"/>')
            lines.append(f'<text x="{px-8}" y="{bottom+18}" font-size="11">{x}</text>')
        lines.append(f'<text x="{left + plot_w/2 - 45}" y="{height-8}" font-size="12">Observed frames</text>')
        lines.append(f'<text x="15" y="{top+60}" font-size="12" transform="rotate(-90 15 60)">MRR</text>')
        colors = {'gru_final': '#1f77b4', 'gru_progression': '#ff7f0e'}
        for cond, vals in y_series.items():
            points = ' '.join(f'{map_x(x):.1f},{map_y(v):.1f}' for x, v in zip(x_values, vals))
            lines.append(f'<polyline fill="none" stroke="{colors.get(cond, "#000000")}" stroke-width="2" points="{points}"/>')
            last_x = x_values[-1]
            last_v = vals[-1]
            lines.append(f'<text x="{map_x(last_x)+5}" y="{map_y(last_v)-8}" font-size="11" fill="{colors.get(cond, "#000000")}">{cond}</text>')
        lines.append('</svg>')
        plot_svg.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'Wrote package to {out_dir}')


if __name__ == '__main__':
    main()
