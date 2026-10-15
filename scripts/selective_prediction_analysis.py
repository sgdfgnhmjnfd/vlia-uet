#!/usr/bin/env python3
"""Selective prediction baseline for early prefix goal disambiguation.

The goal is to evaluate simple predict-vs-defer decision rules using the saved
validation outputs from the temporal progression experiment. We use the existing
source-video split and the real checkpoint weights; thresholds are tuned on the
validation split only for a simple risk-vs-coverage analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_temporal_progression_intention import (  # noqa: E402
    MeanPoolBaseline,
    TemporalGRUEncoder,
    load_pca_transform,
    load_temporal_split,
    stack_split,
)


def build_positive_mask(labels):
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)
    n = len(labels)
    mask = torch.zeros((n, n), dtype=torch.bool)
    for indices in groups.values():
        idx = torch.tensor(indices, dtype=torch.long)
        mask[idx[:, None], idx[None, :]] = True
    return mask


def init_model(condition: str, device: torch.device) -> torch.nn.Module:
    if condition == "mean_pool":
        model = MeanPoolBaseline(hidden_dim=512, output_dim=256, dropout=0.0).to(device)
    elif condition in {"gru_final", "gru_progression"}:
        model = TemporalGRUEncoder(frame_dim=256, hidden_dim=256, output_dim=256, dropout=0.0).to(device)
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    return model


def load_best_model(checkpoint_path: Path, condition: str, device: torch.device) -> torch.nn.Module:
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = init_model(condition, device)
    state = obj.get("model_state_dict", obj)
    if isinstance(state, dict) and any(k.startswith("model") for k in state):
        pass
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def compute_prefix_prediction(model, frames, why_target, why_texts, condition: str, prefix_frames: int, device: torch.device):
    with torch.no_grad():
        frames = frames.to(device)
        why_target = why_target.to(device)
        if condition == "mean_pool":
            prefix = frames[:, :prefix_frames]
            pred = model(prefix)
        else:
            prefix_z = model.forward_prefixes(frames)
            pred = prefix_z[:, prefix_frames - 1]
        pred = F.normalize(pred, dim=-1)
        sim = pred @ why_target.T
        probs = torch.softmax(sim / 0.07, dim=1)
        ranking = torch.argsort(sim, dim=1, descending=True)
        positive_mask = build_positive_mask(why_texts).to(device)
        n = sim.shape[0]
        top1_idx = ranking[:, 0]
        top2_idx = ranking[:, 1]
        top1_scores = sim[torch.arange(n), top1_idx]
        top2_scores = sim[torch.arange(n), top2_idx]
        top1_prob = probs[torch.arange(n), top1_idx]
        top2_prob = probs[torch.arange(n), top2_idx]
        top1_margin = top1_scores - top2_scores
        positive_set = positive_mask.to(dtype=torch.bool)
        correct = torch.zeros(n, dtype=torch.bool, device=device)
        for i in range(n):
            correct[i] = (top1_idx[i].item() in torch.nonzero(positive_set[i]).view(-1).tolist())
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1) / math.log(max(2, probs.shape[1]))
        return {
            "pred": pred.detach(),
            "sim": sim.detach(),
            "top1_prob": top1_prob.detach().cpu().numpy(),
            "margin": top1_margin.detach().cpu().numpy(),
            "entropy": entropy.detach().cpu().numpy(),
            "correct": correct.detach().cpu().numpy(),
            "top1_idx": top1_idx.detach().cpu().numpy(),
            "top2_idx": top2_idx.detach().cpu().numpy(),
            "top1_scores": top1_scores.detach().cpu().numpy(),
            "top2_scores": top2_scores.detach().cpu().numpy(),
        }


def aggregate_threshold_metrics(correct, score, threshold, accept_gt):
    accepted = score >= threshold if accept_gt else score <= threshold
    n = len(correct)
    total_accepted = int(accepted.sum())
    if total_accepted == 0:
        return {
            "threshold": float(threshold),
            "coverage": 0.0,
            "error_rate": 1.0,
            "accepted": 0,
            "errors": 0,
            "correct": 0,
            "risk": 1.0,
            "risk_ci_low": 1.0,
            "risk_ci_high": 1.0,
            "coverage_ci_low": 0.0,
            "coverage_ci_high": 0.0,
        }
    errors = int((accepted & (~correct)).sum())
    accepted_correct = int((accepted & correct).sum())
    coverage = float(accepted.mean())
    error_rate = float(errors / total_accepted)
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "error_rate": error_rate,
        "risk": error_rate,
        "accepted": total_accepted,
        "errors": errors,
        "correct": accepted_correct,
    }


def bootstrap_threshold_metrics(correct, score, threshold, accept_gt, n_boot=400, seed=0):
    accepted = (score >= threshold) if accept_gt else (score <= threshold)
    ok = np.asarray(correct, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    n = accepted.shape[0]
    rng = np.random.default_rng(seed)
    risks = []
    coverages = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mask = accepted[idx]
        if not mask.any():
            risks.append(1.0)
            coverages.append(0.0)
            continue
        risk = float((~ok[idx][mask]).mean())
        coverage = float(mask.mean())
        risks.append(risk)
        coverages.append(coverage)
    risks = np.asarray(risks)
    coverages = np.asarray(coverages)
    base = aggregate_threshold_metrics(ok, score, threshold, accept_gt)
    base["risk_ci_low"] = float(np.quantile(risks, 0.025))
    base["risk_ci_high"] = float(np.quantile(risks, 0.975))
    base["coverage_ci_low"] = float(np.quantile(coverages, 0.025))
    base["coverage_ci_high"] = float(np.quantile(coverages, 0.975))
    return base


def evaluate_policy(results, policy: str, thresholds: list[float]):
    metrics = []
    score = results[policy]
    correct = results["correct"]
    if policy == "entropy":
        accept_gt = False
    else:
        accept_gt = True
    for th in thresholds:
        metrics.append(bootstrap_threshold_metrics(correct, score, th, accept_gt))
    return metrics


def make_svg_curve(points, title, path):
    # points is list of (coverage, risk) floats in [0,1]
    width, height = 760, 420
    pad_left, pad_right, pad_top, pad_bottom = 60, 30, 30, 50
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    xs = [pad_left + (p[0] * plot_w) for p in points]
    ys = [height - pad_bottom - (p[1] * plot_h) for p in points]
    if len(points) > 1:
        line = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    else:
        line = f"{xs[0]:.2f},{ys[0]:.2f}"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{pad_left}" y1="{height-pad_bottom}" x2="{width-pad_right}" y2="{height-pad_bottom}" stroke="black"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height-pad_bottom}" stroke="black"/>
  <text x="{width/2 - 60}" y="{height-12}" font-size="12">coverage</text>
  <text x="15" y="{height/2}" font-size="12" transform="rotate(-90 15 {height/2})">risk</text>
  <text x="{pad_left}" y="20" font-size="14">{title}</text>
  <polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{line}"/>
</svg>
'''
    path.write_text(svg, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-cache", type=Path, default=Path('/media/dhqg/d3/datasets/egointent/cache/temporal_intention_v1'))
    parser.add_argument("--stage-a-cache", type=Path, default=Path('/media/dhqg/d3/datasets/egointent/cache/stage_a_v0'))
    parser.add_argument("--why-pca", type=Path, default=Path('/media/dhqg/d3/vlia_outputs/stage_a_clean_v1/why_pca_960_to_256.pt'))
    parser.add_argument("--checkpoint-root", type=Path, default=Path('/media/dhqg/d3/vlia_outputs/temporal_progression_v1'))
    parser.add_argument("--conditions", nargs='+', default=['mean_pool', 'gru_final', 'gru_progression'])
    parser.add_argument("--prefix-lengths", nargs='+', type=int, default=[2, 4, 6, 8])
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path('/home/dhqg/vlia-uet/results/selective_decision_v1'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    why_transform = load_pca_transform(args.why_pca)
    val_rows = load_temporal_split(args.temporal_cache, args.stage_a_cache, 'val')
    val_frames, val_why_target, val_why_texts = stack_split(val_rows, why_transform, device)

    all_rows = []
    for condition in args.conditions:
        for seed in args.seeds:
            ckpt = args.checkpoint_root / condition / f'seed_{seed}' / 'best.pt'
            if not ckpt.exists():
                print(f'SKIP missing checkpoint: {ckpt}')
                continue
            model = load_best_model(ckpt, condition, device)
            for prefix_len in args.prefix_lengths:
                if prefix_len > val_frames.shape[1]:
                    continue
                metrics = compute_prefix_prediction(model, val_frames, val_why_target, val_why_texts, condition, prefix_len, device)
                for policy_name, policy_code in [('confidence', metrics['top1_prob']), ('margin', metrics['margin']), ('entropy', metrics['entropy'])]:
                    if policy_name == 'confidence':
                        thresholds = np.unique(np.concatenate(([0.0], np.linspace(0.0, 0.95, 40), np.sort(metrics['top1_prob'].astype(float)))))
                        score = metrics['top1_prob']
                        accept_gt = True
                    elif policy_name == 'margin':
                        thresholds = np.unique(np.concatenate(([-1.0], np.linspace(-1.0, 1.0, 60), np.sort(metrics['margin'].astype(float)))))
                        score = metrics['margin']
                        accept_gt = True
                    else:
                        thresholds = np.unique(np.concatenate(([1.0], np.linspace(0.0, 1.0, 60), np.sort(metrics['entropy'].astype(float)))))
                        score = metrics['entropy']
                        accept_gt = False
                    results = evaluate_policy({
                        'correct': metrics['correct'],
                        'confidence': metrics['top1_prob'],
                        'margin': metrics['margin'],
                        'entropy': metrics['entropy'],
                    }, policy_name, list(thresholds))
                    for row in results:
                        all_rows.append({
                            'condition': condition,
                            'seed': seed,
                            'prefix_frames': prefix_len,
                            'policy': policy_name,
                            'threshold': row['threshold'],
                            'coverage': row['coverage'],
                            'error_rate': row['error_rate'],
                            'risk_ci_low': row.get('risk_ci_low', row['error_rate']),
                            'risk_ci_high': row.get('risk_ci_high', row['error_rate']),
                            'coverage_ci_low': row.get('coverage_ci_low', row['coverage']),
                            'coverage_ci_high': row.get('coverage_ci_high', row['coverage']),
                            'accepted': row['accepted'],
                            'errors': row['errors'],
                            'correct': row['correct'],
                        })

    if not all_rows:
        raise RuntimeError('No rows generated; check the checkpoint and path configuration.')

    csv_path = args.output_dir / 'selective_decision_thresholds.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['condition','seed','prefix_frames','policy','threshold','coverage','error_rate','accepted','errors','correct'])
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = args.output_dir / 'selective_decision_thresholds.json'
    json_path.write_text(json.dumps(all_rows, indent=2), encoding='utf-8')

    # Compact summary: by condition + prefix + policy, choose the best threshold
    # under a conservative risk constraint. If no threshold meets the target, choose
    # the threshold with the lowest observed risk rather than accept-all defaults.
    summary = []
    groups = defaultdict(list)
    for row in all_rows:
        groups[(row['condition'], row['prefix_frames'], row['policy'])].append(row)
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['threshold'])
        candidates = [r for r in rows if r['coverage'] > 0 and r['error_rate'] <= 0.25]
        if not candidates:
            candidates = rows
        best = min(candidates, key=lambda r: (r['error_rate'], -r['coverage']))
        summary.append({
            'condition': key[0],
            'prefix_frames': key[1],
            'policy': key[2],
            'threshold': best['threshold'],
            'coverage': best['coverage'],
            'error_rate': best['error_rate'],
            'accepted': best['accepted'],
            'risk_ci_low': best.get('risk_ci_low', best['error_rate']),
            'risk_ci_high': best.get('risk_ci_high', best['error_rate']),
        })

    md_path = args.output_dir / 'selective_decision_summary.md'
    lines = [
        '# Selective prediction summary',
        '',
        'Validation-only decision baselines using the saved temporal progression checkpoints.',
        'These curves are reported as risk-vs-coverage summaries, not as validated semantic-evidence claims.',
        '',
        '| Condition | Prefix | Policy | Threshold | Coverage | Error rate | Accepted |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary:
        lines.append(f"| {row['condition']} | {row['prefix_frames']} | {row['policy']} | {row['threshold']:.4f} | {row['coverage']:.4f} | {row['error_rate']:.4f} | {row['accepted']} |")
    lines.extend([
        '',
        'Interpretation: the risk-vs-coverage curve is the relevant object here. A policy only counts as useful if it reduces coverage while keeping risk below the accepted baseline; the current validation-level score distributions do not support a strong defer signal.',
        'The results should therefore be read as a negative baseline for selective decision-making, not as evidence that the model understands semantic sufficiency of early evidence.',
    ])
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # Build a simple risk-vs-coverage SVG for the best condition per prefix.
    plot_dir = args.output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    for condition in args.conditions:
        for prefix_len in args.prefix_lengths:
            subset = [r for r in all_rows if r['condition'] == condition and r['prefix_frames'] == prefix_len and r['policy'] == 'confidence']
            if not subset:
                continue
            ordered = sorted(subset, key=lambda r: (r['coverage'], -r['error_rate']))
            pts = [(float(r['coverage']), float(r['error_rate'])) for r in ordered]
            if not pts:
                continue
            make_svg_curve(pts, f'{condition} prefix={prefix_len} confidence', plot_dir / f'{condition}_{prefix_len}_confidence.svg')

    print(f'Wrote CSV: {csv_path}')
    print(f'Wrote JSON: {json_path}')
    print(f'Wrote summary: {md_path}')


if __name__ == '__main__':
    main()
