from __future__ import annotations

"""Build matplotlib figures from an ETA / CrossChannel run directory (Ant-RH).

Reads ``metrics_summary.json``, ``aco_history.json``, etc. under a run folder and
writes ``fig_result_*.png``. Invoked by ``run_with_result_figures.py`` or standalone
``python -m`` / direct run from repo root.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return str(path)


def _value(x, default=0.0):
    return default if x is None else x


def plot_aco_convergence(aco_history: List[Dict[str, Any]], out: Path) -> Optional[str]:
    if not aco_history:
        return None
    x = [r.get('iteration', i + 1) for i, r in enumerate(aco_history)]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    plotted = False
    for key, label in [
        ('max_score', 'max score'),
        ('mean_score', 'mean score'),
        ('best_leaf_energy', 'best leaf energy'),
    ]:
        y = [r.get(key) for r in aco_history]
        if any(v is not None for v in y):
            ax.plot(x, [_value(v, np.nan) for v in y], linewidth=1.6, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title('ACO convergence from real run history')
    ax.set_xlabel('iteration')
    ax.set_ylabel('value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(out / 'fig_result_aco_convergence.png')


def plot_eta(aco_history: List[Dict[str, Any]], out: Path) -> Optional[str]:
    if not aco_history:
        return None
    x = [r.get('iteration', i + 1) for i, r in enumerate(aco_history)]
    eta = [r.get('eta_s') for r in aco_history]
    elapsed = [r.get('elapsed_s') for r in aco_history]
    if all(v is None for v in eta):
        return None
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, [_value(v, np.nan) for v in eta], linewidth=1.8, label='ETA, s')
    if any(v is not None for v in elapsed):
        ax.plot(x, [_value(v, np.nan) for v in elapsed], linewidth=1.2, label='elapsed, s')
    ax.set_title('ETA and elapsed time from run')
    ax.set_xlabel('iteration')
    ax.set_ylabel('seconds')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return savefig(out / 'fig_result_eta.png')


def plot_stage_timings(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    timings = metrics.get('stage_timings_s') or {}
    if not timings:
        return None
    names = list(timings.keys())
    vals = [float(timings[n]) for n in names]
    fig = plt.figure(figsize=(10, max(4, 0.35 * len(names))))
    ax = fig.add_subplot(111)
    y = np.arange(len(names))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('seconds')
    ax.set_title('Pipeline stage timings')
    ax.grid(True, axis='x', alpha=0.3)
    return savefig(out / 'fig_result_stage_timings.png')


def plot_hit_rate(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    hit = (metrics.get('hit_stats') or {})
    if not hit:
        return None
    rate = hit.get('hit_rate')
    n = hit.get('known_zeros_in_interval')
    hits = hit.get('hits')
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.bar(['hit rate'], [0.0 if rate is None else float(rate)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('fraction')
    ax.set_title(f'Known-zero hit rate: {hits}/{n}')
    ax.grid(True, axis='y', alpha=0.3)
    return savefig(out / 'fig_result_hit_rate.png')


def plot_agent_summary(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    stats = metrics.get('agent_stats') or {}
    rows = [(k, v) for k, v in stats.items() if isinstance(v, dict) and 'mean_score' in v]
    if not rows:
        return None
    names = [k for k, _ in rows]
    mean_scores = [0.0 if v.get('mean_score') is None else float(v.get('mean_score')) for _, v in rows]
    unique_terms = [0.0 if v.get('unique_terminal_nodes') is None else float(v.get('unique_terminal_nodes')) for _, v in rows]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    width = 0.38
    ax.bar(x - width/2, mean_scores, width, label='mean score')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, unique_terms, width, label='unique terminals', alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_title('Agent-type performance from run')
    ax.set_ylabel('mean path score')
    ax2.set_ylabel('unique terminal nodes')
    ax.grid(True, axis='y', alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='best')
    return savefig(out / 'fig_result_agent_summary.png')


def plot_channel_alignment(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    ch = metrics.get('channel_stats') or {}
    align = ch.get('channel_alignment_corr') or {}
    names = sorted({p for key in align.keys() for p in key.split('__')})
    if not names:
        return None
    idx = {n: i for i, n in enumerate(names)}
    mat = np.eye(len(names))
    for key, val in align.items():
        if val is None:
            continue
        a, b = key.split('__')
        mat[idx[a], idx[b]] = mat[idx[b], idx[a]] = float(val)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title('Cross-channel pheromone alignment')
    fig.colorbar(im, ax=ax, label='correlation')
    return savefig(out / 'fig_result_channel_alignment.png')


def plot_pheromone_metrics(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    ch = metrics.get('channel_stats') or {}
    if not ch:
        return None
    keys = ['shared_gini', 'shared_top_10_percent_mass', 'shared_entropy']
    vals = [ch.get(k) for k in keys]
    if all(v is None for v in vals):
        return None
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar([k.replace('shared_', '') for k in keys], [0.0 if v is None else float(v) for v in vals])
    ax.set_ylim(0, 1.05)
    ax.set_title('Shared pheromone concentration metrics')
    ax.grid(True, axis='y', alpha=0.3)
    return savefig(out / 'fig_result_pheromone_metrics.png')


def plot_ramsey_summary(metrics: Dict[str, Any], out: Path) -> Optional[str]:
    r = metrics.get('ramsey_stats') or {}
    if not r:
        return None
    keys = ['mean_run_length', 'shuffled_mean_run_length', 'ramsey_score']
    vals = [r.get(k) for k in keys]
    if all(v is None for v in vals):
        return None
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar([k.replace('_', '\n') for k in keys], [0.0 if v is None else float(v) for v in vals])
    ax.set_title('Ramsey/run-length structure from coloring')
    ax.grid(True, axis='y', alpha=0.3)
    return savefig(out / 'fig_result_ramsey_summary.png')


def copy_existing_pngs(results_dir: Path, out: Path) -> List[str]:
    copied = []
    for p in sorted(results_dir.glob('*.png')):
        target = out / p.name
        if p.resolve() != target.resolve():
            target.write_bytes(p.read_bytes())
        copied.append(str(target))
    return copied


def make_latex_include(fig_paths: List[str], out: Path, relative_to: Path) -> str:
    tex = out / 'paper_figures_from_results.tex'
    lines = [
        '% Auto-generated by figures_from_results.py',
        '% Include this file from your paper after \\usepackage{graphicx}.',
        '',
    ]
    captions = {
        'energy_landscape_candidates.png': 'Energy landscape on the critical line with refined zero candidates from the run.',
        'tree_energy_by_level.png': 'Fractal tree energy by level, computed from the actual node energies.',
        'pheromone_distribution.png': 'Shared pheromone distribution after ACO.',
        'pheromone_channels_summary.png': 'Pheromone-channel summary across DTES-agent types.',
        'ramsey_coloring.png': 'Ramsey coloring induced by dominant transitions at the target tree level.',
        'ramsey_run_lengths.png': 'Run-length histogram for the induced Ramsey coloring.',
        'agent_path_scores.png': 'Agent path-score distribution by DTES-agent type.',
        'aco_max_score.png': 'ACO max score over iterations.',
        'aco_best_energy.png': 'Best leaf energy over iterations.',
        'aco_terminal_diversity.png': 'Terminal-node diversity over iterations.',
        'aco_eta.png': 'ETA estimate over ACO iterations.',
        'fig_result_aco_convergence.png': 'Combined ACO convergence diagnostics from run history.',
        'fig_result_eta.png': 'ETA and elapsed time from the recorded run history.',
        'fig_result_stage_timings.png': 'Measured wall-clock time per pipeline stage.',
        'fig_result_hit_rate.png': 'Known-zero localization hit rate on the configured interval.',
        'fig_result_agent_summary.png': 'Agent-type performance and terminal diversity from the run.',
        'fig_result_channel_alignment.png': 'Cross-channel pheromone alignment matrix.',
        'fig_result_pheromone_metrics.png': 'Shared pheromone concentration metrics.',
        'fig_result_ramsey_summary.png': 'Ramsey score and baseline comparison from induced coloring.',
    }
    for p in fig_paths:
        pp = Path(p)
        if not pp.exists() or pp.suffix.lower() != '.png':
            continue
        rel = os.path.relpath(pp, relative_to).replace('\\', '/')
        cap = captions.get(pp.name, pp.stem.replace('_', ' '))
        label = 'fig:' + pp.stem.replace('_', '-')
        lines.extend([
            '\\begin{figure}[t]',
            '\\centering',
            f'\\includegraphics[width=0.92\\linewidth]{{{rel}}}',
            f'\\caption{{{cap}}}',
            f'\\label{{{label}}}',
            '\\end{figure}',
            '',
        ])
    tex.write_text('\n'.join(lines), encoding='utf-8')
    return str(tex)


def build_figures(results_dir: Path, out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    metrics = read_json(results_dir / 'metrics_summary.json', {})
    aco_history = read_json(results_dir / 'aco_history.json', metrics.get('aco_history', [])) or []
    generated: List[Optional[str]] = []
    generated.extend(copy_existing_pngs(results_dir, out_dir))
    generated.append(plot_aco_convergence(aco_history, out_dir))
    generated.append(plot_eta(aco_history, out_dir))
    generated.append(plot_stage_timings(metrics, out_dir))
    generated.append(plot_hit_rate(metrics, out_dir))
    generated.append(plot_agent_summary(metrics, out_dir))
    generated.append(plot_channel_alignment(metrics, out_dir))
    generated.append(plot_pheromone_metrics(metrics, out_dir))
    generated.append(plot_ramsey_summary(metrics, out_dir))
    fig_paths = [p for p in generated if p]
    tex_path = make_latex_include(fig_paths, out_dir, out_dir)
    manifest = {
        'source_results_dir': str(results_dir),
        'figures_dir': str(out_dir),
        'n_figures': len(fig_paths),
        'figures': fig_paths,
        'latex_include': tex_path,
        'metrics_keys': sorted(metrics.keys()) if isinstance(metrics, dict) else [],
    }
    (out_dir / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description='Build paper-ready figures from Fractal-DTES-ACO-Zeta run outputs.')
    ap.add_argument('--results', default='fractal_dtes_aco_eta_output', help='Directory containing metrics_summary.json/aco_history.json and base PNGs.')
    ap.add_argument('--out', default='paper_figures_from_results', help='Output directory for derived figures and LaTeX include file.')
    args = ap.parse_args()
    manifest = build_figures(Path(args.results), Path(args.out))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
