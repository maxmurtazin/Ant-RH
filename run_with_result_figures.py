from __future__ import annotations

"""One-shot demo: run ``ETAFractalDTESACOZeta`` then build paper-style figures.

Expects ``figures_from_results`` (under ``validation/``) on ``PYTHONPATH`` or run
from repo root so imports resolve. Output defaults: ``fractal_dtes_aco_eta_output/``
and ``paper_figures_from_results/``.
"""

import argparse
import os
from fractal_dtes_aco_zeta_eta import ETAFractalDTESACOZeta
from fractal_dtes_aco_zeta_metrics import ZetaSearchConfig
from figures_from_results import build_figures


def main() -> None:
    ap = argparse.ArgumentParser(description='Run Fractal-DTES-ACO-Zeta and build figures directly from run outputs.')
    ap.add_argument('--out', default='fractal_dtes_aco_eta_output')
    ap.add_argument('--figures', default='paper_figures_from_results')
    ap.add_argument('--t-min', type=float, default=10.0)
    ap.add_argument('--t-max', type=float, default=40.0)
    ap.add_argument('--n-grid', type=int, default=2048)
    ap.add_argument('--tree-depth', type=int, default=8)
    ap.add_argument('--feature-levels', type=int, default=5)
    ap.add_argument('--n-ants', type=int, default=48)
    ap.add_argument('--n-iterations', type=int, default=60)
    ap.add_argument('--early-stop-patience', type=int, default=0)
    ap.add_argument('--spectral-validation', action='store_true')
    ap.add_argument('--spectral-k', type=int, default=50)
    ap.add_argument(
        '--spectral-potential-mode',
        default='neglog',
        choices=('neglog', 'log', 'inverse'),
    )
    ap.add_argument('--spectral-normalized-laplacian', action='store_true')
    ap.add_argument('--export-pheromone-max-n', type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = ZetaSearchConfig(
        t_min=args.t_min,
        t_max=args.t_max,
        n_grid=args.n_grid,
        tree_depth=args.tree_depth,
        feature_levels=args.feature_levels,
        n_ants=args.n_ants,
        n_iterations=args.n_iterations,
        max_ant_steps=20,
        top_candidate_nodes=16,
        verification_abs_tol=1e-6,
        refinement_subgrid=128,
        r0=6.0,
        mp_dps=50,
    )
    cfg.out_dir = args.out
    cfg.verbose_eta = True
    cfg.eta_log_every = 1
    cfg.eta_ema_alpha = 0.20
    cfg.early_stop_patience = args.early_stop_patience
    cfg.early_stop_min_delta = 1e-9
    cfg.metrics_out_path = os.path.join(args.out, 'metrics_summary.json')
    cfg.spectral_validation = args.spectral_validation
    cfg.spectral_k = args.spectral_k
    cfg.spectral_potential_mode = args.spectral_potential_mode
    cfg.spectral_normalized_laplacian = args.spectral_normalized_laplacian
    cfg.export_pheromone_max_n = args.export_pheromone_max_n

    searcher = ETAFractalDTESACOZeta(cfg)
    candidates = searcher.run()
    print('Candidates:', candidates)
    from pathlib import Path
    manifest = build_figures(Path(os.path.abspath(args.out)), Path(os.path.abspath(args.figures)))
    print('Figure manifest:', manifest)


if __name__ == '__main__':
    main()
