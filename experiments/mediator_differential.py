"""Mediator differential between scope classes + PC-neighbor lookup.

Given the convergence audit's scope classes (solved vs unsolved
on the baseline arm), surface trace-features that empirically
*predict* convergence class. Then look up which other variables
are PC-adjacent to the top differentials — those neighbors are
the candidate intervention targets for the next sweep.

Pipeline:
1. Read the corpus, classify envs by `rl.convergence`.
2. For the BASELINE arm only (we want the "natural" failure-mode
   signature, not signatures induced by an intervention):
   for each mediator path, compute Hedges' g of value across
   cells in unsolved envs vs cells in solved envs.
3. Run conservative-PC adjacency on a wide variable set (same
   set §4 of paper_full_range uses).
4. For each top-|g| mediator, list its PC-adjacent nodes.
5. The *other side* of these edges is where we head next: each
   adjacent node is a candidate variable for the substrate
   author to construct a new intervention targeting.

This is the framework's data-driven path for intervention
selection: convergence audit gives scope, mediator differential
ranks failure-mode signatures, PC adjacency points at upstream
causes / downstream effects. The *substrate author* still makes
the final call (which intervention from the literature targets
the named adjacent node), but the candidate set is no longer
literature pattern-matching from the room — it's empirically
ranked.

Usage:
    uv run python experiments/mediator_differential.py
    uv run python experiments/mediator_differential.py --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from corroborate.graph.discovery import discover_adjacency
from corroborate.corpus.persistence import read_runrows
from corroborate_rl.convergence import (
    classify_envs, envs_in_class, mediator_differential,
)


_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)

# Variable set for PC discovery — mirrors paper_full_range §4.
_PC_VARIABLES: tuple[str, ...] = (
    'arm_ddqn',
    'jensen_gap',
    'eval_best_burst_mean',
    'eval_final_mean',
    'mediator.q_gap_late',
    'mediator.q_gap_growth',
    'mediator.q_max_growth',
    'mediator.v_vs_max_delta_late',
    'mediator.td_residual_late',
    'mediator.greedy_match_late',
    'mediator.learning_curve_auc',
    'mediator.learning_curve_auc_peak_truncated',
    'mediator.time_to_threshold',
    'mediator.return_at_25pct_steps',
    'mediator.plateau_slope_late',
    'mediator.q_gap_peak_truncated_late',
    'mediator.td_residual_peak_truncated_late',
    'mediator.greedy_match_peak_truncated_late',
)

# Mediators only (subset of _PC_VARIABLES) for the differential
# scan — we ask which trace-features distinguish solved vs
# unsolved BASELINES, not which env-side variables.
_MEDIATOR_PATHS: tuple[str, ...] = tuple(
    p for p in _PC_VARIABLES if p.startswith('mediator.')
) + ('jensen_gap',)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path', type=Path, default=_DEFAULT_RUNS,
    )
    _ = parser.add_argument(
        '--total-steps', type=int, default=200000,
    )
    _ = parser.add_argument(
        '--top-k', type=int, default=8,
        help='How many top-|g| mediators to surface PC neighbors for.',
    )
    _ = parser.add_argument(
        '--alpha', type=float, default=0.05,
    )
    args = parser.parse_args()
    runs_path = Path(args.runs_path)  # pyright: ignore[reportAny]
    total_steps = int(args.total_steps)  # pyright: ignore[reportAny]
    top_k = int(args.top_k)  # pyright: ignore[reportAny]
    alpha = float(args.alpha)  # pyright: ignore[reportAny]

    if not runs_path.exists():
        print(f'corpus not found: {runs_path}', file=sys.stderr)
        sys.exit(1)

    print('=' * 92)
    print(
        f'Mediator differential audit '
        f'(corpus total_steps={total_steps})'
    )
    print('=' * 92)

    # Classify envs on the baseline arm.
    all_runs = read_runrows(runs_path)
    ts_runs = [
        r for r in all_runs
        if r.measurements.get('total_steps') == total_steps
    ]
    baseline_runs = [r for r in ts_runs if r.arm_key == 'baseline']
    classifications = classify_envs(baseline_runs)

    solved = envs_in_class(classifications, 'solved')
    unsolved = envs_in_class(classifications, 'unsolved')
    print()
    print(
        f'  scope classes: solved={list(solved)!r}  '
        f'unsolved={list(unsolved)!r}'
    )

    if not solved or not unsolved:
        print('  need ≥1 env in each of solved + unsolved; aborting.')
        return

    # ============ Mediator differential on baseline cells ============
    # The honest unit of analysis is `env_mean` — each env
    # contributes one mean value; pool envs (5 unsolved + 6
    # solved). Power is thin but unconfounded.
    #
    # We also report `cell` mode for diagnostic context (showing
    # how reward-scale confounds the per-cell pool — MinAtar
    # values are 4-5 orders of magnitude larger than bsuite, so
    # cross-class pooling at the cell level just rediscovers env
    # identity, not failure-mode signatures).
    print()
    print('=' * 92)
    print(
        '[ENV-MEAN] Per-mediator Hedges\' g — '
        'env-level units, the honest cross-class comparison'
    )
    print('=' * 92)
    diffs = mediator_differential(
        baseline_runs, classifications,
        paths=_MEDIATOR_PATHS,
        class_a='unsolved', class_b='solved',
        aggregation='env_mean',
    )
    print(
        f'  {"path":<48} {"g":>8} {"unsolved_mean":>14} '
        f'{"solved_mean":>13} {"n_unsolved_envs":>16} '
        f'{"n_solved_envs":>14}'
    )
    print('-' * 117)
    for d in diffs:
        g_str = f'{d.g:+8.3f}' if d.g == d.g else '     nan'
        print(
            f'  {d.path:<48} {g_str:>8} '
            f'{d.mean_a:>14.3g} {d.mean_b:>13.3g} '
            f'{d.n_a:>16} {d.n_b:>14}'
        )

    print()
    print('=' * 92)
    print(
        '[CELL, RAW] for diagnostic only — confounded by env scale'
    )
    print('=' * 92)
    cell_diffs = mediator_differential(
        baseline_runs, classifications,
        paths=_MEDIATOR_PATHS,
        class_a='unsolved', class_b='solved',
        aggregation='cell',
    )
    print(
        f'  {"path":<48} {"g":>8} {"n_unsolved_cells":>17} '
        f'{"n_solved_cells":>15}'
    )
    print('-' * 90)
    for d in cell_diffs[:8]:
        g_str = f'{d.g:+8.3f}' if d.g == d.g else '     nan'
        print(
            f'  {d.path:<48} {g_str:>8} '
            f'{d.n_a:>17} {d.n_b:>15}'
        )

    # ============ PC adjacency on the full corpus ============
    print()
    print('=' * 92)
    print(
        f'Conservative-PC adjacency (alpha={alpha}, '
        f'stratify_by=env_name, max_conditioning=1)'
    )
    print('=' * 92)

    df = pl.read_parquet(runs_path).filter(
        pl.col('total_steps') == total_steps,
    ).with_columns(
        (pl.col('intervention_name') == 'ddqn').cast(pl.Int64).alias('arm_ddqn'),
    )
    pc_df = df.drop_nulls(subset=list(_PC_VARIABLES))
    for v in _PC_VARIABLES:
        if pc_df[v].dtype.is_float():
            pc_df = pc_df.filter(~pl.col(v).is_nan())
    if pc_df.height < 30:
        print(f'  too few rows ({pc_df.height}) for PC; aborting.')
        return
    print(f'  rows after NaN filter: {pc_df.height}')

    adjacency = discover_adjacency(
        pc_df, variables=list(_PC_VARIABLES),
        alpha=alpha, max_conditioning=1,
        stratify_by='env_name',
    )
    print(f'  edges discovered: {len(adjacency.edges)}')

    # ============ Top-K mediators × PC neighbors ============
    print()
    print('=' * 92)
    print(
        f'Top-{top_k} differential mediators × PC neighbors '
        '(the "where to head next" candidates)'
    )
    print('=' * 92)

    finite_diffs = [d for d in diffs if d.g == d.g][:top_k]
    if not finite_diffs:
        print('  no finite-g differentials; aborting.')
        return

    edges_by_node: dict[str, list[str]] = {}
    for edge in adjacency.edges:
        u, v = sorted(edge)
        edges_by_node.setdefault(u, []).append(v)
        edges_by_node.setdefault(v, []).append(u)

    for d in finite_diffs:
        neighbors = sorted(edges_by_node.get(d.path, []))
        # Hide self / outcome from candidate set — outcome is the
        # downstream effect, not a candidate to intervene on.
        candidates = [
            n for n in neighbors
            if n != d.path
            and not n.startswith('outcome.')
        ]
        print()
        print(
            f'  {d.path}  g={d.g:+.3f}  '
            f'(unsolved-mean={d.mean_a:.3g}, '
            f'solved-mean={d.mean_b:.3g})'
        )
        if not candidates:
            print(f'    no PC neighbors (excluding outcome.* and self)')
            continue
        for n in candidates:
            print(f'    → {n}')


if __name__ == '__main__':
    main()
