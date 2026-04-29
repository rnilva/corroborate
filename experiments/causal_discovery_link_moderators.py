"""Causal discovery on the per-(env, burst) link-moderator panel.

After the meta-regression deconfounding (5c447f2 → 4661c24)
identified four candidate moderators of g_link
(bootstrap_fraction, log_horizon, empirical_reward_density,
log_obs_dim), this script applies the framework's
causal-discovery primitives:

  1. Conservative-PC adjacency (`discover_adjacency`) at depth ≤ 2
     over (covariates ∪ {g_link}). Identifies which moderators
     remain edge-adjacent to g_link after conditioning on
     candidate separators.

  2. DoWhy backdoor + refutation triple on the strongest
     surviving edge: causal ATE estimate at rung-2-conditional-
     on-DAG.

These are **observational** primitives — the env-features are
predetermined by the env, not interventionally manipulated. The
discovery's value is reducing the ambiguity about which
moderators are direct-link contributors vs which are
correlated-but-screened-off.

Usage:
  uv run python experiments/causal_discovery_link_moderators.py
  uv run python experiments/causal_discovery_link_moderators.py \
    --corpus ddqn_effective_cohort --total-steps 200000
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.bridges_dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.causal_discovery import discover_adjacency
from corroborate.rl.env_catalogue import get
from corroborate.statistics import hedges_g_paired


# Reuse the helpers from analyze_per_burst_meta_regression.
import sys
sys.path.insert(0, 'experiments')
from analyze_per_burst_meta_regression import (  # type: ignore[import-untyped]
    _empirical_reward_features, _load_arrays,
)


_REWARD_DENSITY_MAP: dict[str, float] = {
    'terminal_only': 0.0,
    'shaped': 1.0,
    'event_triggered': 1.0,
    'per_step': 2.0,
}


def _build_panel(corpus: str, total_steps: int, *, include_env: bool = False) -> pl.DataFrame:
    runs_path = Path('experiments/data') / corpus / 'runs.parquet'
    if not runs_path.exists():
        runs_path = (
            Path('experiments/data') / corpus / 'runs_with_mediators.parquet'
        )
    df = pl.read_parquet(str(runs_path)).filter(
        pl.col('total_steps') == total_steps,
    )
    envs = sorted(df['env_name'].unique().to_list())
    rows: list[dict[str, float]] = []
    for env in envs:
        spec = get(env)
        n_a = spec.n_actions
        obs_n = 1
        for d in spec.observation_shape:
            obs_n *= int(d)
        horizon = float(spec.horizon) if spec.horizon else 1000.0
        empirical = _empirical_reward_features(corpus, env)
        if empirical is None:
            continue
        nonzero_reward_frac, bootstrap_fraction = empirical
        arrays = _load_arrays(corpus, env)
        if arrays is None:
            continue
        delta_bias, delta_ret = arrays
        n_pairs, n_bursts = delta_ret.shape
        for b in range(n_bursts):
            dr = list(map(float, delta_ret[:, b].tolist()))
            g, se = hedges_g_paired(dr)
            if not (
                isinstance(g, float) and math.isfinite(g)
                and isinstance(se, float) and math.isfinite(se) and se > 0.0
            ):
                continue
            row: dict[str, object] = {
                'g_link': float(g),
                'log_action_dim': math.log(max(n_a, 2)),
                'log_obs_dim': math.log(max(obs_n, 1)),
                'log_horizon': math.log(max(horizon, 1.0)),
                'empirical_reward_density': float(nonzero_reward_frac),
                'bootstrap_fraction': float(bootstrap_fraction),
                'burst_index': float(b),
                'mean_dbias': float(delta_bias[:, b].mean()),
            }
            if include_env:
                row['env_name'] = env
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='ddqn')
    parser.add_argument('--total-steps', type=int, default=200_000)
    args = parser.parse_args()
    corpus: str = args.corpus
    total_steps: int = args.total_steps

    print('=' * 100)
    print(f'Causal discovery on per-(env, burst) link moderator panel '
          f'[corpus={corpus}, total_steps={total_steps}]')
    print('=' * 100)
    panel = _build_panel(corpus, total_steps, include_env=True)
    print(f'  n_strata={panel.height}  n_features={len(panel.columns)}')

    variables = (
        'g_link',
        'bootstrap_fraction',
        'log_horizon',
        'log_obs_dim',
        'empirical_reward_density',
        'log_action_dim',
        'mean_dbias',
    )

    # Stage 1a: Conservative-PC adjacency at depth ≤ 2 — no JCI.
    print()
    print('Stage 1a — Conservative-PC adjacency (depth ≤ 2, no JCI)')
    adj = discover_adjacency(
        panel, variables=variables,
        alpha=0.05, max_conditioning=2,
    )
    print()
    print('Surviving edges:')
    for edge in sorted(
        adj.edges, key=lambda e: tuple(sorted(e)),
    ):
        a, b = sorted(edge)
        print(f'  {a:<28} ⟷ {b}')
    print()
    print('Edges removed (with separating sets):')
    for edge, seps in sorted(
        adj.separating_sets.items(),
        key=lambda kv: tuple(sorted(kv[0])),
    ):
        a, b = sorted(edge)
        sep_str = ', '.join(
            '{}'.format(', '.join(sorted(s))) if s else '∅'
            for s in sorted(seps, key=len)
        )
        print(f'  {a:<28} ⊥ {b:<28}  | {{ {sep_str} }}')

    # Stage 1b: PC with JCI — within-env-stratified CI tests pooled
    # via Fisher z. Addresses the within-env-correlation concern
    # (149 strata aren't iid; bursts within an env share cells).
    print()
    print('Stage 1b — Conservative-PC with JCI (stratify_by=env_name)')
    adj_jci = discover_adjacency(
        panel, variables=variables,
        alpha=0.05, max_conditioning=2,
        stratify_by='env_name',
    )
    print()
    print('Surviving edges (JCI):')
    for edge in sorted(adj_jci.edges, key=lambda e: tuple(sorted(e))):
        a, b = sorted(edge)
        print(f'  {a:<28} ⟷ {b}')

    # Stage 2: DoWhy backdoor on the surviving direct-edge candidate
    # for g_link. Use bootstrap_fraction → g_link with the rest as
    # confounders (those that are NOT screened off from g_link).
    g_link_neighbors = sorted(
        nb for edge in adj.edges for nb in edge
        if 'g_link' in edge and nb != 'g_link'
    )
    print()
    print(f'g_link neighbors: {g_link_neighbors}')

    if not g_link_neighbors:
        print('  no surviving edges from g_link; skipping DoWhy.')
        return

    treatment = 'bootstrap_fraction' if (
        'bootstrap_fraction' in g_link_neighbors
    ) else g_link_neighbors[0]
    print()
    print(f'Stage 2 — DoWhy backdoor: {treatment} → g_link | confounders')

    confounders = [v for v in g_link_neighbors if v != treatment]
    dag: list[tuple[str, str]] = [
        *((c, treatment) for c in confounders),
        *((c, 'g_link') for c in confounders),
        (treatment, 'g_link'),
    ]
    record: Mapping[str, np.ndarray] = {
        col: np.asarray(panel[col].to_list(), dtype=np.float64)
        for col in panel.columns
        if col != 'env_name'  # exclude string column from numeric record
    }
    triple = (
        ('backdoor_ate', backdoor_ate(
            treatment, 'g_link', graph=dag,
            expected_sign=+1, threshold=0.05,
        )),
        ('placebo', placebo_refutation(
            treatment, 'g_link', graph=dag, tolerance=0.1,
        )),
        ('random_common_cause', random_common_cause_refutation(
            treatment, 'g_link', graph=dag, tolerance=0.1,
        )),
    )
    print(f'  DAG: {len(confounders)} confounders → {treatment} → g_link')
    for label, bridge in triple:
        r = bridge(record)  # type: ignore[arg-type]
        keystat = ''
        if 'ate' in r.stats:
            ate = r.stats['ate']
            keystat = (
                f'ATE={float(ate):+.4f}'
                if isinstance(ate, (int, float)) else 'ATE=?'
            )
        elif 'placebo_ate' in r.stats:
            p = r.stats['placebo_ate']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'placebo={float(p):+.4f} real={float(real):+.4f}'
                if isinstance(p, (int, float))
                and isinstance(real, (int, float))
                else 'placebo=?'
            )
        elif 'drift' in r.stats:
            d = r.stats['drift']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'drift={float(d):.4f} real={float(real):+.4f}'
                if isinstance(d, (int, float))
                and isinstance(real, (int, float))
                else 'drift=?'
            )
        print(f'  {label:<22} verdict={r.verdict.value:<22} {keystat}')


if __name__ == '__main__':
    main()
