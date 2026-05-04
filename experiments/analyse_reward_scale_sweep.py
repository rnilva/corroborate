"""Analyse the reward_scale_sweep — Pearl-rung-2 causal probe of
the env-level `log_mc_variance → g_link` attenuator.

Usage:
  uv run python experiments/analyse_reward_scale_sweep.py

Reads `experiments/data/reward_scale_sweep/{runs,traces}.parquet`,
computes paired Hedges' g per (env, reward_scale, burst), and
regresses g_link on `log(reward_scale)` to test whether scaling
reward (and thus mc_variance, by `scale²`) causally moves DDQN's
link benefit.

Predictions, framed against the observational finding:

  Observational: env-level β(log_mc_variance) on g_link = −0.021
                 (p=0.018), high-spread envs see less DDQN benefit.

  Causal-probe:  β(log(reward_scale)) on g_link should be
                 negative and significant — scaling reward UP
                 should attenuate DDQN; scaling DOWN should
                 enhance it.

If reward_scale ⊥ g_link in expectation (β ≈ 0), the
observational signal is a confound — log_mc_variance is just a
proxy for something correlated with env identity, not the causal
moderator.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import statsmodels.api as sm

from corroborate.analyses.paired_g_per_burst import (
    DEFAULT_PER_BURST_SOURCE, paired_g_per_burst,
)


def main() -> None:
    runs = pl.read_parquet(
        'experiments/data/reward_scale_sweep/runs.parquet',
        columns=['id', 'env_name', 'intervention_name', 'seed',
                 'reward_scale'],
    )
    # The merge step OOM'd on the 5.7GB of traces; concat the per-arm
    # tmp shards directly with column projection (just `id` and
    # `mc_return`) so we never materialize the full traces in memory.
    from pathlib import Path
    tmp_dir = Path('experiments/data/reward_scale_sweep/tmp')
    trace_shards = sorted(p for p in tmp_dir.iterdir()
                          if p.name.endswith('__traces.parquet'))
    print(f'Reading {len(trace_shards)} trace shards lazily...')
    traces = pl.concat(
        [pl.scan_parquet(str(p)).select(['id', 'mc_return']).collect()
         for p in trace_shards],
        how='diagonal_relaxed',
    )
    joined = traces.join(runs, on='id', how='inner').filter(
        pl.col('mc_return').is_not_null(),
    )
    print(f'Loaded {joined.shape[0]} cells (runs × non-null mc).')

    # Per-(env, reward_scale): build cells, run paired_g_per_burst.
    panel: list[dict[str, object]] = []
    for env_name in sorted(joined['env_name'].unique().to_list()):
        for rs in sorted(joined.filter(
            pl.col('env_name') == env_name,
        )['reward_scale'].unique().to_list()):
            sub = joined.filter(
                (pl.col('env_name') == env_name)
                & (pl.col('reward_scale') == rs),
            )
            cells = [
                {'env_name': r['env_name'],
                 'intervention_name': r['intervention_name'],
                 'seed': r['seed'],
                 'mc_return': np.asarray(r['mc_return'], dtype=np.float64)}
                for r in sub.iter_rows(named=True)
            ]
            res = paired_g_per_burst.fn(
                cells, treatment_arm='ddqn',
                baseline_arm='vanilla_dqn',
                pair_by=('seed',), source=DEFAULT_PER_BURST_SOURCE,
            )
            for s in res.strata:
                if s.n_pairs < 2 or math.isnan(s.g) or math.isnan(s.se) \
                        or s.se <= 0.0:
                    continue
                panel.append({
                    'env_name': env_name,
                    'reward_scale': float(rs),
                    'burst_index': s.burst_index,
                    'g_link': s.g,
                    'se_link': s.se,
                    'n_pairs': s.n_pairs,
                })

    if not panel:
        raise RuntimeError('Empty panel — no per-burst paired_g cells.')

    df = pl.DataFrame(panel).with_columns([
        pl.col('reward_scale').log().alias('log_reward_scale'),
    ])
    print(f'Panel: {df.shape}, envs: {df["env_name"].n_unique()}, '
          f'scales: {sorted(df["reward_scale"].unique().to_list())}')

    # Per-(env, scale) summary
    print()
    print('=== Per-(env, scale) g_link summary ===')
    summary = df.group_by(['env_name', 'reward_scale']).agg([
        pl.col('g_link').mean().alias('g_link_mean'),
        pl.col('g_link').std().alias('g_link_sd'),
        pl.len().alias('n_bursts'),
    ]).sort(['env_name', 'reward_scale'])
    print(summary)

    # Univariate causal test: g_link ~ log(reward_scale)
    print()
    print('=== Causal probe: g_link ~ log(reward_scale) ===')
    y = df['g_link'].to_numpy()
    w = 1.0 / df['se_link'].to_numpy() ** 2
    x = df['log_reward_scale'].to_numpy()
    X = sm.add_constant(x)
    fit = sm.WLS(y, X, weights=w).fit()
    print(f'  β(log_reward_scale): {fit.params[1]:+.4f} '
          f'(SE={fit.bse[1]:.4f}, p={fit.pvalues[1]:.4f})')
    print(f'  intercept: {fit.params[0]:+.4f}')
    print(f'  r² = {fit.rsquared:.3f}, N = {len(y)}')
    print()
    if fit.pvalues[1] < 0.05:
        sign = 'NEGATIVE' if fit.params[1] < 0 else 'POSITIVE'
        print(f'  → SIGNIFICANT {sign} effect: scaling reward '
              f'{"attenuates" if fit.params[1] < 0 else "enhances"} '
              f'DDQN benefit. log_mc_variance is causally moving g_link.')
    else:
        print('  → NULL: reward_scale does not move g_link. '
              'log_mc_variance is observational only — '
              'a proxy for env identity, not a causal moderator.')

    # Per-env decomposition
    print()
    print('=== Per-env: g_link ~ log(reward_scale) within each env ===')
    for env in sorted(df['env_name'].unique().to_list()):
        sub = df.filter(pl.col('env_name') == env)
        if sub['reward_scale'].n_unique() < 2:
            continue
        y_e = sub['g_link'].to_numpy()
        w_e = 1.0 / sub['se_link'].to_numpy() ** 2
        x_e = sub['log_reward_scale'].to_numpy()
        X_e = sm.add_constant(x_e)
        fit_e = sm.WLS(y_e, X_e, weights=w_e).fit()
        print(f'  {env:<20} β={fit_e.params[1]:+.4f} '
              f'p={fit_e.pvalues[1]:.4f}  N={len(y_e)}')

    df.write_parquet('/tmp/reward_scale_sweep_panel.parquet')
    print(f'\nSaved panel → /tmp/reward_scale_sweep_panel.parquet')


if __name__ == '__main__':
    main()
