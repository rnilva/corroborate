"""PC discovery across 3 envs at γ=0.999 (FR, SI, Asterix).

Tests whether the arm→outcome separating sets are stable cross-env
or env-specific. If stable → universal mechanism. If env-specific
(e.g., Asterix separator differs from FR/SI) → structurally
distinct mechanism worthy of separate explanation.

Per-env panel: arm + (jens, repeat_rate, growth, entropy) + outcome.
Pooled JCI panel: same variables, stratified by env_name.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import experiments.findings.ddqn_three_conditions  # populate registries

from corroborate.graph.discovery import discover_adjacency


_VARS = (
    'arm',
    'jensen_gap',
    'policy_growth_fraction',
    'state_repeat_rate_within_episode_window64_late',
    'state_hash_entropy_late',
    'eval_best_burst_raw_mean',
)


_FILTERS = {
    'FR γ=0.999': (
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('total_steps') == 1000000)
        & (pl.col('eval_every') == 20000)  # the new 50-burst sweep
    ),
    'SI γ=0.999': (
        (pl.col('env_name') == 'SpaceInvaders-MinAtar')
        & (pl.col('gamma') == 0.999)
    ),
    'Asterix γ=0.999': (
        (pl.col('env_name') == 'Asterix-MinAtar')
        & (pl.col('gamma') == 0.999)
    ),
}


def _prepare(df: pl.DataFrame, scope: pl.Expr) -> pl.DataFrame:
    cells = df.filter(scope).with_columns(
        (pl.col('arm_key') != 'baseline').cast(pl.Float64).alias('arm'),
    )
    # Drop rows missing any required var
    for v in _VARS[1:]:  # skip 'arm' (already coerced)
        cells = cells.filter(pl.col(v).is_finite())
    return cells


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn_three_conditions.parquet')
    print(f'cache rows: {len(df)}')
    print()

    cells_by_env: dict[str, pl.DataFrame] = {}
    for env_label, scope in _FILTERS.items():
        cells = _prepare(df, scope)
        cells_by_env[env_label] = cells
        print(f'{env_label}: {len(cells)} cells (vanilla {(cells["arm"]==0.0).sum()}, DDQN {(cells["arm"]==1.0).sum()})')
    print()

    print('=== Per-env PC discovery (max_conditioning=2, α=0.05) ===')
    for env_label, cells in cells_by_env.items():
        if len(cells) < 20:
            print(f'\n  {env_label}: TOO FEW CELLS ({len(cells)}); skipping')
            continue
        print(f'\n  --- {env_label} (n={len(cells)}) ---')
        adj = discover_adjacency(
            cells, variables=_VARS, max_conditioning=2, alpha=0.05,
        )
        print(f'  Surviving edges ({len(adj.edges)}):')
        for e in sorted(adj.edges, key=lambda fs: tuple(sorted(fs))):
            a, b = sorted(e)
            sep = adj.separating_sets.get(e, frozenset())
            sep_str = ' | '.join('{' + ', '.join(sorted(s)) + '}' for s in sep) if sep else 'MARGINAL'
            print(f'    {a:50s} ↔ {b:35s}  sep: {sep_str}')
        # Specifically: arm ↔ outcome status
        arm_outcome = frozenset({'arm', 'eval_best_burst_raw_mean'})
        if arm_outcome in adj.edges:
            sep = adj.separating_sets.get(arm_outcome, frozenset())
            print(f'  arm ↔ outcome edge: SURVIVED (sep sets tested but did not separate)')
        else:
            sep = adj.separating_sets.get(arm_outcome, frozenset())
            print(f'  arm ↔ outcome edge: REMOVED by separators:')
            for s in sep:
                print(f'    sep = {sorted(s)}')

    print()
    print('=== Pooled JCI panel (stratify_by=env_name, n=180) ===')
    pooled = df.filter(
        ((pl.col('env_name') == 'FourRooms-misc') & (pl.col('gamma')==0.999) & (pl.col('total_steps')==1000000) & (pl.col('eval_every')==20000))
        | ((pl.col('env_name') == 'SpaceInvaders-MinAtar') & (pl.col('gamma')==0.999))
        | ((pl.col('env_name') == 'Asterix-MinAtar') & (pl.col('gamma')==0.999))
    ).with_columns(
        (pl.col('arm_key') != 'baseline').cast(pl.Float64).alias('arm'),
    )
    for v in _VARS[1:]:
        pooled = pooled.filter(pl.col(v).is_finite())
    print(f'Pooled n: {len(pooled)}')
    adj_pool = discover_adjacency(
        pooled, variables=_VARS, max_conditioning=2, alpha=0.05,
        stratify_by='env_name',
    )
    print(f'Pooled JCI edges ({len(adj_pool.edges)}):')
    for e in sorted(adj_pool.edges, key=lambda fs: tuple(sorted(fs))):
        a, b = sorted(e)
        sep = adj_pool.separating_sets.get(e, frozenset())
        sep_str = ' | '.join('{' + ', '.join(sorted(s)) + '}' for s in sep) if sep else 'MARGINAL'
        print(f'  {a:50s} ↔ {b:35s}  sep: {sep_str}')
    arm_outcome = frozenset({'arm', 'eval_best_burst_raw_mean'})
    if arm_outcome in adj_pool.edges:
        print('  arm ↔ outcome edge: SURVIVED in pooled JCI')
    else:
        sep = adj_pool.separating_sets.get(arm_outcome, frozenset())
        print('  arm ↔ outcome edge: REMOVED in pooled JCI by:')
        for s in sep:
            print(f'    sep = {sorted(s)}')


if __name__ == '__main__':
    main()
