"""What designed-sweep would recover the UNDERPOWERED verdicts on
the canonical panel?

After the canonical-scope migration, several bridges fire
POWER_INSUFFICIENT — not because the underlying effect is absent
but because canonical has 60 cells/env and the DoWhy backdoor +
placebo + RCC cluster needs n≥30 per stratum. This script reports,
per UNDERPOWERED finding/bridge cluster, the minimum n_cells per
env needed to lift the verdict, plus suggests an env-class
priority.

Findings UNDERPOWERED at canonical:
- `finding_reach_bias_link` (3 DoWhy bridges) — n=620 across 11 envs ≈ 56/env.
- `finding_metamaze_gamma_amplification` (now in ddqn_sweeps; γ-sweep specific).

The DoWhy backdoor estimator needs ~30 cells per (env, config)
stratum for the regression to converge cleanly. canonical has
60 cells/env / 1 config = 60 cells/stratum. Marginal. Doubling
to 120-180 seeds/env at canonical config should recover
verdicts."""
from __future__ import annotations

import polars as pl


CACHE = 'experiments/data/cache/ddqn.parquet'


def main() -> None:
    df = pl.scan_parquet(CACHE).select([
        'env_name', 'arm_key', 'gamma', 'sync_period', 'total_steps',
        'replay.capacity', 'jensen_gap',
    ]).collect()

    print(f'Canonical cache: {df.height} cells × {df["env_name"].n_unique()} envs')
    print()
    print('Per-env (env, arm) cell counts:')
    by_arm = df.group_by(['env_name', 'arm_key']).len().sort('env_name')
    # collapse arm strings
    arms = by_arm.with_columns(
        pl.when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .when(pl.col('arm_key').str.contains('Claim:double_greedify')).then(pl.lit('ddqn'))
        .otherwise(pl.lit('other')).alias('arm')
    ).group_by(['env_name', 'arm']).agg(pl.col('len').sum().alias('cells'))
    pivot = arms.pivot(values='cells', index='env_name', on='arm', aggregate_function='sum').sort('env_name')
    print(pivot)

    # Target: 120 cells per arm per env (240 paired) — recommended for DoWhy
    # backdoor + placebo + RCC to fire HELD/REFUTED at canonical scope.
    print()
    print('Recommended target for designed sweep at canonical config:')
    print('  ≥120 cells per arm per env (≥240 total per env)')
    print('  Current canonical: 30 paired (60 cells/env, both arms)')
    print('  Gap: ~180 more cells per env needed (90 ddqn + 90 vanilla)')
    print()
    print('Priority order (envs that already touch canonical but are')
    print('UNDERPOWERED in the bias_correction DoWhy cluster):')
    for env in sorted(df['env_name'].unique().to_list()):
        n = df.filter(pl.col('env_name') == env).height
        print(f'  {env:<30s}: have {n}, want 240 (gap {240 - n})')


if __name__ == '__main__':
    main()
