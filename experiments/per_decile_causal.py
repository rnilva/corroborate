"""Per-decile DoWhy backdoor: do(arm=ddqn) → mc_return | covariates.

Pearl rung-2 take on the per-burst DDQN graph. Unrolls the
paired-delta-per-burst dataset to one row per (cell, burst, arm),
then for each training-time decile runs DoWhy backdoor:

  treatment: arm (binary: 1=ddqn, 0=vanilla_dqn)
  outcome:   mc_return at that burst
  confounders: env_id (int factor), replay.capacity, optimizer.inner.lr,
               sync_period, n_actions, log_obs_dim, log_horizon
  graph: arm + every confounder → outcome (no path through other vars)

Output: per-decile ATE, refutation drift, sample size.

The cell-mean Pearl rung-2 bridge `state_coverage_kl_causes_outcome`
already establishes the framework can do backdoor adjustment.
This applies the same machinery to DDQN itself, stratified by
training time, to expose the early-positive / late-negative
trajectory at rung 2."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.analyses.dowhy import (
    backdoor_ate as backdoor_ate_analysis,
)


_BURST = Path('experiments/data/ddqn_universal/paired_delta_per_burst.parquet')


def _unroll(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot the paired-delta-per-burst rows to long arm format —
    one row per (cell, burst, arm). Each pair becomes 2 rows."""
    keep_cols = [
        'corpus', 'env_name', 'total_steps', 'seed', 'burst_index',
        'burst_frac', 'replay.capacity', 'replay.batch_size',
        'optimizer.inner.lr', 'sync_period', 'n_actions',
        'log_obs_dim', 'log_horizon',
    ]
    vanilla = df.select([
        *keep_cols,
        pl.lit(0).alias('arm'),
        pl.col('mc_vanilla').alias('mc_return'),
        pl.col('bias_vanilla').alias('bias'),
    ])
    ddqn = df.select([
        *keep_cols,
        pl.lit(1).alias('arm'),
        pl.col('mc_ddqn').alias('mc_return'),
        pl.col('bias_ddqn').alias('bias'),
    ])
    return pl.concat([vanilla, ddqn], how='vertical_relaxed')


def _env_to_int(df: pl.DataFrame) -> pl.DataFrame:
    """Stable integer factor for env_name."""
    envs = sorted(df['env_name'].unique().to_list())
    env_to_idx = {e: i for i, e in enumerate(envs)}
    return df.with_columns(
        env_id=pl.col('env_name').map_elements(
            lambda e: env_to_idx[e], return_dtype=pl.Int64,
        ),
    )


_CONFOUNDERS: tuple[str, ...] = (
    'env_id', 'replay.capacity', 'replay.batch_size',
    'optimizer.inner.lr', 'sync_period',
    'n_actions', 'log_obs_dim', 'log_horizon',
)


def _backdoor_dag() -> list[tuple[str, str]]:
    return [
        *[(c, 'mc_return') for c in _CONFOUNDERS],
        *[(c, 'arm') for c in _CONFOUNDERS],
        ('arm', 'mc_return'),
    ]


def _cells_from_df(sub: pl.DataFrame) -> list[dict[str, float]]:
    """Project the long-format DataFrame to a list of cell dicts
    that the `analyses/dowhy.py:backdoor_ate` `@analysis` consumes
    (one dict per row, with treatment / outcome / confounders)."""
    cols = ['arm', 'mc_return', *_CONFOUNDERS]
    sub_typed = sub.with_columns(
        [pl.col(c).cast(pl.Float64) for c in _CONFOUNDERS]
        + [pl.col('arm').cast(pl.Float64),
           pl.col('mc_return').cast(pl.Float64)],
    )
    return [
        {c: float(row[c]) for c in cols}
        for row in sub_typed.iter_rows(named=True)
    ]


def main() -> None:
    df = pl.read_parquet(str(_BURST))
    long = _unroll(df)
    long = _env_to_int(long)
    long = long.with_columns(
        decile=(pl.col('burst_frac') * 10).cast(pl.Int64).clip(0, 9),
    )
    print(f'long-format rows: {len(long)}')
    n_envs = long['env_name'].n_unique()
    n_arms = long['arm'].n_unique()
    print(f'envs: {n_envs}, arms: {n_arms}')
    print()

    dag = _backdoor_dag()

    def run_block(label: str, subset: pl.DataFrame) -> None:
        print()
        print(f'### {label} (n_rows={len(subset)}) ###')
        if len(subset) == 0:
            return
        print(f'{"decile":<7} {"n":>6} {"ate":>10} {"verdict":<22}')
        print(f'{"-"*7} {"-"*6} {"-"*10} {"-"*22}')
        for d in range(10):
            sub = subset.filter(pl.col('decile') == d)
            if len(sub) < 50:
                continue
            cells = _cells_from_df(sub)
            try:
                res = backdoor_ate_analysis.fn(
                    cells=cells,
                    treatment='arm', outcome='mc_return',
                    dag=dag,
                )
                # expected_sign=+1, threshold=0.0 — substrate-side
                # verdict mapping (was on the gone Bridge[R] factory).
                ate_held = (
                    res.identified and res.ate > 0.0
                )
                verdict_str = 'held' if ate_held else 'no_effect'
                print(f'{d:<7} {len(sub):>6} {res.ate:>+10.4f} '
                      f'{verdict_str:<22}')
            except (ValueError, KeyError, TypeError) as e:
                print(f'{d:<7} {len(sub):>6} ERROR: {e!s:<40.40s}')

    # 1. Full universe (baseline reading).
    run_block('FULL UNIVERSE', long)
    # 2. Long-horizon only (MinAtar 1M).
    run_block('LONG-HORIZON (total_steps >= 1M)',
              long.filter(pl.col('total_steps') >= 1000000))
    # 3. Pixel envs at long horizon (cleanest setting for the
    #    per-burst attenuation pattern).
    run_block(
        'PIXEL+LONG (log_obs_dim >= 5 AND total_steps >= 1M)',
        long.filter(
            (pl.col('log_obs_dim') >= 5.0)
            & (pl.col('total_steps') >= 1000000),
        ),
    )
    # 4. SpaceInvaders only at long horizon — where the
    #    attenuation bridge fired HELD with g=−0.42.
    run_block(
        'SpaceInvaders-MinAtar AT 1M',
        long.filter(
            (pl.col('env_name') == 'SpaceInvaders-MinAtar')
            & (pl.col('total_steps') >= 1000000),
        ),
    )


if __name__ == '__main__':
    main()
