"""§6 acceptance — per-env PC on the DDQN corpus.

Reads `runs.parquet`, partitions by `env_name`, and runs
`discover_adjacency` (no JCI) over the 5 mechanism+outcome
variables plus `arm_ddqn` within each env. Reports the surviving
adjacency per env.

PAPER §6.1's framing: pooled JCI-PC tests for *uniform* edges
across strata. When pooled returns "no edge" (§4's verdict for
arm_ddqn↔outcome.*), per-env PC asks the per-stratum question:
"in *this* env, is there a within-env edge?" The current corpus
only tracks one mechanism feature (`mechanism.jensen_gap`), so
this smoke is the *thin* form of §6 — it cannot reproduce the
three-regime mediator taxonomy (TD-convergence / action-margin /
stay-greedy) which needs the full 8-mediator set computed from
per-step traces.

The thin §6 still answers a useful question: in which envs does
ANY of the mechanism / outcome variables survive a within-env edge
to `arm_ddqn`? That tells us where the bridge has within-env
signal vs where it's null even before mediator decomposition.

Run: `JAX_PLATFORMS=cpu uv run python experiments/smoke_per_env_pc_dqn.py`."""
from __future__ import annotations

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from pathlib import Path

import polars as pl

from corroborate.causal_discovery import discover_adjacency


_RUNS_PATH = Path(__file__).parent / 'data' / 'ddqn' / 'runs.parquet'


_VARIABLES: list[str] = [
    'arm_ddqn',
    'mechanism.jensen_gap',
    'outcome.late_window_mean',
    'outcome.eval_final_mean',
    'outcome.eval_best_burst_mean',
    'outcome.eval_best_burst_step',
]


def _prepare_dataframe() -> pl.DataFrame:
    df = pl.read_parquet(_RUNS_PATH)
    df = df.with_columns(
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    df = df.drop_nulls(subset=_VARIABLES)
    df = df.filter(
        ~pl.any_horizontal(
            [pl.col(v).is_nan() for v in _VARIABLES if df[v].dtype.is_float()],
        ),
    )
    return df


def main() -> None:
    df = _prepare_dataframe()
    n_obs_total = df.height
    envs = sorted(df['env_name'].unique().to_list())
    print(f'corpus: {n_obs_total} cells × {len(envs)} envs')
    print(f'PC variables: {_VARIABLES}')
    print(f'partitioning by: env_name (no JCI; per-env subsets)')
    print(f'α=0.05, max_conditioning=1')
    print()

    summary: list[tuple[str, int, frozenset[frozenset[str]]]] = []
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        n_env = env_df.height
        # Skip envs with constant arm_ddqn (e.g. one arm dropped)
        # or constant outcome — Spearman is undefined.
        if n_env < 5 or env_df['arm_ddqn'].n_unique() < 2:
            print(f'─ {env:<28} n={n_env:>3}  SKIP (insufficient variation)')
            continue
        # Constant-outcome envs (e.g. Freeway, MountainCar in §6.2)
        # produce all-NaN CI tests; the algorithm leaves no edges.
        constant_cols = [
            v for v in _VARIABLES
            if env_df[v].dtype.is_float() and float(env_df[v].std() or 0.0) == 0.0
        ]
        if constant_cols:
            print(f'─ {env:<28} n={n_env:>3}  SKIP '
                  f'(constant: {", ".join(constant_cols)})')
            continue
        adj = discover_adjacency(
            env_df, variables=_VARIABLES,
            alpha=0.05, max_conditioning=1,
        )
        summary.append((env, n_env, adj.edges))
        print(f'─ {env:<28} n={n_env:>3}  {len(adj.edges)} edges')
        for edge in sorted(
            (tuple(sorted(e)) for e in adj.edges),
        ):
            print(f'    {edge[0]:<32} ─ {edge[1]}')

    print()
    print('=' * 72)
    print('§6 thin summary — arm_ddqn within-env neighbours')
    print('=' * 72)
    print()
    arm_envs: list[tuple[str, list[str]]] = []
    for env, _, edges in summary:
        neighbours = sorted(
            [next(iter(e - {'arm_ddqn'})) for e in edges if 'arm_ddqn' in e],
        )
        if neighbours:
            arm_envs.append((env, neighbours))

    if arm_envs:
        print(f'{len(arm_envs)} env(s) with within-env arm_ddqn edge:')
        for env, neighbours in arm_envs:
            print(f'  {env:<28} → {", ".join(neighbours)}')
    else:
        print('  no env shows a within-env edge from arm_ddqn at α=0.05')
        print('  (consistent with §4: pooled and per-env both null on the')
        print('   thin variable set; richer mediator features needed)')


if __name__ == '__main__':
    main()
