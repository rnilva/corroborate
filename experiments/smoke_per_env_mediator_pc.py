"""§5 + rich §6 — within-env Pearson + per-env PC on the
12-variable mediator-augmented corpus.

Reads `runs_with_mediators.parquet` (produced by
`compute_mediators.py`). Reproduces the structural findings from
PAPER §5.1, §5.2, §6.2:

- §5.1 — within-env Pearson against `outcome.eval_final_mean`,
  pooled across arms. Highlights mediators with |r| > 0.5.
- §5.2 — same Pearson within each arm separately, surfacing
  arm-conditional mediators (e.g. CartPole's vanilla→DDQN sign
  flip on q_gap_late).
- §6.2 — per-env PC over the full 12-variable set (arm + mechanism
  + 8 mediators + 2 outcomes), reporting `outcome.eval_final_mean`-
  neighbours per env. The frequency table over neighbours surfaces
  the three operational regimes (TD-convergence / action-margin /
  stay-greedy) when corpus structure cooperates.

Run: `JAX_PLATFORMS=cpu uv run python
experiments/smoke_per_env_mediator_pc.py`."""
from __future__ import annotations

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr  # type: ignore[reportMissingTypeStubs]

from corroborate.causal_discovery import discover_adjacency


_ENRICHED_PATH = (
    Path(__file__).parent / 'data' / 'ddqn' / 'runs_with_mediators.parquet'
)


_MEDIATORS: tuple[str, ...] = (
    'mediator.q_gap_late',
    'mediator.q_gap_growth',
    'mediator.q_max_growth',
    'mediator.v_vs_max_delta_late',
    'mediator.td_residual_late',
    'mediator.greedy_match_late',
    'mediator.fill_ratio_late',
    'mediator.epsilon_late',
)


_OUTCOME = 'outcome.eval_final_mean'

# 12-var set for per-env PC: arm + mechanism + 8 mediators + 2 outcomes.
_PC_VARIABLES: tuple[str, ...] = (
    'arm_ddqn',
    'mechanism.jensen_gap',
    *_MEDIATORS,
    'outcome.eval_final_mean',
    'outcome.late_window_mean',
)


def _load() -> pl.DataFrame:
    if not _ENRICHED_PATH.exists():
        raise SystemExit(
            f'{_ENRICHED_PATH} not found. Run compute_mediators.py first.',
        )
    df = pl.read_parquet(_ENRICHED_PATH)
    df = df.with_columns(
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    return df


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r + p, NaN-safe. Drops rows where either is NaN.
    Returns (NaN, NaN) when post-filter n < 3 or any side constant."""
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return float('nan'), float('nan')
    xs = x[finite]
    ys = y[finite]
    if float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return float('nan'), float('nan')
    r, p = pearsonr(xs, ys)
    return float(r), float(p)


def _print_within_env_pearson(df: pl.DataFrame) -> None:
    envs = sorted(df['env_name'].unique().to_list())
    print('=' * 84)
    print('§5.1 — Within-env Pearson r vs outcome.eval_final_mean (pooled across arms)')
    print('=' * 84)
    print(f'{"env":<28} ' + ' '.join(
        f'{m.removeprefix("mediator."):<22}' for m in _MEDIATORS
    ))
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 3:
            continue
        outcome = np.asarray(env_df[_OUTCOME].to_list(), dtype=np.float64)
        cells: list[str] = []
        for m in _MEDIATORS:
            mediator_arr = np.asarray(env_df[m].to_list(), dtype=np.float64)
            r, _ = _pearson(mediator_arr, outcome)
            if np.isnan(r):
                cells.append(f'{"nan":<22}')
            else:
                marker = '*' if abs(r) > 0.5 else ' '
                cells.append(f'{r:>+.2f}{marker}                 '[:22])
        print(f'{env:<28} ' + ' '.join(cells))


def _print_arm_conditional_pearson(df: pl.DataFrame) -> None:
    print()
    print('=' * 84)
    print('§5.2 — Within-env Pearson by arm (mediators with |r| > 0.4 in either arm)')
    print('=' * 84)
    envs = sorted(df['env_name'].unique().to_list())
    arms = ('vanilla_dqn', 'ddqn')
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 6:
            continue
        outcome_pooled = np.asarray(env_df[_OUTCOME].to_list(), dtype=np.float64)
        # Skip envs with constant outcome.
        finite_pool = np.isfinite(outcome_pooled)
        if int(np.sum(finite_pool)) < 3:
            continue
        if float(np.std(outcome_pooled[finite_pool])) == 0.0:
            continue
        any_strong = False
        env_lines: list[str] = []
        for m in _MEDIATORS:
            row_cells: list[str] = []
            kept = False
            for arm in arms:
                arm_df = env_df.filter(pl.col('intervention_name') == arm)
                if arm_df.height < 3:
                    row_cells.append(f'{arm}: nan')
                    continue
                ax = np.asarray(arm_df[m].to_list(), dtype=np.float64)
                ay = np.asarray(arm_df[_OUTCOME].to_list(), dtype=np.float64)
                r, _ = _pearson(ax, ay)
                if not np.isnan(r) and abs(r) > 0.4:
                    kept = True
                row_cells.append(f'{arm}: {r:>+.2f}'
                                 if not np.isnan(r) else f'{arm}: nan')
            if kept:
                any_strong = True
                env_lines.append(
                    f'  {m.removeprefix("mediator."):<22}  '
                    + '  '.join(row_cells)
                )
        if any_strong:
            print(f'  {env}:')
            for line in env_lines:
                print(line)


def _per_env_pc(df: pl.DataFrame) -> list[tuple[str, frozenset[str]]]:
    """Run discover_adjacency over each env's subset, return list of
    (env, neighbours-of-outcome.eval_final_mean) pairs."""
    out: list[tuple[str, frozenset[str]]] = []
    envs = sorted(df['env_name'].unique().to_list())
    for env in envs:
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        # NaN-rows in any PC variable break the Spearman tests; drop.
        pc_df = env_df.drop_nulls(subset=list(_PC_VARIABLES))
        # Filter NaN floats too (drop_nulls only handles polars-null).
        for v in _PC_VARIABLES:
            if pc_df[v].dtype.is_float():
                pc_df = pc_df.filter(~pl.col(v).is_nan())
        if pc_df.height < 5 or pc_df['arm_ddqn'].n_unique() < 2:
            continue
        constant_cols = [
            v for v in _PC_VARIABLES
            if pc_df[v].dtype.is_float()
            and float(pc_df[v].std() or 0.0) == 0.0
        ]
        if constant_cols:
            continue
        adj = discover_adjacency(
            pc_df, variables=list(_PC_VARIABLES),
            alpha=0.05, max_conditioning=1,
        )
        outcome_neighbours = frozenset(
            v for edge in adj.edges if _OUTCOME in edge
            for v in edge if v != _OUTCOME
        )
        out.append((env, outcome_neighbours))
    return out


def _print_per_env_pc(
    pairs: list[tuple[str, frozenset[str]]],
) -> None:
    print()
    print('=' * 84)
    print(f'§6.2 — Per-env PC over the full 12-variable set; '
          f'{_OUTCOME}-neighbours')
    print('=' * 84)
    surviving = [(env, ns) for env, ns in pairs if ns]
    null = [env for env, ns in pairs if not ns]
    print(f'{len(surviving)} of {len(pairs)} testable envs surface ≥1 neighbour')
    print()
    for env, ns in pairs:
        ns_str = (
            ', '.join(sorted(n.removeprefix('mediator.') for n in ns))
            if ns else '(none)'
        )
        print(f'  {env:<28} → {ns_str}')

    print()
    print('Mediator-frequency table (across surviving envs):')
    counter: Counter[str] = Counter()
    for _, ns in surviving:
        for n in ns:
            counter[n] += 1
    for n, count in counter.most_common():
        label = n.removeprefix('mediator.') if n.startswith('mediator.') else n
        print(f'  {label:<28} surviving in {count} env(s)')

    if null:
        print()
        print(f'Null envs (no surviving {_OUTCOME}-neighbour): '
              f'{len(null)}')
        for env in null:
            print(f'  {env}')


def main() -> None:
    df = _load()
    n_obs = df.height
    n_envs = df['env_name'].n_unique()
    print(f'corpus: {n_obs} cells × {n_envs} envs '
          f'(loaded {_ENRICHED_PATH.name})')
    n_with_mediators = df.filter(
        pl.col('mediator.q_gap_late').is_not_null()
        & ~pl.col('mediator.q_gap_late').is_nan()
    ).height
    print(f'cells with non-NaN mediators: {n_with_mediators}')
    print()

    _print_within_env_pearson(df)
    _print_arm_conditional_pearson(df)
    pairs = _per_env_pc(df)
    _print_per_env_pc(pairs)


if __name__ == '__main__':
    main()
