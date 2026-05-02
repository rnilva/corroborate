"""Time-to-first-solve as the link metric on the DDQN corpus.

The headline DDQN finding (mechanism HELD ↛ link HELD) reads the
*steady-state* outcome: `eval_best_burst_mean` averaged
over the late-training distribution. That metric saturates at
the discounted-return ceiling for envs where both arms eventually
solve, hiding any *sample-efficiency* effect of DDQN.

This script tests a different link: among cells that solved their
env at all, does DDQN reach threshold *faster* than vanilla?

Method:
  - For each cell, take `eval_best_burst_step` as a proxy
    for "step at first crossing of solve threshold". Best-burst
    step is an upper bound on first-crossing step — for monotonic
    learners they coincide; for unstable cells the best crossing
    might be later than the first. Use it as a first-pass proxy;
    if interesting, redo with streaming-trace exact first-crossing.
  - Filter to (env, seed) pairs where BOTH ddqn and vanilla cells
    have `eval_best_burst_mean >= env_threshold` (both solved).
  - Per env: `paired_comparison_from_runs(treatment=ddqn,
    baseline=vanilla, outcome_path=eval_best_burst_step,
    pair_by=('seed',), predicted_direction='a_lt_b')`. DDQN should
    solve faster ⇒ smaller best-burst-step.
  - Random-effects pool over envs, stratified by solve-rate
    class (high ≥80%, mixed 30–80%, low <30%).

Honest scope:
  - Only 200k-step cells (50k cells have inflated NaN rates).
  - Threshold 'absent' envs excluded.
  - The proxy can lag the exact first-crossing for unstable cells;
    inflated symmetrically across both arms it shouldn't bias the
    paired g, but the variance is wider than the exact metric.

Usage:
  uv run python experiments/time_to_solve_ddqn.py
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_comparison_from_runs
from corroborate.rl.env_solve_thresholds import (
    SOLVE_THRESHOLDS, envs_with_threshold,
)
from corroborate.schema import RunRow
from corroborate.statistics import (
    PooledStats, random_effects_summary, random_effects_verdict,
)
from corroborate.verdict import Verdict


_RUNS = Path('experiments/data/ddqn/runs_with_mediators.parquet')
_PROXY_PATH = 'eval_best_burst_step'
_OUTCOME_PATH = 'eval_best_burst_mean'


def _load_solved_runs(env_name: str) -> tuple[list[RunRow], list[RunRow]]:
    """Load (ddqn, vanilla) RunRows for one env, filtered to cells
    that solved. Pre-filtered to 200k cells."""
    spec = SOLVE_THRESHOLDS[env_name]
    if spec.threshold is None:
        return [], []
    df = pl.read_parquet(_RUNS).filter(
        (pl.col('env_name') == env_name)
        & (pl.col('total_steps') == 200000)
        & (pl.col(_OUTCOME_PATH) >= spec.threshold)
    )
    rows = [RunRow.from_row_dict(d) for d in _to_dicts(df)]
    ddqn = [r for r in rows if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [
        r for r in rows
        if r.measurements.get('intervention_name') == 'vanilla_dqn'
    ]
    return ddqn, vanilla


def _solve_class(ddqn_n: int, vanilla_n: int) -> str:
    n = min(ddqn_n, vanilla_n)
    if n >= 24:  # ≥ 80% of 30 seeds
        return 'high'
    if n >= 9:  # 30-80%
        return 'mixed'
    return 'low'


def main() -> None:
    print('=' * 100)
    print('Time-to-first-solve link — DDQN vs vanilla on 200k corpus')
    print('  proxy: eval_best_burst_step')
    print('  pair_by: seed (within env)')
    print('  predicted: ddqn < vanilla (DDQN solves faster)')
    print('=' * 100)

    envs = envs_with_threshold()
    per_env: list[tuple[str, str, int, float, float, Verdict]] = []
    g_se_by_class: dict[str, list[tuple[float, float]]] = {
        'high': [], 'mixed': [], 'low': [],
    }
    for env in envs:
        ddqn_runs, vanilla_runs = _load_solved_runs(env)
        if not ddqn_runs or not vanilla_runs:
            continue
        cls = _solve_class(len(ddqn_runs), len(vanilla_runs))
        cmp = paired_comparison_from_runs(
            ddqn_runs, vanilla_runs,
            outcome_path=_PROXY_PATH,
            pair_by=('seed',),
            predicted_direction='a_lt_b',
        )
        g = cmp.measurements.get(f'{_PROXY_PATH}.effect_size_g', float('nan'))
        se = cmp.measurements.get(f'{_PROXY_PATH}.se', float('nan'))
        n_pairs = cmp.measurements.get('n_pairs', 0)
        g_f = float(g) if isinstance(g, (int, float)) else float('nan')
        se_f = float(se) if isinstance(se, (int, float)) else float('nan')
        n_pairs_i = int(n_pairs) if isinstance(n_pairs, (int, float)) else 0
        per_env.append((env, cls, n_pairs_i, g_f, se_f, cmp.verdict))
        g_se_by_class[cls].append((g_f, se_f))

    # Per-env table.
    print(f'  {"env":<25} {"class":<6} {"n_pairs":<8} {"g":<8} {"SE":<6}    verdict')
    print('-' * 80)
    for env, cls, n, g, se, v in per_env:
        g_str = f'{g:+.3f}' if g == g else '   nan'
        se_str = f'{se:.3f}' if se == se else '  nan'
        print(
            f'  {env:<25} {cls:<6} {n:<8} {g_str:<8} {se_str:<6}    '
            f'{v.value}'
        )

    # Pooled by stratum.
    print()
    print('Random-effects pool by solve-rate class:')
    for cls in ('high', 'mixed', 'low'):
        pairs = g_se_by_class[cls]
        if not pairs:
            print(f'  {cls:<6} (no envs)')
            continue
        pool = random_effects_summary(pairs)
        verdict, _ = random_effects_verdict(
            pool, predicted_direction='a_lt_b',
        )
        print(_format_pool(cls, pool, verdict))

    # All envs combined.
    print()
    all_pairs = [p for ps in g_se_by_class.values() for p in ps]
    pool_all = random_effects_summary(all_pairs)
    v_all, _ = random_effects_verdict(pool_all, predicted_direction='a_lt_b')
    print(_format_pool('ALL', pool_all, v_all))


def _format_pool(label: str, pool: PooledStats, v: Verdict) -> str:
    if pool.n_cells < 2:
        return f'  {label:<6} n_envs={pool.n_cells} (too few for pooling)'
    return (
        f'  {label:<6} n_envs={pool.n_cells} '
        f'g_pooled={pool.pooled_g:+.3f} '
        f'I²={pool.I2:.2f} '
        f'PI=[{pool.pi_lo:+.3f}, {pool.pi_hi:+.3f}]  '
        f'verdict={v.value}'
    )


if __name__ == '__main__':
    main()
