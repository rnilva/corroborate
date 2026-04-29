"""Stratify DDQN's link by `jensen_dormancy_gap` on the action-dim
sweep — the framework's-own answer to "what's the scope of DDQN?"

Reads:
  experiments/data/action_dim_sweep/runs.parquet
  experiments/data/action_dim_sweep/traces.parquet  (online_std_q_per_step persisted)

Per-cell dormancy_gap reconstruction:
  observed = mechanism.jensen_gap (already in runs)
  σ_late   = mean over late half of online_std_q_per_step (from traces)
  |A|      = q_network.action_dim or env-derived (looked up via env_catalogue)
  floor    = σ_late · √(2 log |A|)
  gap      = max(0, floor − observed)

Then:
  - Distribution of dormancy_gap by env / intervention.
  - Per-env paired g on outcome.eval_best_burst_mean (DDQN vs
    vanilla, pair-by seed), stratified by:
      class A: pairs where BOTH cells are premise-active (gap=0)
      class B: pairs where AT LEAST ONE cell is premise-dormant
  - Random-effects pool of class-A vs class-B per envs.

Usage:
  uv run python experiments/analyze_action_dim_dormancy.py
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_comparison_from_runs
from corroborate.rl.env_catalogue import get
from corroborate.schema import RunRow
from corroborate.statistics import (
    PooledStats, random_effects_summary, random_effects_verdict,
)
from corroborate.verdict import Verdict


_DATA = Path('experiments/data/action_dim_sweep')
_RUNS = _DATA / 'runs.parquet'
_TRACES = _DATA / 'traces.parquet'


def _late_sigma_per_id(traces_path: Path) -> dict[str, float]:
    """Mean of `online_std_q_per_step` over the late half of each
    cell's trajectory. Read trace once, project per id."""
    df = pl.read_parquet(
        traces_path,
        columns=['id', 'online_std_q_per_step'],
    )
    out: dict[str, float] = {}
    for row in df.iter_rows(named=True):
        cell_id = row['id']
        arr = row['online_std_q_per_step']
        if not isinstance(cell_id, str) or arr is None:
            continue
        v = np.asarray(arr, dtype=np.float64)
        v = v[~np.isnan(v)]
        if v.size < 2:
            out[cell_id] = float('nan')
            continue
        late = v[v.size // 2:]
        out[cell_id] = float(np.mean(late))
    return out


def _action_dim_for(env_name: str) -> int:
    """Look up |A| from the env catalogue. Action_dim is an env
    property, not a per-cell measurement — looking it up here
    keeps the analysis decoupled from the run's leaf
    measurements."""
    return int(get(env_name).n_actions)


def _augment_runs_with_dormancy(
    runs_df: pl.DataFrame, sigma_by_id: Mapping[str, float],
) -> pl.DataFrame:
    """Add `dormancy_gap`, `floor`, `sigma_late` columns to runs_df."""
    out_rows: list[dict[str, object]] = []
    for row in runs_df.iter_rows(named=True):
        cell_id = row['id']
        env_name = row['env_name']
        observed = row.get('mechanism.jensen_gap')
        if not isinstance(cell_id, str) or not isinstance(env_name, str):
            continue
        sigma = sigma_by_id.get(cell_id, float('nan'))
        try:
            n_actions = _action_dim_for(env_name)
        except KeyError:
            n_actions = 0
        if (
            n_actions < 2 or math.isnan(sigma)
            or not isinstance(observed, (int, float))
        ):
            floor = float('nan')
            gap = float('nan')
        else:
            floor = sigma * math.sqrt(2.0 * math.log(n_actions))
            gap = max(0.0, floor - float(observed))
        out_rows.append({
            **row,
            'mediator.jensen_floor_late': floor,
            'mediator.jensen_dormancy_gap': gap,
            'mediator.sigma_late': sigma,
            'mediator.action_dim': n_actions,
        })
    return pl.DataFrame(out_rows)


def _per_env_g(
    runs: list[RunRow], env_name: str, *,
    outcome_path: str = 'outcome.eval_best_burst_mean',
) -> tuple[float, float, int]:
    """Paired g per env on outcome, DDQN-vs-vanilla pair-by seed."""
    env_runs = [
        r for r in runs
        if r.measurements.get('env_name') == env_name
    ]
    ddqn = [r for r in env_runs if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [r for r in env_runs if r.measurements.get('intervention_name') == 'vanilla_dqn']
    if not ddqn or not vanilla:
        return float('nan'), float('nan'), 0
    cmp = paired_comparison_from_runs(
        ddqn, vanilla,
        outcome_path=outcome_path,
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    g_v = cmp.measurements.get(f'{outcome_path}.effect_size_g', float('nan'))
    se_v = cmp.measurements.get(f'{outcome_path}.se', float('nan'))
    n_v = cmp.measurements.get('n_pairs', 0)
    g = float(g_v) if isinstance(g_v, (int, float)) else float('nan')
    se = float(se_v) if isinstance(se_v, (int, float)) else float('nan')
    n = int(n_v) if isinstance(n_v, (int, float)) else 0
    return g, se, n


def _filter_by_dormancy(
    runs: Iterable[RunRow], gap_by_id: Mapping[str, float],
    *, premise_active: bool,
) -> list[RunRow]:
    """Filter to cells whose dormancy_gap meets the predicate.
    `premise_active=True` keeps cells with gap == 0 (HELD); False
    keeps cells with gap > 0 (dormant)."""
    out: list[RunRow] = []
    for r in runs:
        g = gap_by_id.get(r.id, float('nan'))
        if math.isnan(g):
            continue
        if premise_active and g == 0.0:
            out.append(r)
        elif (not premise_active) and g > 0.0:
            out.append(r)
    return out


def _format_pool(label: str, p: PooledStats, v: Verdict) -> str:
    if p.n_cells < 2:
        return f'  {label:<20} n_envs={p.n_cells} (too few)'
    return (
        f'  {label:<20} n_envs={p.n_cells} '
        f'g_pooled={p.pooled_g:+.3f} '
        f'I²={p.I2:.2f} '
        f'PI=[{p.pi_lo:+.3f}, {p.pi_hi:+.3f}]  '
        f'verdict={v.value}'
    )


def main() -> None:
    print('=' * 100)
    print('Action-dim sweep — DDQN link stratified by jensen_dormancy_gap')
    print('=' * 100)

    if not _RUNS.exists() or not _TRACES.exists():
        raise SystemExit(f'corpus not found at {_DATA}')

    sigma_by_id = _late_sigma_per_id(_TRACES)
    runs_df = pl.read_parquet(_RUNS)
    aug = _augment_runs_with_dormancy(runs_df, sigma_by_id)

    # Per-env summary of dormancy gap.
    print()
    print(f'  {"env":<25} {"|A|":>4} {"σ̄_v":>8} {"σ̄_d":>8} {"obs_v":>8} '
          f'{"obs_d":>8} {"floor_v":>8} {"floor_d":>8} '
          f'{"%active_v":>10} {"%active_d":>10}')
    print('-' * 110)
    envs = sorted(set(aug['env_name'].to_list()))
    for env in envs:
        e = aug.filter(pl.col('env_name') == env)
        for arm in ('vanilla_dqn', 'ddqn'):
            arm_df = e.filter(pl.col('intervention_name') == arm)
            if arm_df.height == 0:
                continue
        van = e.filter(pl.col('intervention_name') == 'vanilla_dqn')
        ddq = e.filter(pl.col('intervention_name') == 'ddqn')
        n_actions = _action_dim_for(env)
        sig_v = float(van['mediator.sigma_late'].drop_nulls().mean() or float('nan'))
        sig_d = float(ddq['mediator.sigma_late'].drop_nulls().mean() or float('nan'))
        obs_v = float(van['mechanism.jensen_gap'].drop_nulls().mean() or float('nan'))
        obs_d = float(ddq['mechanism.jensen_gap'].drop_nulls().mean() or float('nan'))
        flr_v = float(van['mediator.jensen_floor_late'].drop_nulls().mean() or float('nan'))
        flr_d = float(ddq['mediator.jensen_floor_late'].drop_nulls().mean() or float('nan'))
        act_v_n = int((van['mediator.jensen_dormancy_gap'] == 0.0).sum() or 0)
        act_d_n = int((ddq['mediator.jensen_dormancy_gap'] == 0.0).sum() or 0)
        print(
            f'  {env:<25} {n_actions:>4} '
            f'{sig_v:>8.2f} {sig_d:>8.2f} '
            f'{obs_v:>8.2f} {obs_d:>8.2f} '
            f'{flr_v:>8.2f} {flr_d:>8.2f} '
            f'{act_v_n}/{van.height:<8} {act_d_n}/{ddq.height:<8}'
        )

    # Per-env paired g + dormancy stratification.
    print()
    print('Per-env paired g on outcome.eval_best_burst_mean:')
    print(f'  {"env":<25} {"|A|":>4} {"all_n":>6} {"g_all":>7} '
          f'{"act_n":>6} {"g_act":>7} {"dor_n":>6} {"g_dor":>7}')
    print('-' * 100)
    runs = [
        RunRow.from_row_dict(d) for d in _to_dicts(runs_df)
    ]
    gap_by_id: dict[str, float] = dict(zip(
        aug['id'].to_list(),
        aug['mediator.jensen_dormancy_gap'].to_list(),
    ))
    g_all_pairs: list[tuple[float, float]] = []
    g_act_pairs: list[tuple[float, float]] = []
    g_dor_pairs: list[tuple[float, float]] = []
    for env in envs:
        n_a = _action_dim_for(env)
        env_runs = [r for r in runs if r.measurements.get('env_name') == env]
        g_all, se_all, n_all = _per_env_g(env_runs, env)
        g_all_pairs.append((g_all, se_all))
        active_runs = _filter_by_dormancy(env_runs, gap_by_id, premise_active=True)
        g_act, se_act, n_act = _per_env_g(active_runs, env)
        g_act_pairs.append((g_act, se_act))
        dormant_runs = _filter_by_dormancy(env_runs, gap_by_id, premise_active=False)
        g_dor, se_dor, n_dor = _per_env_g(dormant_runs, env)
        g_dor_pairs.append((g_dor, se_dor))
        print(
            f'  {env:<25} {n_a:>4} '
            f'{n_all:>6} {g_all:>+7.3f} '
            f'{n_act:>6} {g_act:>+7.3f} '
            f'{n_dor:>6} {g_dor:>+7.3f}'
        )

    print()
    print('Random-effects pool by stratum:')
    pool_all = random_effects_summary(g_all_pairs)
    pool_act = random_effects_summary(g_act_pairs)
    pool_dor = random_effects_summary(g_dor_pairs)
    v_all, _ = random_effects_verdict(pool_all, predicted_direction='a_gt_b')
    v_act, _ = random_effects_verdict(pool_act, predicted_direction='a_gt_b')
    v_dor, _ = random_effects_verdict(pool_dor, predicted_direction='a_gt_b')
    print(_format_pool('all cells', pool_all, v_all))
    print(_format_pool('premise active', pool_act, v_act))
    print(_format_pool('premise dormant', pool_dor, v_dor))


if __name__ == '__main__':
    main()
