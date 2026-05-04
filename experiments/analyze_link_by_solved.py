"""Stratified link analysis on solved/unsolved/saturated cells.

The 10-env analysis surfaces a measurement-degeneracy: 4 of the
"solved" envs (CartPole, Catch, DiscountingChain — and to a
lesser extent BernoulliBandit) saturate `eval_best_burst_mean`
at the discounted ceiling. Both arms read the *same* number, so
the paired g is structurally null regardless of policy quality.

This script stratifies cells by:
  - **saturated**: corpus-max is reached on both arms; no headroom.
  - **solved-with-headroom**: `is_solved(env, best) is True` AND
    not saturated.
  - **unsolved**: `is_solved(env, best) is False`.
  - **no-threshold**: env's threshold is `None` in the catalogue.

The link verdict is then computed *only* on cells with outcome
headroom — saturated cells are excluded because they can't carry
link signal.

Usage:
  uv run python experiments/analyze_link_by_solved.py
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import polars as pl

from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.corpus.aggregate import hypothesis_comparison_from_cells
from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import Intervention
from corroborate_rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate_rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import get
from corroborate_rl.env_solve_thresholds import SOLVE_THRESHOLDS, is_solved
from corroborate.corpus.schema import HypothesisComparisonRow, RunRow
from corroborate.stats import random_effects_summary, random_effects_verdict


_CORPORA: tuple[Path, ...] = (
    Path('experiments/data/action_dim_sweep/runs.parquet'),
    Path('experiments/data/action_dim_wide/runs.parquet'),
)


def _make_hypothesis(name: str) -> Hypothesis[DQNTrajectoryRecord]:
    intervention: dict[str, object] = {
        'total_steps': 200_000, 'eval_every': 20_000,
        'n_episodes': 5, 'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(inner=Adam(lr=1e-3), warmup_steps=100),
        'sync_period': 100,
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=intervention, predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        intervention['bootstrap'] = partial(
            bootstrap, greedification=double_greedify,
        )
        return Hypothesis(
            name='ddqn', intervention=intervention, predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(
                    slot_path='bootstrap',
                    replacement=partial(
                        bootstrap, greedification=double_greedify,
                    ),
                ),
            ),
        )
    raise ValueError(name)


def _classify_cell(
    env_name: str, best_burst: float, corpus_max_per_env: dict[str, float],
    *, ceiling_eps: float = 1e-3,
) -> str:
    """Classify a cell into:
    - 'saturated' if best_burst is within `ceiling_eps` of the
      corpus-max for that env (both arms reach this same value;
      nothing for DDQN to exceed).
    - 'solved' if `is_solved(env, best) is True` and not saturated.
    - 'unsolved' if `is_solved(env, best) is False`.
    - 'no_threshold' if the env has no canonical threshold."""
    corpus_max = corpus_max_per_env.get(env_name, float('-inf'))
    if abs(best_burst - corpus_max) <= ceiling_eps:
        return 'saturated'
    spec = SOLVE_THRESHOLDS.get(env_name)
    if spec is None or spec.threshold is None:
        return 'no_threshold'
    is_s = is_solved(env_name, best_burst)
    if is_s is True:
        return 'solved'
    if is_s is False:
        return 'unsolved'
    return 'no_threshold'


def _augment_with_class(df: pl.DataFrame) -> pl.DataFrame:
    corpus_max = {
        env: float(df.filter(pl.col('env_name') == env)['eval_best_burst_mean'].max() or float('-inf'))
        for env in df['env_name'].unique()
    }
    classes = []
    for row in df.iter_rows(named=True):
        env = row['env_name']
        best = row.get('eval_best_burst_mean')
        if not isinstance(best, (int, float)) or math.isnan(float(best)):
            classes.append('missing')
            continue
        classes.append(_classify_cell(env, float(best), corpus_max))
    return df.with_columns(pl.Series('_class', classes))


def _per_env_link(
    runs: list[RunRow], outcome_path: str = 'eval_final_mean',
) -> HypothesisComparisonRow:
    ddqn_h = _make_hypothesis('ddqn')
    vanilla_h = _make_hypothesis('vanilla_dqn')
    ddqn_h_typed = Hypothesis(
        name=ddqn_h.name, intervention=ddqn_h.intervention,
        predicted_direction='a_gt_b',
        intervention_arms=ddqn_h.intervention_arms,
    )
    ddqn = [r for r in runs if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [r for r in runs if r.measurements.get('intervention_name') == 'vanilla_dqn']
    return hypothesis_comparison_from_cells(
        ddqn_h_typed, ddqn, vanilla,
        outcome_path=outcome_path, pair_by=('seed',),
        group_by='env_name', baseline_h=vanilla_h,
    )


def _print_per_group(label: str, comp: HypothesisComparisonRow) -> None:
    print(f'  --- {label} ---')
    for gs in sorted(comp.per_group, key=lambda g: str(g.group_value)):
        env = str(gs.group_value)
        try:
            n_a = get(env).n_actions
        except Exception:
            n_a = 0
        g_str = (
            f'{gs.effect_size_g:+.3f}' if gs.effect_size_g is not None else '   nan'
        )
        se_str = (
            f'{gs.se:.2f}' if gs.se is not None and gs.se > 0 else '  nan'
        )
        print(
            f'    {env:<25} |A|={n_a:<3} n={gs.n_pairs:<3} '
            f'g={g_str:<8} se={se_str:<5} {gs.verdict.value}'
        )
    if comp.pooled is not None:
        v, _ = random_effects_verdict(comp.pooled, predicted_direction='a_gt_b')
        print(
            f'    pooled g={comp.pooled.pooled_g:+.3f} '
            f'I²={comp.pooled.I2:.2f} '
            f'PI=[{comp.pooled.pi_lo:+.3f}, {comp.pooled.pi_hi:+.3f}] '
            f'verdict={v.value}'
        )


def main() -> None:
    print('=' * 100)
    print('Link analysis stratified by solved/saturated/unsolved cell-class')
    print('=' * 100)

    dfs = [pl.read_parquet(p) for p in _CORPORA if p.exists()]
    df = pl.concat(dfs, how='vertical_relaxed')
    print(f'  total cells: {df.height}')
    aug = _augment_with_class(df)

    print()
    print(f'  {"env":<25} {"saturated":>10} {"solved":>8} {"unsolved":>10} '
          f'{"no_thr":>8} {"missing":>8}')
    print('-' * 90)
    for env in sorted(aug['env_name'].unique()):
        e = aug.filter(pl.col('env_name') == env)
        cls_counts = {c: 0 for c in (
            'saturated', 'solved', 'unsolved', 'no_threshold', 'missing'
        )}
        for c in e['_class'].to_list():
            cls_counts[c] = cls_counts.get(c, 0) + 1
        print(
            f'  {env:<25} {cls_counts["saturated"]:>10} '
            f'{cls_counts["solved"]:>8} {cls_counts["unsolved"]:>10} '
            f'{cls_counts["no_threshold"]:>8} {cls_counts["missing"]:>8}'
        )

    print()
    print('Link analysis on each cell-class subset:')
    print('  (paired g on eval_final_mean per env)')
    for cls in ('solved', 'no_threshold', 'unsolved'):
        sub = aug.filter(pl.col('_class') == cls)
        if sub.height == 0:
            print(f'\n  {cls}: 0 cells')
            continue
        runs = [RunRow.from_row_dict(d) for d in _to_dicts(sub)]
        # Need both arms to compute paired g.
        envs_with_both = [
            env for env in sorted({r.measurements.get('env_name') for r in runs if isinstance(r.measurements.get('env_name'), str)})
            if any(r.measurements.get('env_name') == env and r.measurements.get('intervention_name') == 'ddqn' for r in runs)
            and any(r.measurements.get('env_name') == env and r.measurements.get('intervention_name') == 'vanilla_dqn' for r in runs)
        ]
        runs_filtered = [r for r in runs if r.measurements.get('env_name') in envs_with_both]
        if not runs_filtered:
            print(f'\n  {cls}: no env has both arms; skipped')
            continue
        try:
            comp = _per_env_link(runs_filtered, outcome_path='eval_final_mean')
            print()
            _print_per_group(f'class={cls!r}, n_envs={len(envs_with_both)}', comp)
        except ValueError as e:
            print(f'\n  {cls}: skipped ({e})')

    # Best-burst link on the same cell-class subsets — same logic
    # but on best-burst (which the saturation classifier uses as
    # input). Useful for comparing "did DDQN reach a higher peak?"
    print()
    print('Best-burst link analysis on each cell-class subset:')
    print('  (paired g on eval_best_burst_mean per env)')
    for cls in ('solved', 'no_threshold', 'unsolved'):
        sub = aug.filter(pl.col('_class') == cls)
        if sub.height == 0:
            continue
        runs = [RunRow.from_row_dict(d) for d in _to_dicts(sub)]
        envs_with_both = [
            env for env in sorted({r.measurements.get('env_name') for r in runs if isinstance(r.measurements.get('env_name'), str)})
            if any(r.measurements.get('env_name') == env and r.measurements.get('intervention_name') == 'ddqn' for r in runs)
            and any(r.measurements.get('env_name') == env and r.measurements.get('intervention_name') == 'vanilla_dqn' for r in runs)
        ]
        runs_filtered = [r for r in runs if r.measurements.get('env_name') in envs_with_both]
        if not runs_filtered:
            continue
        try:
            comp = _per_env_link(runs_filtered, outcome_path='eval_best_burst_mean')
            print()
            _print_per_group(f'class={cls!r}, n_envs={len(envs_with_both)}', comp)
        except ValueError as e:
            print(f'  skipped ({e})')


if __name__ == '__main__':
    main()
