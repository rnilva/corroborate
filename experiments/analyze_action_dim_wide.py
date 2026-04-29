"""Canonical scope-search analysis on the action-dim corpora.

Uses the framework's stage-6 + stage-9 primitives directly:

  Stage 6 (AGGREGATE) — `hypothesis_comparison_from_cells(...,
    group_by='env_name')` produces per-env GroupStats + a DL
    random-effects pool in one call.

  Stage 9 (SCOPE-PREDICT) — `meta_regress_comparison(comparison,
    covariate_for=...)` consumes the per-env table and runs
    inverse-variance-weighted OLS of g on caller-provided
    covariates, returning typed CovariateCoefficients with CIs.

This script reads action_dim_sweep + action_dim_wide combined,
constructs the same Hypothesis objects used at collection time
(arm_key match required by from_cells), aggregates per-env on
both `mechanism.jensen_gap` (predicted DDQN < vanilla) and
`outcome.eval_final_mean` (predicted DDQN > vanilla), and
meta-regresses the mechanism g on log(action_dim).

Usage:
  uv run python experiments/analyze_action_dim_wide.py
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import polars as pl

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import hypothesis_comparison_from_cells
from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.meta_regression import (
    MetaRegressionResult, meta_regress_comparison,
)
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get
from corroborate.schema import HypothesisComparisonRow, RunRow
from corroborate.statistics import random_effects_verdict


_CORPORA: tuple[Path, ...] = (
    Path('experiments/data/action_dim_sweep/runs.parquet'),
    Path('experiments/data/action_dim_wide/runs.parquet'),
)


def _make_hypothesis(name: str) -> Hypothesis[DQNTrajectoryRecord]:
    """Reconstruct the Hypothesis used at collection time. The
    arm_key derived from `intervention_arms` must match the
    persisted RunRow.arm_key for `hypothesis_comparison_from_cells`
    to accept the runs."""
    intervention: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(inner=Adam(lr=1e-3), warmup_steps=100),
        'sync_period': 100,
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=intervention,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        intervention['bootstrap'] = partial(
            bootstrap, greedification=double_greedify,
        )
        return Hypothesis(
            name='ddqn', intervention=intervention,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(
                    slot_path='bootstrap',
                    replacement=partial(
                        bootstrap, greedification=double_greedify,
                    ),
                ),
            ),
        )
    raise ValueError(f'unknown hypothesis: {name}')


def _load_combined() -> tuple[list[RunRow], list[RunRow]]:
    """Load both corpora, partitioned by intervention name."""
    ddqn: list[RunRow] = []
    vanilla: list[RunRow] = []
    for p in _CORPORA:
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        for d in _to_dicts(df):
            r = RunRow.from_row_dict(d)
            name = r.measurements.get('intervention_name')
            if name == 'ddqn':
                ddqn.append(r)
            elif name == 'vanilla_dqn':
                vanilla.append(r)
    return ddqn, vanilla


def _stratified_comparison(
    ddqn_runs: Sequence[RunRow],
    vanilla_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    predicted_direction: str,
) -> HypothesisComparisonRow:
    """Stage-6 stratified aggregation: per-env paired g + DL pool.
    Mutates the treatment Hypothesis's predicted_direction for the
    paired test (same Hypothesis identity, different verdicts on
    different outcome paths)."""
    ddqn_h = _make_hypothesis('ddqn')
    vanilla_h = _make_hypothesis('vanilla_dqn')
    # Override predicted_direction for this aggregation.
    ddqn_h_typed = Hypothesis(
        name=ddqn_h.name, intervention=ddqn_h.intervention,
        bridges=ddqn_h.bridges,
        predicted_direction=predicted_direction,  # type: ignore[arg-type]
        intervention_arms=ddqn_h.intervention_arms,
        cycle_id=ddqn_h.cycle_id,
    )
    return hypothesis_comparison_from_cells(
        ddqn_h_typed, ddqn_runs, vanilla_runs,
        outcome_path=outcome_path,
        pair_by=('seed',),
        group_by='env_name',
        baseline_h=vanilla_h,
    )


def _fmt_per_group(comparison: HypothesisComparisonRow) -> None:
    """Pretty-print per-group rows from a stratified comparison."""
    print(f'  {"env":<25} {"|A|":>4} {"n":>4} '
          f'{"g":>8} {"se":>5} {"verdict":<22}')
    print('-' * 80)
    for gs in sorted(comparison.per_group, key=lambda g: str(g.group_value)):
        env = str(gs.group_value)
        try:
            n_a = get(env).n_actions
        except Exception:
            n_a = 0
        g_str = (
            f'{gs.effect_size_g:+.3f}'
            if gs.effect_size_g is not None
            else '   nan'
        )
        se_str = (
            f'{gs.se:.2f}'
            if gs.se is not None and gs.se > 0
            else '  nan'
        )
        print(
            f'  {env:<25} {n_a:>4} {gs.n_pairs:>4} '
            f'{g_str:>8} {se_str:>5} {gs.verdict.value:<22}'
        )


def _action_dim_covariate(env_name: object) -> Mapping[str, float]:
    """Per-env covariate vector for meta-regression. Read from the
    env catalogue so it's reproducible across corpora."""
    try:
        n_a = get(str(env_name)).n_actions
    except Exception:
        return {}
    return {'log_action_dim': math.log(max(n_a, 2))}


def _fmt_meta(label: str, res: MetaRegressionResult) -> None:
    print(f'\nMeta-regression: {label}')
    print(f'  n_strata={res.n_strata}  R²={res.r_squared:+.3f}  '
          f'intercept={res.intercept:+.3f}')
    for c in res.coefficients:
        sig = '✓ SIGNIFICANT' if c.is_significant else ' '
        print(
            f'  {c.name:<18} β={c.coefficient:+.3f}  '
            f'CI=[{c.ci_lo:+.3f}, {c.ci_hi:+.3f}]  '
            f'p={c.p_value:.4f}  {sig}'
        )
    if res.cleavage_axes:
        print(f'  CLEAVAGE: {", ".join(res.cleavage_axes)}')
    else:
        print('  no cleavage axis')


def main() -> None:
    print('=' * 100)
    print('Canonical scope-search — action-dim dependency on DDQN mechanism + link')
    print('=' * 100)
    ddqn, vanilla = _load_combined()
    print(f'  ddqn cells: {len(ddqn)}  vanilla cells: {len(vanilla)}')

    # Stage 6: stratified per-env aggregation on mechanism.
    print()
    print('--- Mechanism edge: paired g on mechanism.jensen_gap (predicted DDQN < vanilla) ---')
    mech = _stratified_comparison(
        ddqn, vanilla,
        outcome_path='mechanism.jensen_gap',
        predicted_direction='a_lt_b',
    )
    _fmt_per_group(mech)
    if mech.pooled is not None:
        v_pool, _ = random_effects_verdict(mech.pooled, predicted_direction='a_lt_b')
        print(f'  pooled g={mech.pooled.pooled_g:+.3f}  '
              f'I²={mech.pooled.I2:.2f}  '
              f'PI=[{mech.pooled.pi_lo:+.3f}, {mech.pooled.pi_hi:+.3f}]  '
              f'verdict={v_pool.value}')

    # Stage 6: link aggregation.
    print()
    print('--- Link edge: paired g on outcome.eval_final_mean (predicted DDQN > vanilla) ---')
    link = _stratified_comparison(
        ddqn, vanilla,
        outcome_path='outcome.eval_final_mean',
        predicted_direction='a_gt_b',
    )
    _fmt_per_group(link)
    if link.pooled is not None:
        v_pool, _ = random_effects_verdict(link.pooled, predicted_direction='a_gt_b')
        print(f'  pooled g={link.pooled.pooled_g:+.3f}  '
              f'I²={link.pooled.I2:.2f}  '
              f'PI=[{link.pooled.pi_lo:+.3f}, {link.pooled.pi_hi:+.3f}]  '
              f'verdict={v_pool.value}')

    # Stage 9: meta-regression on log(action_dim).
    if mech.per_group:
        try:
            mech_meta = meta_regress_comparison(mech, _action_dim_covariate)
            _fmt_meta('mechanism g ~ log_action_dim', mech_meta)
        except ValueError as e:
            print(f'\nMeta-regression mechanism: skipped ({e})')
    if link.per_group:
        try:
            link_meta = meta_regress_comparison(link, _action_dim_covariate)
            _fmt_meta('link g ~ log_action_dim', link_meta)
        except ValueError as e:
            print(f'\nMeta-regression link: skipped ({e})')


if __name__ == '__main__':
    main()
