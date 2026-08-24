"""Tests for `paired_comparison` — the canonical paired-comparison
analysis (formerly `hypothesis_comparison_from_cells` /
`HypothesisComparisonRow.from_cells`, moved to
`corroborate.analyses.paired_comparison` in Phase 3).

Validates:
1. Single-group `paired_comparison` produces correct per-arm +
   Hedges' g on a synthetic paired corpus.
2. Stratified `paired_comparison` (group_by='env_name')
   partitions, runs per-stratum stats, pools via random-effects,
   mirrors pooled_g in the top-level result.
3. Duplicate `pair_by` keys raise ValueError (silent dedup is the
   bug class this replaces).
4. Asymmetric counts drop unmatched pairs and report the count
   in `n_dropped_unpaired`.
5. Same-arm contrast raises ValueError (HPO-smuggle indicator)."""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.paired.paired_comparison import paired_comparison
from corroborate.data import cells_to_dataframe


def _cell(
    *,
    arm_key: str,
    seed: int,
    env: str,
    outcome: float,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a flat cell dict matching what the runner emits."""
    cell: dict[str, object] = {
        'arm_key': arm_key,
        'seed': seed,
        'env_name': env,
        'outcome.value': outcome,
    }
    if extras:
        cell.update(extras)
    return cell


# ============ Single-group paired_comparison ============

def test_single_group_paired_hedges_g() -> None:
    """One env, 6 seeds, treatment outcome systematically beats
    baseline with non-zero Δ variance → finite paired Hedges' g.
    group_by=None."""
    deltas = [0.4, 0.6, 0.5, 0.7, 0.55, 0.45]
    cells: list[Mapping[str, object]] = []
    for i in range(6):
        cells.append(_cell(
            arm_key='ddqn', seed=i, env='Acrobot',
            outcome=1.0 + i * 0.1 + deltas[i],
        ))
        cells.append(_cell(
            arm_key='vanilla', seed=i, env='Acrobot',
            outcome=1.0 + i * 0.1,
        ))
    result = paired_comparison.fn(
        cells_to_dataframe(cells),
        treatment_arm='ddqn', baseline_arm='vanilla',
        outcome_path='outcome.value',
        pair_by=('seed',),
        predicted_direction='a_gt_b',
    )
    assert result.treatment_arm == 'ddqn'
    assert result.baseline_arm == 'vanilla'
    assert result.pair_by == ('seed',)
    assert result.group_by is None
    assert result.per_group == ()
    assert result.pooled is None
    assert result.arm_a_n == 6 and result.arm_b_n == 6
    assert result.effect_size_g is not None and result.effect_size_g > 0


def test_single_group_no_pairs_returns_empty_stats() -> None:
    """Treatment + baseline at disjoint seeds → 0 pairs survive →
    empty stats. Bridges author the verdict via `holds_when`
    against the underpowered shape (`effect_size_g is None` /
    `arm_a_n == 0`)."""
    cells: list[Mapping[str, object]] = [
        _cell(arm_key='ddqn', seed=0, env='X', outcome=1.0),
        _cell(arm_key='vanilla', seed=1, env='X', outcome=0.5),
    ]
    result = paired_comparison.fn(
        cells_to_dataframe(cells),
        treatment_arm='ddqn', baseline_arm='vanilla',
        outcome_path='outcome.value', pair_by=('seed',),
    )
    assert result.effect_size_g is None
    assert result.se is None
    assert result.arm_a_n == 0
    assert result.n_dropped_unpaired == 2


# ============ Stratified paired_comparison (group_by) ============

def test_stratified_per_group_plus_pooled() -> None:
    """Two envs × 6 seeds; per-seed Δ varies so per-stratum
    Hedges' g is finite. Stratified mode produces 2 GroupStats
    and a pooled summary."""
    deltas_a = [0.4, 0.6, 0.5, 0.7, 0.55, 0.45]
    deltas_b = [0.7, 0.9, 0.8, 1.0, 0.85, 0.75]
    cells: list[Mapping[str, object]] = []
    for env, deltas in (('A', deltas_a), ('B', deltas_b)):
        for s in range(6):
            cells.append(_cell(
                arm_key='ddqn', seed=s, env=env,
                outcome=1.0 + s * 0.1 + deltas[s],
            ))
            cells.append(_cell(
                arm_key='vanilla', seed=s, env=env,
                outcome=1.0 + s * 0.1,
            ))
    result = paired_comparison.fn(
        cells_to_dataframe(cells),
        treatment_arm='ddqn', baseline_arm='vanilla',
        outcome_path='outcome.value',
        pair_by=('seed',),
        group_by='env_name',
        predicted_direction='a_gt_b',
    )
    assert result.group_by == 'env_name'
    assert len(result.per_group) == 2
    assert result.pooled is not None
    assert result.pooled.n_cells == 2
    assert result.effect_size_g is not None
    assert math.isclose(result.effect_size_g, result.pooled.pooled_g)
    group_values = {gs.group_value for gs in result.per_group}
    assert group_values == {'A', 'B'}


def test_stratified_drops_groups_with_one_arm() -> None:
    """Treatment has env C but baseline doesn't — env C contributes
    to n_dropped_unpaired and is absent from per_group."""
    cells: list[Mapping[str, object]] = []
    for s in range(3):
        cells.append(_cell(
            arm_key='t', seed=s, env='A', outcome=1.0 + s * 0.1,
        ))
        cells.append(_cell(
            arm_key='b', seed=s, env='A', outcome=0.5 + s * 0.1,
        ))
    cells.append(_cell(
        arm_key='t', seed=0, env='C', outcome=2.0,
    ))
    result = paired_comparison.fn(
        cells_to_dataframe(cells),
        treatment_arm='t', baseline_arm='b',
        outcome_path='outcome.value',
        pair_by=('seed',),
        group_by='env_name',
    )
    assert {gs.group_value for gs in result.per_group} == {'A'}
    assert result.n_dropped_unpaired >= 1


# ============ Validation: duplicate pair_by ============

def test_raises_on_duplicate_pair_by_in_treatment() -> None:
    """Two treatment cells with the same seed → silent dict
    overwrite would mask a misconfigured slice. Raise loudly."""
    cells: list[Mapping[str, object]] = [
        _cell(arm_key='ddqn', seed=0, env='A', outcome=1.0),
        _cell(arm_key='ddqn', seed=0, env='A', outcome=1.5),  # same seed
        _cell(arm_key='vanilla', seed=0, env='A', outcome=0.5),
    ]
    try:
        paired_comparison.fn(
            cells_to_dataframe(cells),
            treatment_arm='ddqn', baseline_arm='vanilla',
            outcome_path='outcome.value', pair_by=('seed',),
        )
    except ValueError as e:
        assert 'duplicate pair_by' in str(e)
        return
    raise AssertionError('expected ValueError')


# ============ Validation: same-arm HPO-smuggle ============

def test_raises_on_same_arm() -> None:
    """When treatment and baseline share an arm key (HPO-smuggle
    indicator), `paired_comparison` raises ValueError."""
    cells: list[Mapping[str, object]] = [
        _cell(arm_key='x', seed=0, env='A', outcome=1.5),
        _cell(arm_key='x', seed=1, env='A', outcome=1.0),
    ]
    try:
        paired_comparison.fn(
            cells_to_dataframe(cells),
            treatment_arm='x', baseline_arm='x',
            outcome_path='outcome.value', pair_by=('seed',),
        )
    except ValueError as e:
        assert 'self-against-self' in str(e)
        return
    raise AssertionError('expected ValueError')
