"""Tests for `HypothesisComparisonRow.from_cells` — the canonical
cross-arm aggregator + supporting primitives (FactRow,
fact_from_bridge_result, natural_strength_from_stats,
transitive_reads).

Validates:
1. Single-group `from_cells` produces correct per-arm + Hedges' g
   on a synthetic paired corpus.
2. Stratified `from_cells` (group_by='env_name') partitions, runs
   per-stratum stats, pools via random-effects, mirrors pooled_g
   in the top-level row.
3. Duplicate `pair_by` keys raise ValueError (silent dedup is the
   bug class this replaces).
4. Asymmetric counts drop unmatched pairs and report the count
   in `n_dropped_unpaired`.
5. FactRow projection from BridgeResult correctly extracts
   natural_strength + delta_i + evidentiary_level.
6. transitive_reads closes over the measurable graph.
7. Bridge.transitive_reads unions targets + measurable closure.
"""
from __future__ import annotations

from collections.abc import Mapping

import math

from corroborate.claim import claim
from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention


@claim
def _stub_treatment_op(x: int) -> int:
    """Stub for tests: gives test hypotheses a non-baseline
    `intervention_arms` so `from_cells` doesn't trip the
    same-arm-key HPO-smuggle assertion."""
    return x


_STUB_INTERVENTION = Intervention(
    slot_path='stub', replacement=_stub_treatment_op,
)
_TREATMENT_ARM_KEY = _STUB_INTERVENTION.arm_key()
from corroborate.schema import (
    HypothesisComparisonRow,
    RunRow,
)
from corroborate.verdict import Verdict


def _run(
    cell_id: str,
    *,
    intervention_name: str,
    seed: int,
    env: str,
    outcome: float,
    extras: Mapping[str, object] | None = None,
    arm_key: str | None = None,
) -> RunRow:
    """Build a minimal RunRow with the measurements `from_cells`
    needs (intervention_name, seed, env_name, outcome).

    `arm_key` defaults to a name-derived placeholder so the
    treatment-vs-baseline arm-key consistency check in
    `from_cells` doesn't trip on test fixtures. Production
    code derives this from `Hypothesis.arm_key()`."""
    measurements: dict[str, object] = {
        'intervention_name': intervention_name,
        'seed': seed,
        'env_name': env,
        'outcome.value': outcome,
    }
    if extras:
        measurements.update(extras)
    resolved_arm_key = (
        arm_key if arm_key is not None
        else (
            'baseline' if intervention_name == 'vanilla'
            else _TREATMENT_ARM_KEY
        )
    )
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00+00:00',
        verdict=Verdict.HELD,
        arm_key=resolved_arm_key,
        measurements=measurements,  # type: ignore[arg-type]
    )


def _hypothesis(name: str, predicted: str | None) -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name=name,
        intervention={},
        predicted_direction=predicted,  # type: ignore[arg-type]
        intervention_arms=(_STUB_INTERVENTION,),
    )


# ============ Single-group from_cells ============

def test_from_cells_single_group_paired_hedges_g() -> None:
    """One env, 6 seeds, treatment outcome systematically beats
    baseline with non-zero Δ variance → finite paired Hedges' g.
    group_by=None."""
    # Per-seed Δs vary so SD(Δ) > 0 and Hedges' g is defined.
    deltas = [0.4, 0.6, 0.5, 0.7, 0.55, 0.45]
    treatment = [
        _run(f't{i}', intervention_name='ddqn', seed=i,
             env='Acrobot', outcome=1.0 + i * 0.1 + deltas[i])
        for i in range(6)
    ]
    baseline = [
        _run(f'b{i}', intervention_name='vanilla', seed=i,
             env='Acrobot', outcome=1.0 + i * 0.1)
        for i in range(6)
    ]
    h = _hypothesis('ddqn', 'a_gt_b')
    row = HypothesisComparisonRow.from_cells(
        h, treatment, baseline,
        outcome_path='outcome.value',
        pair_by=('seed',),
    )
    assert row.intervention_name == 'ddqn'
    assert row.pair_by == ('seed',)
    assert row.group_by is None
    assert row.per_group == ()
    assert row.pooled is None
    assert row.arm_a_n == 6 and row.arm_b_n == 6
    assert row.effect_size_g is not None and row.effect_size_g > 0


def test_from_cells_single_group_no_pairs_returns_underpowered_row() -> None:
    """Treatment + baseline at disjoint seeds → 0 pairs survive →
    POWER_INSUFFICIENT verdict, all stats None."""
    treatment = [_run('t0', intervention_name='ddqn', seed=0,
                      env='X', outcome=1.0)]
    baseline = [_run('b1', intervention_name='vanilla', seed=1,
                     env='X', outcome=0.5)]
    h = _hypothesis('ddqn', None)
    row = HypothesisComparisonRow.from_cells(
        h, treatment, baseline,
        outcome_path='outcome.value', pair_by=('seed',),
    )
    assert row.verdict is Verdict.POWER_INSUFFICIENT
    assert row.effect_size_g is None
    assert row.n_dropped_unpaired == 2


# ============ Stratified from_cells (group_by) ============

def test_from_cells_stratified_per_group_plus_pooled() -> None:
    """Two envs × 6 seeds; per-seed Δ varies so per-stratum
    Hedges' g is finite. Stratified mode produces 2 GroupStats and
    a pooled summary."""
    treatment = []
    baseline = []
    deltas_a = [0.4, 0.6, 0.5, 0.7, 0.55, 0.45]
    deltas_b = [0.7, 0.9, 0.8, 1.0, 0.85, 0.75]
    for env, deltas in (('A', deltas_a), ('B', deltas_b)):
        for s in range(6):
            treatment.append(_run(
                f't_{env}_{s}', intervention_name='ddqn', seed=s,
                env=env, outcome=1.0 + s * 0.1 + deltas[s],
            ))
            baseline.append(_run(
                f'b_{env}_{s}', intervention_name='vanilla', seed=s,
                env=env, outcome=1.0 + s * 0.1,
            ))
    h = _hypothesis('ddqn', 'a_gt_b')
    row = HypothesisComparisonRow.from_cells(
        h, treatment, baseline,
        outcome_path='outcome.value',
        pair_by=('seed',),
        group_by='env_name',
    )
    assert row.group_by == 'env_name'
    assert len(row.per_group) == 2
    assert row.pooled is not None
    assert row.pooled.n_cells == 2
    # Top-level effect_size_g mirrors pooled_g.
    assert row.effect_size_g is not None
    assert math.isclose(row.effect_size_g, row.pooled.pooled_g)
    # Per-group group_values cover both envs.
    group_values = {gs.group_value for gs in row.per_group}
    assert group_values == {'A', 'B'}


def test_from_cells_stratified_drops_groups_with_one_arm() -> None:
    """Treatment has env C but baseline doesn't — env C contributes
    to n_dropped_unpaired and is absent from per_group."""
    treatment = [
        _run(f't_A_{s}', intervention_name='t', seed=s,
             env='A', outcome=1.0 + s * 0.1)
        for s in range(3)
    ] + [
        _run('t_C_0', intervention_name='t', seed=0, env='C',
             outcome=2.0),
    ]
    baseline = [
        _run(f'b_A_{s}', intervention_name='b', seed=s,
             env='A', outcome=0.5 + s * 0.1, arm_key='baseline')
        for s in range(3)
    ]
    h = _hypothesis('t', None)
    row = HypothesisComparisonRow.from_cells(
        h, treatment, baseline,
        outcome_path='outcome.value',
        pair_by=('seed',),
        group_by='env_name',
    )
    # Only env A has both arms; env C dropped.
    assert {gs.group_value for gs in row.per_group} == {'A'}
    assert row.n_dropped_unpaired >= 1


# ============ Validation: duplicate pair_by ============

def test_from_cells_raises_on_duplicate_pair_by_in_treatment() -> None:
    """Two treatment rows with the same seed → silent dict
    overwrite would mask a misconfigured slice. Raise loudly."""
    treatment = [
        _run('t0', intervention_name='ddqn', seed=0,
             env='A', outcome=1.0),
        _run('t0_dup', intervention_name='ddqn', seed=0,
             env='A', outcome=1.5),  # same seed
    ]
    baseline = [
        _run('b0', intervention_name='vanilla', seed=0,
             env='A', outcome=0.5),
    ]
    h = _hypothesis('ddqn', None)
    try:
        HypothesisComparisonRow.from_cells(
            h, treatment, baseline,
            outcome_path='outcome.value', pair_by=('seed',),
        )
    except ValueError as e:
        assert 'duplicate pair_by' in str(e)
        return
    raise AssertionError('expected ValueError')


# ============ Arm-key consistency (A3) ============

def test_from_cells_records_arm_keys() -> None:
    """`from_cells` populates `treatment_arm_key` from
    `h.arm_key()` and defaults `baseline_arm_key` to
    `'baseline'`."""
    treatment = [
        _run('t0', intervention_name='ddqn', seed=0,
             env='A', outcome=1.5),
        _run('t1', intervention_name='ddqn', seed=1,
             env='A', outcome=1.6),
    ]
    baseline = [
        _run('b0', intervention_name='vanilla', seed=0,
             env='A', outcome=1.0),
        _run('b1', intervention_name='vanilla', seed=1,
             env='A', outcome=1.0),
    ]
    h = _hypothesis('ddqn', 'a_gt_b')
    row = HypothesisComparisonRow.from_cells(
        h, treatment, baseline,
        outcome_path='outcome.value', pair_by=('seed',),
    )
    assert row.treatment_arm_key == h.arm_key()
    assert row.baseline_arm_key == 'baseline'
    assert row.treatment_arm_key != row.baseline_arm_key


@claim
def _per_replay_op(x: int) -> int:
    return x + 1


def test_from_cells_uses_baseline_h_arm_key() -> None:
    """When `baseline_h` is provided, its `arm_key()` becomes
    `baseline_arm_key` — supports treatment-vs-treatment
    comparisons."""
    h_baseline: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='per', intervention={},
        intervention_arms=(
            Intervention(slot_path='replay', replacement=_per_replay_op),
        ),
    )
    treatment = [
        _run('t0', intervention_name='ddqn', seed=0,
             env='A', outcome=1.5),
    ]
    baseline = [
        _run('b0', intervention_name='per', seed=0,
             env='A', outcome=1.2,
             arm_key=h_baseline.arm_key()),
    ]
    h_treatment = _hypothesis('ddqn', None)
    row = HypothesisComparisonRow.from_cells(
        h_treatment, treatment, baseline,
        outcome_path='outcome.value', pair_by=('seed',),
        baseline_h=h_baseline,
    )
    assert row.treatment_arm_key == h_treatment.arm_key()
    assert row.baseline_arm_key == h_baseline.arm_key()


def test_from_cells_raises_on_same_arm_key() -> None:
    """When treatment and baseline share an arm key (HPO-smuggle
    indicator), `from_cells` raises ValueError."""
    treatment = [
        _run('t0', intervention_name='x', seed=0,
             env='A', outcome=1.5),
    ]
    baseline = [
        _run('b0', intervention_name='x', seed=0,
             env='A', outcome=1.0),
    ]
    h_treatment = _hypothesis('x', None)
    h_baseline = _hypothesis('x', None)
    assert h_treatment.arm_key() == h_baseline.arm_key()
    try:
        HypothesisComparisonRow.from_cells(
            h_treatment, treatment, baseline,
            outcome_path='outcome.value', pair_by=('seed',),
            baseline_h=h_baseline,
        )
    except ValueError as e:
        assert 'arm_key' in str(e)
        assert 'HPO-smuggle' in str(e)
        return
    raise AssertionError('expected ValueError')


