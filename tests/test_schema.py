"""Tests for schema row dataclasses + as_dict/from_row_dict round-trip.

Verifies each row type round-trips losslessly through its dict
representation. The new shape is flat columnar: provenance fields
+ each measurement entry at top level (unprefixed). No JSON
wrapping. Strict pyright passes throughout — type narrowing in
from_row_dict happens via TypeIs predicates and isinstance checks,
not cast or `# type: ignore`."""
from __future__ import annotations

from corroborate.schema import (
    ComparisonRow,
    RunRow,
)
from corroborate.verdict import RefutationClass, Verdict


# ============ RunRow round-trip ============

def test_run_row_as_dict_flattens_measurements_to_top_level() -> None:
    run = RunRow(
        id='run-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
        measurements={
            'gamma': 0.99,
            'optimizer.inner.lr': 0.001,
            'env_name': 'CartPole-v1',
            'seed': 42,
            'late_window_mean': 120.5,
            'bridge.some_bridge.verdict': 'held',
        },
    )
    d = run.as_dict()
    assert d['id'] == 'run-1'
    assert d['parent_id'] is None
    assert d['cycle_id'] == 'cycle-7'
    assert d['timestamp'] == '2026-04-27T10:00:00Z'
    assert d['verdict'] == 'held'
    # Measurements at top level, unprefixed.
    assert d['gamma'] == 0.99
    assert d['optimizer.inner.lr'] == 0.001
    assert d['env_name'] == 'CartPole-v1'
    assert d['seed'] == 42
    assert d['late_window_mean'] == 120.5
    assert d['bridge.some_bridge.verdict'] == 'held'


def test_run_row_round_trip() -> None:
    run = RunRow(
        id='run-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
        measurements={
            'gamma': 0.99,
            'optimizer.inner.lr': 0.001,
            'env_name': 'CartPole-v1',
            'seed': 42,
            'total_steps': 30_000,
            'intervention_name': 'dqn_with_double_greedify',
            'late_window_mean': 120.5,
            'bridge.some_bridge.verdict': 'held',
        },
    )
    run2 = RunRow.from_row_dict(run.as_dict())
    assert run == run2


def test_run_row_default_measurements_round_trip() -> None:
    run = RunRow(
        id='run-2',
        parent_id=None,
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.POWER_INSUFFICIENT,
    )
    run2 = RunRow.from_row_dict(run.as_dict())
    assert run == run2
    assert dict(run2.measurements) == {}


def test_run_row_skips_none_columns_on_read() -> None:
    """When a parquet column null-pads (this row didn't write it),
    `from_row_dict` skips it — the leaf doesn't appear in
    `measurements`."""
    d = {
        'id': 'r', 'parent_id': None, 'cycle_id': None,
        'timestamp': 't', 'verdict': 'held',
        'gamma': 0.99,
        'optimizer.inner.lr': None,  # null-padded by polars
    }
    run = RunRow.from_row_dict(d)
    assert run.measurements == {'gamma': 0.99}


# ============ ComparisonRow round-trip ============

def test_comparison_row_round_trip_full() -> None:
    cmp = ComparisonRow(
        id='cmp-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        treatment_arm_id='arm-ddqn',
        baseline_arm_id='arm-vanilla',
        predicted_direction='a_gt_b',
        verdict=Verdict.HELD,
        refutation_class=None,
        adequately_powered=True,
        measurements={
            'env_name': 'Asterix-MinAtar',
            'intervention_name': 'ddqn',
            'n_treatment': 15,
            'n_baseline': 15,
            'late_window_mean.arm_a_mean': 42.5,
            'late_window_mean.arm_a_sd': 3.1,
            'late_window_mean.arm_b_mean': 39.0,
            'late_window_mean.arm_b_sd': 4.2,
            'late_window_mean.effect_size_g': 0.91,
            'late_window_mean.se': 0.32,
            'late_window_mean.derived_q': 0.93,
            'late_window_mean.delta_i_population': 0.66,
        },
    )
    cmp2 = ComparisonRow.from_row_dict(cmp.as_dict())
    assert cmp == cmp2


def test_comparison_row_round_trip_with_optional_nones() -> None:
    """`refutation_class` and `predicted_direction` round-trip
    through None vs explicit values."""
    cmp = ComparisonRow(
        id='cmp-2',
        parent_id=None,
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        treatment_arm_id='arm-t',
        baseline_arm_id='arm-b',
        predicted_direction=None,
        verdict=Verdict.POWER_INSUFFICIENT,
        refutation_class=RefutationClass.UNDERPOWERED,
        adequately_powered=False,
        measurements={
            'env_name': 'Acrobot-v1',
            'intervention_name': 'underpowered',
            'n_treatment': 5,
            'n_baseline': 5,
            'late_window_mean.arm_a_mean': -200.0,
            'late_window_mean.arm_b_mean': -180.0,
        },
    )
    cmp2 = ComparisonRow.from_row_dict(cmp.as_dict())
    assert cmp == cmp2
    assert cmp2.predicted_direction is None
    assert cmp2.refutation_class is RefutationClass.UNDERPOWERED


# ============ Collection composition ============

def test_run_row_homogeneous_collection() -> None:
    """RunRows compose into a typed list — basic discipline that
    a `list[RunRow]` is well-typed and pyright tracks element
    types through the collection."""
    runs: list[RunRow] = []
    runs.append(RunRow(
        id='r1', parent_id=None,
        cycle_id=None, timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
    ))
    assert len(runs) == 1


# ============ Verdict round-trip ============

def test_run_row_verdict_round_trip_all_values() -> None:
    """Each Verdict enum value round-trips through its string
    representation."""
    for v in Verdict:
        run = RunRow(
            id='r', parent_id=None, cycle_id=None,
            timestamp='t', verdict=v,
        )
        run2 = RunRow.from_row_dict(run.as_dict())
        assert run2.verdict is v


def test_comparison_row_refutation_class_round_trip_all_values() -> None:
    """Each RefutationClass enum value round-trips."""
    for rc in RefutationClass:
        cmp = ComparisonRow(
            id='c', parent_id=None, cycle_id=None, timestamp='t',
            treatment_arm_id='at', baseline_arm_id='ab',
            predicted_direction=None,
            verdict=Verdict.NO_EFFECT,
            refutation_class=rc,
            adequately_powered=True,
        )
        cmp2 = ComparisonRow.from_row_dict(cmp.as_dict())
        assert cmp2.refutation_class is rc
