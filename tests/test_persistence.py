"""Tests for parquet round-trip across all five row types.

Each row type has paired write/read functions; the test pattern
is: construct row → write to tmp parquet → read back → assert
equality. The persistence layer is flat columnar — every
provenance field and every measurement entry becomes its own
typed parquet column. Heterogeneous rows null-pad cleanly."""
from __future__ import annotations

from pathlib import Path

from corroborate.persistence import (
    read_armrows,
    read_comparisonrows,
    read_corpusrows,
    read_runrows,
    write_armrows,
    write_comparisonrows,
    write_corpusrows,
    write_runrows,
)
from corroborate.schema import (
    ArmRow,
    ComparisonRow,
    CorpusRow,
    RunRow,
)
from corroborate.verdict import RefutationClass, Verdict


# ============ Fixtures ============

def _sample_runrow() -> RunRow:
    return RunRow(
        id='run-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
        measurements={
            'env_name': 'CartPole-v1',
            'seed': 42,
            'total_steps': 30_000,
            'intervention_name': 'dqn_with_double_greedify',
            'gamma': 0.99,
            'optimizer.inner.lr': 0.001,
            'outcome.late_window_mean': 120.5,
            'bridge.some_bridge.verdict': 'held',
            'bridge.some_bridge.stats.rho': 0.85,
        },
    )


# ============ RunRow round-trip ============

def test_runrow_parquet_round_trip_single(tmp_path: Path) -> None:
    path = tmp_path / 'runs.parquet'
    rows = [_sample_runrow()]
    write_runrows(rows, path)
    assert path.exists()

    loaded = read_runrows(path)
    assert len(loaded) == 1
    assert loaded[0] == rows[0]


def test_runrow_parquet_round_trip_multiple_homogeneous(tmp_path: Path) -> None:
    """Multiple rows with the SAME measurement keys round-trip."""
    rows = [
        RunRow(
            id=f'run-{i}', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            measurements={
                'env_name': 'CartPole-v1', 'seed': i,
                'gamma': 0.99,
                'outcome.late_window_mean': float(i),
            },
        )
        for i in range(3)
    ]
    path = tmp_path / 'runs.parquet'
    write_runrows(rows, path)
    loaded = read_runrows(path)
    assert loaded == rows


def test_runrow_parquet_with_no_measurements(tmp_path: Path) -> None:
    """Empty measurements round-trip losslessly."""
    row = RunRow(
        id='no-meas', parent_id=None,
        cycle_id=None, timestamp='t',
        verdict=Verdict.HELD,
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    assert read_runrows(path) == [row]


def test_runrow_parquet_heterogeneous_keys_null_pad(tmp_path: Path) -> None:
    """When two rows carry different measurement paths, polars
    null-pads the missing columns. On read, those nulls are
    skipped — they don't appear in `measurements` of the row that
    didn't carry them."""
    rows_in = [
        RunRow(
            id='r1', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            measurements={'gamma': 0.99, 'env_name': 'CartPole-v1'},
        ),
        RunRow(
            id='r2', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            # No 'gamma' or 'env_name'; carries a different path.
            measurements={'optimizer.inner.lr': 0.001},
        ),
    ]
    path = tmp_path / 'runs.parquet'
    write_runrows(rows_in, path)
    rows_out = read_runrows(path)

    assert 'gamma' in rows_out[0].measurements
    assert 'env_name' in rows_out[0].measurements
    assert 'optimizer.inner.lr' not in rows_out[0].measurements

    assert 'optimizer.inner.lr' in rows_out[1].measurements
    assert 'gamma' not in rows_out[1].measurements
    assert 'env_name' not in rows_out[1].measurements


# ============ ArmRow round-trip ============

def test_armrow_parquet_round_trip(tmp_path: Path) -> None:
    arm = ArmRow(
        id='arm-1',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        run_ids=('run-1', 'run-2', 'run-3'),
        measurements={
            'env_name': 'Asterix-MinAtar',
            'intervention_name': 'ddqn',
            'n': 3,
            'outcome.late_window_mean.arm_mean': 42.5,
            'outcome.late_window_mean.arm_sd': 3.1,
            'gamma': 0.99,
        },
    )
    path = tmp_path / 'arms.parquet'
    write_armrows([arm], path)
    assert read_armrows(path) == [arm]


# ============ ComparisonRow round-trip ============

def test_comparisonrow_parquet_round_trip_full(tmp_path: Path) -> None:
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
            'outcome.late_window_mean.arm_a_mean': 42.5,
            'outcome.late_window_mean.arm_b_mean': 39.0,
            'outcome.late_window_mean.effect_size_g': 0.91,
        },
    )
    path = tmp_path / 'comparisons.parquet'
    write_comparisonrows([cmp], path)
    assert read_comparisonrows(path) == [cmp]


def test_comparisonrow_parquet_with_optional_nones(tmp_path: Path) -> None:
    """Underpowered comparison: refutation_class is set,
    predicted_direction is None. Round-trip preserves both."""
    cmp = ComparisonRow(
        id='cmp-underpowered',
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
        },
    )
    path = tmp_path / 'comparisons.parquet'
    write_comparisonrows([cmp], path)
    loaded = read_comparisonrows(path)
    assert loaded == [cmp]
    assert loaded[0].predicted_direction is None
    assert loaded[0].refutation_class is RefutationClass.UNDERPOWERED


# ============ CorpusRow round-trip ============

def test_corpusrow_parquet_round_trip(tmp_path: Path) -> None:
    corpus = CorpusRow(
        id='corpus-1',
        name='ddqn_link_bridge',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        comparison_ids=('cmp-1', 'cmp-2', 'cmp-3'),
        measurements={
            'n_comparisons': 3,
            'bridge.hasselt_link.verdict': 'power_insufficient',
            'bridge.hasselt_link.stats.pearson_r': 0.28,
        },
    )
    path = tmp_path / 'corpus.parquet'
    write_corpusrows([corpus], path)
    assert read_corpusrows(path) == [corpus]


# ============ Empty collections ============

def test_empty_measurements_via_parquet(tmp_path: Path) -> None:
    """A row with empty measurements must round-trip without
    losing the empty-vs-None distinction."""
    row = RunRow(
        id='r', parent_id=None, cycle_id=None,
        timestamp='t', verdict=Verdict.HELD,
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    loaded = read_runrows(path)
    assert loaded == [row]
    assert dict(loaded[0].measurements) == {}
