"""Tests for `sweep` — multi-cell runner over (env, seed) grid.

Uses synthetic runners (no real training) to exercise the sweep
machinery: cell iteration, bridge application, RunRow construction,
failure capture."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import BridgeResult, bridge
from corroborate.hypothesis import Hypothesis
from corroborate.sweep import sweep
from corroborate.verdict import Verdict


# ============ Fixtures: synthetic runners ============

def _constant_runner(
    h: Hypothesis[Mapping[str, object]],
    env_name: str,
    seed: int,
    total_steps: int,
) -> Mapping[str, object]:
    """Runner that returns a deterministic synthetic record. The
    record carries env_name + seed + total_steps so bridges can
    inspect them."""
    del h
    return {
        'env_name': env_name,
        'seed': seed,
        'total_steps': total_steps,
        'value': 1.0 + seed * 0.1,  # vary per seed
    }


def _failing_runner(
    h: Hypothesis[Mapping[str, object]],
    env_name: str,
    seed: int,
    total_steps: int,
) -> Mapping[str, object]:
    del h, env_name, total_steps
    if seed == 0:
        raise RuntimeError('runner blew up on seed 0')
    return {'value': float(seed)}


def _extract_value(record: Mapping[str, object]) -> float:
    v = record.get('value', 0.0)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


# ============ Bridges that always admit ============

@bridge(targets=('value',))
def _always_admit(record: Mapping[str, object]) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.HELD,
        reason='stub admit',
        stats={},
        name='', targets=(),
    )


@bridge(targets=('value',))
def _always_reject(record: Mapping[str, object]) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason='stub reject',
        stats={},
        name='', targets=(),
    )


# ============ Basic sweep ============

def test_sweep_returns_one_row_per_cell() -> None:
    """A 2-env × 3-seed sweep produces 6 rows, no failures."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit,),
    )
    rows, failures = sweep(
        h,
        env_names=('env_a', 'env_b'),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert len(rows) == 6
    assert len(failures) == 0


def test_sweep_rows_carry_provenance() -> None:
    """Each RunRow carries the cell's env_name and seed."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('cartpole',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert all(r.env_name == 'cartpole' for r in rows)
    assert sorted(r.seed for r in rows) == [0, 1, 2]


def test_sweep_rows_carry_verdict_from_bridges() -> None:
    """Verdict on the RunRow reflects bridge outcomes."""
    h_admit: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_admit',
        intervention={},
        bridges=(_always_admit,),
    )
    h_reject: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_reject',
        intervention={},
        bridges=(_always_reject,),
    )
    rows_admit, _ = sweep(
        h_admit,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    rows_reject, _ = sweep(
        h_reject,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows_admit[0].verdict == Verdict.HELD.value
    assert rows_reject[0].verdict == Verdict.NO_EFFECT.value


def test_sweep_primary_outcome_extracted_per_cell() -> None:
    """`primary_outcome_extractor` is called per record."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(),
    )
    rows, _ = sweep(
        h,
        env_names=('e',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    summaries = sorted(r.primary_outcome_summary for r in rows)
    # _constant_runner returns value = 1.0 + seed * 0.1
    assert summaries == [1.0, 1.1, 1.2]


# ============ Failure tracking ============

def test_sweep_failures_captured_not_raised() -> None:
    """A runner exception captures into CellFailure; sweep
    continues with remaining cells. No silent drops."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit,),
    )
    rows, failures = sweep(
        h,
        env_names=('e',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_failing_runner,
        primary_outcome_extractor=_extract_value,
    )
    # Seed 0 fails; seeds 1, 2 succeed.
    assert len(rows) == 2
    assert len(failures) == 1
    assert failures[0].seed == 0
    assert 'RuntimeError' in failures[0].error
    assert 'blew up' in failures[0].error


def test_sweep_failure_carries_provenance() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_fail_test',
        intervention={},
        bridges=(),
    )
    _, failures = sweep(
        h,
        env_names=('e',),
        seeds=(0,),
        total_steps=10,
        runner=_failing_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert len(failures) == 1
    assert failures[0].intervention_name == 'h_fail_test'
    assert failures[0].env_name == 'e'
    assert failures[0].seed == 0
    assert isinstance(failures[0].duration_s, float)


# ============ Multi-bridge cells ============

def test_sweep_aggregates_multi_bridge_admit() -> None:
    """All bridges admit → cell verdict is held."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit, _always_admit, _always_admit),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict == Verdict.HELD.value
    assert len(rows[0].facts) == 3


def test_sweep_aggregates_mixed_verdicts() -> None:
    """Any reject → cell verdict is no_effect."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit, _always_reject),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict == Verdict.NO_EFFECT.value


def test_sweep_no_bridges_yields_power_insufficient() -> None:
    """Cell with zero bridges has nothing to verdict on; default
    is power_insufficient (the framework's 'cannot tell' tag)."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict == Verdict.POWER_INSUFFICIENT.value


# ============ Provenance: mechanism_key, timestamp, etc. ============

def test_sweep_rows_share_mechanism_key() -> None:
    """All rows from one sweep share the hypothesis's mechanism_key."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={'slot': 'value'},
        bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0, 1), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    keys = {r.mechanism_key for r in rows}
    assert len(keys) == 1
    assert next(iter(keys)) == h.mechanism_key


def test_sweep_rows_have_distinct_ids() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0, 1, 2), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    ids = {r.id for r in rows}
    assert len(ids) == 3


def test_sweep_cycle_id_propagates() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h',
        intervention={},
        bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
        cycle_id='cycle-42',
    )
    assert rows[0].cycle_id == 'cycle-42'
