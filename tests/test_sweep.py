"""Tests for `sweep` — multi-cell runner over (env, seed) grid.

Uses synthetic runners (no real training) to exercise the sweep
machinery: cell iteration, bridge application, RunRow construction
with measurements, failure capture."""
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


# ============ Bridges ============

@bridge(targets=('value',))
def _always_admit(record: Mapping[str, object]) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.HELD,
        reason='stub admit',
        stats={},
        name='always_admit', targets=('value',),
    )


@bridge(targets=('value',))
def _always_reject(record: Mapping[str, object]) -> BridgeResult:
    del record
    return BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason='stub reject',
        stats={},
        name='always_reject', targets=('value',),
    )


@bridge(targets=('value',))
def _invariant_violated(record: Mapping[str, object]) -> BridgeResult:
    """Mimics what `at_most(gap, threshold)` produces when the
    gap exceeds the threshold: a tautological-tagged
    `Verdict.INVARIANT_VIOLATION` directly."""
    del record
    return BridgeResult(
        verdict=Verdict.INVARIANT_VIOLATION,
        reason='theorem precondition broken',
        stats={'kind': 'tautological', 'of_claim': 'some_claim'},
        name='invariant_violated', targets=('value',),
    )


@bridge(targets=('value',))
def _value_above_threshold(record: Mapping[str, object]) -> BridgeResult:
    """Bridge that actually reads its target from the record."""
    v = record.get('value', 0.0)
    if isinstance(v, (int, float)) and v > 1.05:
        return BridgeResult(
            verdict=Verdict.HELD,
            reason=f'value={v}',
            stats={'value': float(v)},
            name='value_above_threshold', targets=('value',),
        )
    if isinstance(v, (int, float)):
        return BridgeResult(
            verdict=Verdict.NO_EFFECT,
            reason=f'value={v} below 1.05',
            stats={'value': float(v)},
            name='value_above_threshold', targets=('value',),
        )
    return BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason='value missing or non-numeric',
        stats={},
        name='value_above_threshold', targets=('value',),
    )


# ============ Basic sweep ============

def test_sweep_returns_one_row_per_cell() -> None:
    """A 2-env × 3-seed sweep produces 6 rows, no failures."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_always_admit,),
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


def test_sweep_rows_carry_provenance_in_measurements() -> None:
    """Each RunRow carries env_name + seed + total_steps +
    intervention_name in measurements."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('cartpole',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    for r in rows:
        assert r.measurements['env_name'] == 'cartpole'
        assert r.measurements['intervention_name'] == 'h'
        assert r.measurements['total_steps'] == 10
    seeds = sorted(int(r.measurements['seed']) for r in rows)
    assert seeds == [0, 1, 2]


def test_sweep_rows_carry_verdict_from_bridges() -> None:
    h_admit: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_admit', intervention={}, bridges=(_always_admit,),
    )
    h_reject: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_reject', intervention={}, bridges=(_always_reject,),
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
    assert rows_admit[0].verdict is Verdict.HELD
    assert rows_reject[0].verdict is Verdict.NO_EFFECT


def test_sweep_primary_outcome_extracted_per_cell() -> None:
    """`primary_outcome_extractor` lands at
    `outcome.late_window_mean`."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, _ = sweep(
        h,
        env_names=('e',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    outcomes = sorted(
        float(r.measurements['outcome.late_window_mean'])
        for r in rows
    )
    # _constant_runner returns value = 1.0 + seed * 0.1
    assert outcomes == [1.0, 1.1, 1.2]


# ============ Failure tracking ============

def test_sweep_failures_captured_not_raised() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_always_admit,),
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


def test_sweep_failure_carries_provenance() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_fail_test', intervention={}, bridges=(),
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


# ============ Multi-bridge cells ============

def test_sweep_aggregates_multi_bridge_admit() -> None:
    """All bridges admit → cell verdict is HELD; each bridge's
    verdict lands as a separate measurement."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
        bridges=(_always_admit, _always_reject, _value_above_threshold),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    # value=1.0 → reject under value_above_threshold
    assert rows[0].verdict is Verdict.NO_EFFECT
    assert 'bridge.always_admit.verdict' in rows[0].measurements
    assert 'bridge.always_reject.verdict' in rows[0].measurements
    assert 'bridge.value_above_threshold.verdict' in rows[0].measurements


def test_sweep_aggregates_mixed_verdicts() -> None:
    """Any reject → cell verdict is NO_EFFECT."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
        bridges=(_always_admit, _always_reject),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict is Verdict.NO_EFFECT


def test_sweep_no_bridges_yields_power_insufficient() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict is Verdict.POWER_INSUFFICIENT


# ============ Provenance: distinct ids, cycle_id ============

def test_sweep_rows_have_distinct_ids() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_always_admit,),
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
        name='h', intervention={}, bridges=(_always_admit,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
        cycle_id='cycle-42',
    )
    assert rows[0].cycle_id == 'cycle-42'


# ============ Invariant-precedence (verdict aggregation) ============

def test_sweep_invariant_violation_overrides_no_effect() -> None:
    """A tautological-tagged INVARIANT_VIOLATION wins over plain
    NO_EFFECT, producing INVARIANT_VIOLATION at the cell level."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
        bridges=(_invariant_violated, _always_reject),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict is Verdict.INVARIANT_VIOLATION


def test_sweep_invariant_violation_overrides_held() -> None:
    """Even with HELD bridges, an invariant-violation tag dominates."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
        bridges=(_always_admit, _invariant_violated),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict is Verdict.INVARIANT_VIOLATION


def test_sweep_invariant_held_uses_invariant_prefix() -> None:
    """A tautological-tagged HELD result lands under
    `invariant.<name>.verdict`, not `bridge.<name>.verdict`."""
    @bridge(targets=('value',))
    def invariant_held(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD,
            reason='theorem precondition holds',
            stats={'kind': 'tautological'},
            name='theorem_check', targets=('value',),
        )

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(invariant_held,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(0,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].verdict is Verdict.HELD
    assert 'invariant.theorem_check.verdict' in rows[0].measurements
    assert 'bridge.theorem_check.verdict' not in rows[0].measurements


# ============ Bridge↔record contract ============

def test_sweep_bridge_reads_record_per_seed() -> None:
    """A bridge that consumes the record produces per-seed
    verdicts based on the cell's record content."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_value_above_threshold,),
    )
    # _constant_runner returns value = 1.0 + seed * 0.1, so:
    #   seed=0: 1.0  → NO_EFFECT (≤ 1.05)
    #   seed=1: 1.1  → HELD      (> 1.05)
    #   seed=2: 1.2  → HELD
    rows, _ = sweep(
        h,
        env_names=('e',),
        seeds=(0, 1, 2),
        total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    by_seed = {int(r.measurements['seed']): r.verdict for r in rows}
    assert by_seed[0] is Verdict.NO_EFFECT
    assert by_seed[1] is Verdict.HELD
    assert by_seed[2] is Verdict.HELD


def test_sweep_bridge_stats_flow_into_measurements() -> None:
    """The BridgeResult.stats produced by the bridge body land
    under `bridge.<name>.stats.<key>` measurements."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(_value_above_threshold,),
    )
    rows, _ = sweep(
        h,
        env_names=('e',), seeds=(2,), total_steps=10,
        runner=_constant_runner,
        primary_outcome_extractor=_extract_value,
    )
    assert rows[0].measurements['bridge.value_above_threshold.stats.value'] == 1.2
