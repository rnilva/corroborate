"""Tests for `aggregate` — sweep → ComparisonRow hand-off.

Covers the framework's three non-IO primitives that aren't
already exercised in test_statistics.py:

- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts.
- `leaf_signature(measurements)` — configurational fingerprint
  used as a group-by key.
- `reconstruct_bridge_results(run)` — inverse of
  `_bridge_result_to_measurements`. Lossless round-trip of
  name/verdict/targets/stats from RunRow.measurements.

Paired-comparison + cross-env link tests live in
test_statistics.py and the §3 smoke."""
from __future__ import annotations

from corroborate.aggregate import (
    _bridge_result_to_measurements,
    aggregate_cell_verdict,
    leaf_signature,
    reconstruct_bridge_results,
)
from corroborate.bridge import BridgeResult
from corroborate.schema import MeasurementLeaf, RunRow
from corroborate.verdict import Verdict


# ============ aggregate_cell_verdict ============

def test_aggregate_cell_verdict_empty_yields_power_insufficient() -> None:
    assert aggregate_cell_verdict(()) is Verdict.POWER_INSUFFICIENT


def test_aggregate_cell_verdict_invariant_violation_dominates() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.INVARIANT_VIOLATION),
    ) is Verdict.INVARIANT_VIOLATION
    assert aggregate_cell_verdict(
        (Verdict.NO_EFFECT, Verdict.INVARIANT_VIOLATION),
    ) is Verdict.INVARIANT_VIOLATION


def test_aggregate_cell_verdict_no_effect_dominates_held() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.NO_EFFECT),
    ) is Verdict.NO_EFFECT


def test_aggregate_cell_verdict_all_held_yields_held() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.HELD),
    ) is Verdict.HELD


def test_aggregate_cell_verdict_mixed_held_power_insufficient() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.POWER_INSUFFICIENT),
    ) is Verdict.POWER_INSUFFICIENT


# ============ leaf_signature ============

def test_leaf_signature_filters_outcome_and_intervention_name() -> None:
    """`outcome.*`, `bridge.*`, `invariant.*` always filtered.
    `intervention_name` is the framework-typed Hypothesis name —
    also always filtered."""
    measurements: dict[str, MeasurementLeaf] = {
        'gamma': 0.99,
        'optimizer.inner.lr': 0.001,
        'intervention_name': 'h',
        'outcome.late_window_mean': 100.0,
        'bridge.x.verdict': 'held',
        'invariant.y.verdict': 'held',
    }
    sig = leaf_signature(measurements)
    keys = [k for k, _ in sig]
    assert 'gamma' in keys
    assert 'optimizer.inner.lr' in keys
    assert 'intervention_name' not in keys
    assert not any(k.startswith('outcome.') for k in keys)
    assert not any(k.startswith('bridge.') for k in keys)
    assert not any(k.startswith('invariant.') for k in keys)


def test_leaf_signature_excludes_substrate_exogenous_keys() -> None:
    """Substrate-supplied exogenous keys are excluded when caller
    passes them. The framework does NOT hardcode RL key names —
    each substrate names its own exogenous keys."""
    measurements: dict[str, MeasurementLeaf] = {
        'gamma': 0.99,
        'env_name': 'CartPole-v1',
        'seed': 0,
        'total_steps': 1000,
        'intervention_name': 'h',
    }
    # RL substrate's exogenous keys.
    sig = leaf_signature(
        measurements,
        exogenous_keys=frozenset({'env_name', 'seed', 'total_steps'}),
    )
    keys = [k for k, _ in sig]
    assert 'gamma' in keys
    assert 'env_name' not in keys
    assert 'seed' not in keys
    assert 'total_steps' not in keys


def test_leaf_signature_default_no_exogenous_filter() -> None:
    """Without `exogenous_keys`, only `intervention_name` and
    output prefixes are filtered. Substrate keys pass through."""
    sig = leaf_signature({'env_name': 'e', 'seed': 0, 'gamma': 0.9})
    keys = [k for k, _ in sig]
    # All three pass through; framework doesn't hardcode RL names.
    assert set(keys) == {'env_name', 'seed', 'gamma'}


def test_leaf_signature_is_sorted_and_hashable() -> None:
    sig_a = leaf_signature({'b': 1, 'a': 2})
    sig_b = leaf_signature({'a': 2, 'b': 1})
    assert sig_a == sig_b
    # Hashable as a dict key.
    _: dict[tuple[tuple[str, str], ...], int] = {sig_a: 0}


# ============ reconstruct_bridge_results ============

def _runrow_with_measurements(
    measurements: dict[str, MeasurementLeaf],
) -> RunRow:
    return RunRow(
        id='c1', parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00+00:00',
        verdict=Verdict.HELD,
        measurements=measurements,
    )


def test_reconstruct_bridge_results_round_trips_name_verdict_targets_stats() -> None:
    """Forward: BridgeResult → measurements via cell_runner;
    Inverse: measurements → BridgeResult via aggregate. All four
    fields preserved exactly."""
    original = BridgeResult(
        verdict=Verdict.HELD,
        reason='g=0.42 ≥ 0.2 threshold',
        stats={'g': 0.42, 'se': 0.1, 'tier': 'interventional'},
        name='paired_hedges_g',
        targets=('online_max_q_per_step', 'online_min_q_per_step'),
    )
    measurements = _bridge_result_to_measurements(original)
    run = _runrow_with_measurements(measurements)

    rebuilt = reconstruct_bridge_results(run)
    assert len(rebuilt) == 1
    r = rebuilt[0]
    assert r.name == original.name
    assert r.verdict is original.verdict
    assert r.reason == original.reason
    assert r.targets == original.targets
    assert dict(r.stats) == dict(original.stats)


def test_reconstruct_bridge_results_handles_invariant_prefix() -> None:
    """Invariant results (kind='tautological') write under
    `invariant.<name>.*`. Reconstruction picks them up alongside
    bridge results."""
    inv = BridgeResult(
        verdict=Verdict.HELD,
        reason='gap 0.01 ≤ threshold 0.05',
        stats={'kind': 'tautological', 'gap_value': 0.01,
               'threshold': 0.05, 'of_claim': 'periodic_copy'},
        name='periodic_copy_at_most',
        targets=('td_error',),
    )
    bridge = BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason='g=0.05 < 0.2 MDE',
        stats={'g': 0.05},
        name='outcome_diff',
        targets=('outcome.late_window_mean',),
    )
    measurements: dict[str, MeasurementLeaf] = {}
    measurements.update(_bridge_result_to_measurements(inv))
    measurements.update(_bridge_result_to_measurements(bridge))

    rebuilt = reconstruct_bridge_results(
        _runrow_with_measurements(measurements),
    )
    by_name = {r.name: r for r in rebuilt}
    assert by_name['periodic_copy_at_most'].targets == ('td_error',)
    assert by_name['periodic_copy_at_most'].stats['gap_value'] == 0.01
    assert by_name['outcome_diff'].verdict is Verdict.NO_EFFECT
    assert by_name['outcome_diff'].targets == ('outcome.late_window_mean',)


def test_reconstruct_bridge_results_empty_targets_decode_as_empty_tuple() -> None:
    """A bridge with no reads (e.g. `lambda _: BridgeResult(..., targets=())`)
    persists as `bridge.<name>.targets = ''` and reconstructs as
    an empty tuple, NOT a single-element `('',)`."""
    no_reads = BridgeResult(
        verdict=Verdict.HELD,
        reason='trivially true',
        stats={'value': 1.0},
        name='unconditional',
        targets=(),
    )
    measurements = _bridge_result_to_measurements(no_reads)
    rebuilt = reconstruct_bridge_results(
        _runrow_with_measurements(measurements),
    )
    assert len(rebuilt) == 1
    assert rebuilt[0].targets == ()


def test_reconstruct_bridge_results_skips_runs_with_no_bridges() -> None:
    """Runs whose measurements are pure HPs / outcomes (no
    `bridge.*` or `invariant.*` keys) reconstruct to an empty
    tuple, not an error."""
    run = _runrow_with_measurements({
        'gamma': 0.99, 'outcome.late_window_mean': 50.0,
    })
    assert reconstruct_bridge_results(run) == ()


