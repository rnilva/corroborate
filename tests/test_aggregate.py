"""Tests for `aggregate` — sweep → ComparisonRow hand-off.

Covers the framework's two non-IO primitives that aren't already
exercised in test_statistics.py:

- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts.
- `leaf_signature(measurements)` — configurational fingerprint
  used as a group-by key.

Paired-comparison + cross-env link tests live in
test_statistics.py and the §3 smoke."""
from __future__ import annotations

from corroborate.aggregate import aggregate_cell_verdict, leaf_signature
from corroborate.schema import MeasurementLeaf
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
