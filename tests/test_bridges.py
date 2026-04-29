"""Tests for `corroborate.bridges` — generic bridge factories.

Each factory turns shape parameters (record key, threshold,
expected sign) into a typed `Bridge[Mapping]` ready for a
Hypothesis. Verdicts use corroborate's typology: HELD,
NO_EFFECT, INVARIANT_VIOLATION."""
from __future__ import annotations

import numpy as np

from corroborate.bridges import (
    correlation,
    mean_exceeds,
    monotonic,
    variance_shrinks,
)
from corroborate.verdict import Verdict


# ============ monotonic ============

def test_monotonic_held_when_late_exceeds_early() -> None:
    b = monotonic('x', threshold=0.0)
    record = {'x': np.array([1.0, 2.0, 3.0, 4.0])}
    r = b(record)
    assert r.verdict is Verdict.HELD
    assert r.stats['value'] == 2.0  # late mean - early mean
    assert r.targets == ('x',)


def test_monotonic_no_effect_when_below_threshold() -> None:
    b = monotonic('x', threshold=10.0)
    record = {'x': np.array([1.0, 2.0, 3.0, 4.0])}
    r = b(record)
    assert r.verdict is Verdict.NO_EFFECT


def test_monotonic_no_effect_on_decreasing_series() -> None:
    b = monotonic('x', threshold=0.0)
    record = {'x': np.array([4.0, 3.0, 2.0, 1.0])}
    r = b(record)
    assert r.verdict is Verdict.NO_EFFECT


# ============ correlation ============

def test_correlation_held_on_matching_sign_and_magnitude() -> None:
    b = correlation('a', 'b', expected_sign=1, threshold=0.5)
    record = {
        'a': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'b': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    r = b(record)
    assert r.verdict is Verdict.HELD
    rho = r.stats['rho']
    assert isinstance(rho, float) and abs(rho - 1.0) < 1e-9


def test_correlation_no_effect_on_wrong_sign() -> None:
    b = correlation('a', 'b', expected_sign=1, threshold=0.5)
    record = {
        'a': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'b': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    }
    r = b(record)
    assert r.verdict is Verdict.NO_EFFECT


def test_correlation_two_sided_via_zero_expected_sign() -> None:
    b = correlation('a', 'b', expected_sign=0, threshold=0.5)
    record = {
        'a': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'b': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    }
    r = b(record)
    assert r.verdict is Verdict.HELD


def test_correlation_invariant_violation_on_constant_series() -> None:
    b = correlation('a', 'b')
    record = {
        'a': np.array([1.0, 1.0, 1.0]),
        'b': np.array([1.0, 2.0, 3.0]),
    }
    r = b(record)
    assert r.verdict is Verdict.INVARIANT_VIOLATION


# ============ mean_exceeds ============

def test_mean_exceeds_held_above_threshold() -> None:
    b = mean_exceeds('x', threshold=2.0)
    record = {'x': np.array([1.0, 3.0, 5.0])}
    r = b(record)
    assert r.verdict is Verdict.HELD
    assert r.stats['mean'] == 3.0


def test_mean_exceeds_no_effect_below_threshold() -> None:
    b = mean_exceeds('x', threshold=10.0)
    record = {'x': np.array([1.0, 3.0, 5.0])}
    r = b(record)
    assert r.verdict is Verdict.NO_EFFECT


# ============ variance_shrinks ============

def test_variance_shrinks_held_when_late_var_below_ratio() -> None:
    b = variance_shrinks('x', ratio=0.5)
    # Early half [1, 5] has variance 4; late half [3, 3] has variance 0.
    # Ratio = 0/4 = 0 < 0.5 → HELD.
    record = {'x': np.array([1.0, 5.0, 3.0, 3.0])}
    r = b(record)
    assert r.verdict is Verdict.HELD


def test_variance_shrinks_no_effect_when_late_var_at_or_above_ratio() -> None:
    b = variance_shrinks('x', ratio=0.5)
    # Both halves have variance 4 → ratio = 1.0 ≥ 0.5 → NO_EFFECT.
    record = {'x': np.array([1.0, 5.0, 1.0, 5.0])}
    r = b(record)
    assert r.verdict is Verdict.NO_EFFECT


def test_variance_shrinks_invariant_violation_on_zero_early_variance() -> None:
    b = variance_shrinks('x', ratio=0.5)
    # Early half constant → var_early = 0 → ratio undefined.
    record = {'x': np.array([1.0, 1.0, 2.0, 3.0])}
    r = b(record)
    assert r.verdict is Verdict.INVARIANT_VIOLATION


# ============ Bridge contract ============

def test_factories_return_bridges_with_targets_and_name() -> None:
    """Each factory's returned Bridge carries `name` and `targets`
    that match the inputs — used by Bridge.transitive_reads."""
    b1 = monotonic('reward', threshold=0.0)
    assert b1.name == 'monotonic(reward)'
    assert b1.targets == ('reward',)

    b2 = correlation('a', 'b')
    assert b2.name == 'correlation(a,b)'
    assert b2.targets == ('a', 'b')

    b3 = mean_exceeds('q_max', threshold=10.0, name='custom_name')
    assert b3.name == 'custom_name'
    assert b3.targets == ('q_max',)
