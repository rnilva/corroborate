"""Tests for `@invariant(of=claim, targets=...)` — tautological-
tagged bridges attached to Claims (axiom 18) — and the `bounded`
theorem-condition invariant factory."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from corroborate.bridge import Bridge, BridgeResult
from corroborate.claim import claim
from corroborate.invariant import bounded, invariant
from corroborate.reductions import from_key, max_abs
from corroborate.verdict import Verdict


# ============ Decoration produces a Bridge ============

def test_invariant_returns_bridge() -> None:
    @claim
    def some_claim(x: int) -> int:
        return x

    @invariant(of=some_claim, targets=('q_max',))
    def q_bounded(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='|q|<1000',
            stats={}, name='', targets=(),
        )

    assert isinstance(q_bounded, Bridge)
    assert q_bounded.targets == ('q_max',)


def test_invariant_default_name_includes_claim_name() -> None:
    @claim
    def my_step(x: int) -> int:
        return x

    @invariant(of=my_step, targets=('x',))
    def my_test(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    assert 'my_test' in my_test.name
    assert 'my_step' in my_test.name


def test_invariant_explicit_name_overrides() -> None:
    @claim
    def some_claim(x: int) -> int:
        return x

    @invariant(of=some_claim, targets=('x',), name='custom_inv')
    def fn(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    assert fn.name == 'custom_inv'


# ============ Tag injection ============

def test_invariant_injects_tautological_kind() -> None:
    """The wrapper auto-tags every returned BridgeResult's stats
    with `kind='tautological'`. The author does not have to set
    it (and existing stats are preserved)."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok',
            stats={'value': 0.5, 'threshold': 1.0},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['kind'] == 'tautological'
    assert result.stats['value'] == 0.5
    assert result.stats['threshold'] == 1.0


def test_invariant_injects_of_claim_name() -> None:
    @claim
    def my_specific_claim(x: int) -> int:
        return x

    @invariant(of=my_specific_claim, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['of_claim'] == 'my_specific_claim'


def test_invariant_tags_added_even_on_reject() -> None:
    """Whether HELD or REJECT, the invariant tag is added — that's
    how aggregate_verdict will distinguish 'invariant violation'
    from 'normal refutation' downstream."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.NO_EFFECT, reason='violated', stats={},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.verdict is Verdict.NO_EFFECT
    assert result.stats['kind'] == 'tautological'


# ============ Stats preservation ============

def test_invariant_preserves_existing_stats() -> None:
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('x',))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='',
            stats={'a': 1, 'b': 'foo', 'c': True, 'd': 3.14},
            name='', targets=(),
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.stats['a'] == 1
    assert result.stats['b'] == 'foo'
    assert result.stats['c'] is True
    assert result.stats['d'] == 3.14
    assert result.stats['kind'] == 'tautological'  # added on top


def test_invariant_uses_decorator_targets_when_inner_empty() -> None:
    """The wrapper backfills targets from the decorator if the
    inner BridgeResult left them empty — convenience for authors
    who don't bother setting them inside the body."""
    @claim
    def step(x: int) -> int:
        return x

    @invariant(of=step, targets=('q_max', 'epsilon'))
    def inv(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),  # empty — decorator backfills
        )

    record: Mapping[str, object] = {'x': 0}
    result = inv(record)
    assert result.targets == ('q_max', 'epsilon')


# ============ bounded(of=Measurable, ...) factory ============

def test_bounded_returns_bridge_with_measurable_reads_as_targets() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = bounded(
        max_abs(from_key('q_max')),
        threshold=1e3,
        theorem='Banach contraction on T*',
        of_claim=some_step,
    )
    assert isinstance(inv, Bridge)
    # The measurable's reads are propagated as the bridge's targets.
    assert inv.targets == ('q_max',)


def test_bounded_held_when_value_under_threshold() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = bounded(
        max_abs(from_key('q_max')),
        threshold=10.0,
        theorem='Banach contraction',
        of_claim=some_step,
    )
    record: Mapping[str, jnp.ndarray] = {
        'q_max': jnp.asarray([1.0, 2.0, -3.0, 5.0]),
    }
    result = inv(record)
    assert result.verdict is Verdict.HELD
    assert result.stats['kind'] == 'tautological'
    assert result.stats['theorem'] == 'Banach contraction'
    assert result.stats['value'] == 5.0
    assert result.stats['threshold'] == 10.0


def test_bounded_invariant_violation_when_value_over_threshold() -> None:
    @claim
    def some_step(x: int) -> int:
        return x

    inv = bounded(
        max_abs(from_key('q_max')),
        threshold=2.0,
        theorem='Q-bounded under contraction',
        of_claim=some_step,
    )
    record: Mapping[str, jnp.ndarray] = {
        'q_max': jnp.asarray([1.0, 50.0, 3.0]),
    }
    result = inv(record)
    # |max_q| = 50 > 2 → out of theorem's domain → INVARIANT_VIOLATION
    assert result.verdict is Verdict.INVARIANT_VIOLATION
    assert result.stats['kind'] == 'tautological'
    assert result.stats['value'] == 50.0


def test_bounded_carries_of_claim_name() -> None:
    @claim
    def my_specific_step(x: int) -> int:
        return x

    inv = bounded(
        max_abs(from_key('x')),
        threshold=100.0,
        theorem='dummy theorem',
        of_claim=my_specific_step,
    )
    record: Mapping[str, jnp.ndarray] = {'x': jnp.asarray([1.0])}
    result = inv(record)
    assert result.stats['of_claim'] == 'my_specific_step'
