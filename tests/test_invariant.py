"""Tests for `@invariant(of=claim, targets=...)` — tautological-
tagged bridges attached to Claims (axiom 18)."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import Bridge, BridgeResult
from corroborate.claim import claim
from corroborate.invariant import invariant
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
