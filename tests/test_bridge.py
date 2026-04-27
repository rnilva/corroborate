"""Tests for `Bridge`, `BridgeResult`, and the `@bridge`
decorator. Strict typing exercised: Records are `Mapping[str,
object]` and value narrowing happens via `isinstance` (not cast)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from corroborate.bridge import Bridge, BridgeResult, bridge
from corroborate.verdict import RefutationClass, Verdict


# ============ BridgeResult ============

def test_bridge_result_minimal_construction() -> None:
    r = BridgeResult(
        verdict=Verdict.HELD,
        reason='ρ = +0.92 with adequate power',
        stats={'rho': 0.92, 'n': 30},
        name='correlation(q_mean, epsilon)',
        targets=('q_mean', 'epsilon'),
    )
    assert r.verdict is Verdict.HELD
    assert r.refutation_class is None
    assert r.stats['rho'] == 0.92


def test_bridge_result_with_refutation_class() -> None:
    """NO_EFFECT verdicts can carry a refutation_class refinement."""
    r = BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason='|g| < MDE',
        stats={'g': 0.05, 'mde': 0.5},
        name='outcome',
        targets=('final_return',),
        refutation_class=RefutationClass.NULL_EFFECT,
    )
    assert r.refutation_class is RefutationClass.NULL_EFFECT


def test_bridge_result_stats_value_types_are_scalar_only() -> None:
    """Stats are typed as `Mapping[str, float|int|bool|str]` —
    pyright rejects arrays/lists at compile time. Verified here at
    runtime by constructing with each allowed type."""
    r = BridgeResult(
        verdict=Verdict.HELD,
        reason='all primitive scalar types accepted',
        stats={
            'pearson': 0.83,
            'n': 17,
            'adequately_powered': True,
            'estimand': 'E[Y | do(X)]',
        },
        name='multi_stat',
        targets=('x',),
    )
    assert r.stats['n'] == 17
    assert r.stats['adequately_powered'] is True


# ============ @bridge decorator ============

def test_bridge_decorator_explicit_targets() -> None:
    @bridge(targets=('max_q_late',))
    def max_q_decreases(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='stub', stats={},
            name='', targets=(),
        )

    assert isinstance(max_q_decreases, Bridge)
    assert max_q_decreases.name == 'max_q_decreases'
    assert max_q_decreases.targets == ('max_q_late',)


def test_bridge_decorator_explicit_name_overrides_fn_name() -> None:
    @bridge(targets=('a', 'b'), name='custom_name')
    def some_fn(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='stub', stats={},
            name='', targets=(),
        )

    assert some_fn.name == 'custom_name'


def test_bridge_decorator_empty_string_name_is_kept() -> None:
    """Empty string is a valid (if odd) name; only `None` falls
    back to the function name."""
    @bridge(targets=('x',), name='')
    def some_fn(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='stub', stats={},
            name='', targets=(),
        )

    assert some_fn.name == ''


# ============ Calling a Bridge ============

def test_bridge_passes_record_through_to_fn() -> None:
    captured: list[Mapping[str, object]] = []

    @bridge(targets=('x',))
    def echo(record: Mapping[str, object]) -> BridgeResult:
        captured.append(record)
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok',
            stats={}, name='', targets=(),
        )

    record_in: Mapping[str, object] = {'x': 42, 'y': 'foo'}
    result = echo(record_in)
    assert result.verdict is Verdict.HELD
    assert len(captured) == 1
    assert captured[0]['x'] == 42


def test_bridge_value_narrowing_via_isinstance() -> None:
    """A bridge body narrows record values at use site via
    `isinstance` — no cast, no Any. This is the framework's
    discipline for handling Record's heterogeneous values."""
    @bridge(targets=('count',))
    def positive_count(record: Mapping[str, object]) -> BridgeResult:
        v = record['count']
        if isinstance(v, int) and v > 0:
            return BridgeResult(
                verdict=Verdict.HELD, reason=f'count={v}',
                stats={'count': v}, name='', targets=(),
            )
        return BridgeResult(
            verdict=Verdict.NO_EFFECT, reason='count missing or non-positive',
            stats={}, name='', targets=(),
        )

    held: Mapping[str, object] = {'count': 7}
    refuted: Mapping[str, object] = {'count': 'not-an-int'}

    assert positive_count(held).verdict is Verdict.HELD
    assert positive_count(refuted).verdict is Verdict.NO_EFFECT


# ============ Factory pattern ============

def test_bridge_factory_pattern() -> None:
    """Factories — functions returning `Bridge` instances — are
    the framework's pattern for parameterized bridges. No special
    framework support needed; just compose `@bridge` with
    closures."""
    def monotonic_of(target: str) -> Bridge[Mapping[str, object]]:
        @bridge(targets=(target,), name=f'monotonic({target})')
        def fn(record: Mapping[str, object]) -> BridgeResult:
            return BridgeResult(
                verdict=Verdict.HELD, reason=f'ok on {target}',
                stats={}, name='', targets=(),
            )
        return fn

    b1: Bridge[Mapping[str, object]] = monotonic_of('q_mean')
    b2: Bridge[Mapping[str, object]] = monotonic_of('ep_return')
    assert b1.name == 'monotonic(q_mean)'
    assert b1.targets == ('q_mean',)
    assert b2.name == 'monotonic(ep_return)'
    assert b2.targets == ('ep_return',)
    assert b1 != b2  # different fn closures


# ============ TypedDict Record (typed-body pattern) ============

class _DQNRecord(TypedDict):
    """Test fixture — a typed record schema. In real code this
    would live in the theory module."""
    max_q_late: float
    epsilon: float
    seed: int


def test_bridge_with_typeddict_record_typed_body() -> None:
    """Bridges authored against a TypedDict get fully typed field
    access inside the body — no isinstance narrows. This is the
    payoff for parameterising Bridge[R: Mapping[str, object]]."""
    @bridge(targets=('max_q_late',))
    def max_q_decreases(record: _DQNRecord) -> BridgeResult:
        # `record['max_q_late']` is typed as `float` directly
        # because _DQNRecord declares the per-key type. No
        # isinstance() narrowing needed.
        v = record['max_q_late']
        return BridgeResult(
            verdict=Verdict.HELD if v < 100.0 else Verdict.NO_EFFECT,
            reason=f'max_q_late = {v:.3f}',
            stats={'value': v},
            name='', targets=(),
        )

    record: _DQNRecord = {
        'max_q_late': 42.0,
        'epsilon': 0.1,
        'seed': 0,
    }
    result = max_q_decreases(record)
    assert result.verdict is Verdict.HELD
    assert result.stats['value'] == 42.0


# ============ Equality ============

def test_bridge_equality_by_field() -> None:
    def f(record: Mapping[str, object]) -> BridgeResult:
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    b1 = Bridge(fn=f, name='b', targets=('x',))
    b2 = Bridge(fn=f, name='b', targets=('x',))
    assert b1 == b2
