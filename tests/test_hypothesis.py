"""Tests for `Hypothesis[R]` and HP-value canonicalization. Strict
typing exercised: bridges and hypothesis share R.

`MechanismKey` no longer exists as a framework artifact; the
configurational identity of a hypothesis is recovered from its
runs' `measurements` via `aggregate.hp_signature`. These tests
cover the data-class shape + the `_canonical_str` helper used for
HP-leaf serialization."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import BridgeResult, bridge
from corroborate.claim import claim
from corroborate.hypothesis import (
    Hypothesis,
    _canonical_str,  # pyright: ignore[reportPrivateUsage]
)
from corroborate.verdict import Verdict


# ============ Construction ============

def test_hypothesis_minimal() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='baseline', intervention={},
    )
    assert h.name == 'baseline'
    assert h.intervention == {}
    assert h.bridges == ()
    assert h.predicted_direction is None


def test_hypothesis_with_bridges() -> None:
    @bridge(targets=('x',))
    def b(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name='', targets=(),
        )

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='with_bridges',
        intervention={'slot': 'value'},
        bridges=(b,),
    )
    assert len(h.bridges) == 1
    assert h.bridges[0] is b


def test_hypothesis_predicted_direction() -> None:
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, predicted_direction='a_gt_b',
    )
    assert h.predicted_direction == 'a_gt_b'


# ============ HP-value canonicalization ============

def test_canonical_str_distinguishes_value_types() -> None:
    """Different types canonicalise distinctly so '1' (int)
    doesn't collide with '1' (str)."""
    assert _canonical_str(1) != _canonical_str('1')


def test_canonical_str_handles_claims() -> None:
    """Claim-typed values canonicalise via `Claim:<name>`."""
    @claim
    def my_alternative(x: int) -> int:
        return x * 2

    assert _canonical_str(my_alternative) == 'Claim:my_alternative'


def test_canonical_str_distinguishes_different_claims() -> None:
    @claim
    def alt_a(x: int) -> int:
        return x

    @claim
    def alt_b(x: int) -> int:
        return x

    assert _canonical_str(alt_a) != _canonical_str(alt_b)


def test_canonical_str_handles_plain_callables() -> None:
    """Plain callables canonicalise via their `__name__`."""
    def plain_alternative(x: int) -> int:
        return x

    s = _canonical_str(plain_alternative)
    assert 'plain_alternative' in s


def test_canonical_str_bool_distinct_from_int() -> None:
    """bool is a subclass of int; the canonicaliser handles them
    separately so True/False don't collide with 1/0."""
    assert _canonical_str(True) != _canonical_str(1)
    assert _canonical_str(False) != _canonical_str(0)


def test_canonical_str_partial_canonicalises_keywords() -> None:
    """`functools.partial(fn, kw=value)` canonicalises by
    recursing into `.func` and lex-encoding `.keywords`. Two
    independently-constructed partials with the same wrapped
    callable + same kwargs are equal."""
    from functools import partial

    def fn(x: int, *, kw: int = 0) -> int:
        return x + kw

    p1 = partial(fn, kw=5)
    p2 = partial(fn, kw=5)
    assert _canonical_str(p1) == _canonical_str(p2)


def test_canonical_str_partial_distinguishes_kwargs() -> None:
    """Partials with different baked kwargs canonicalise
    distinctly."""
    from functools import partial

    def fn(*, kw: int = 0) -> int:
        return kw

    p1 = partial(fn, kw=5)
    p2 = partial(fn, kw=10)
    assert _canonical_str(p1) != _canonical_str(p2)


def test_canonical_str_dataclass_field_expansion() -> None:
    """Frozen-dataclass instances canonicalise by sorted-field
    expansion."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class HP:
        a: int = 1
        b: float = 2.0

    s_default = _canonical_str(HP())
    s_changed = _canonical_str(HP(a=2))
    assert s_default != s_changed
    # Same values → same canonical string.
    assert _canonical_str(HP(a=1, b=2.0)) == _canonical_str(HP(a=1, b=2.0))


def test_canonical_str_tuple_recurses() -> None:
    """Tuples of scalars get a stable canonical form via element-
    wise recursion."""
    s = _canonical_str((1, 2, 3))
    assert s == '(1,2,3)'
