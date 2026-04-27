"""Tests for `Claim`, `claim`, `TraceContext`, and `is_claim`.

Property-and-type level: that the wrapper is signature-preserving,
trace records what fired, outside-context calls are pass-through,
the TypeIs narrowing works, and the Protocol-based trace boundary
holds heterogeneous Claim types together."""
from __future__ import annotations

from corroborate.claim import Claim, ClaimRecord, TraceContext, claim, is_claim


# ============ Signature preservation ============

def test_claim_returns_claim_instance() -> None:
    @claim
    def add(a: int, b: int) -> int:
        return a + b

    assert isinstance(add, Claim)
    assert add.name == 'add'


def test_claim_is_callable_with_native_signature() -> None:
    @claim
    def double(x: int) -> int:
        return x * 2

    result = double(21)
    assert result == 42


def test_claim_preserves_name_via_fn_dunder() -> None:
    @claim
    def some_specific_function() -> None:
        return None

    assert some_specific_function.name == 'some_specific_function'


# ============ TraceContext ============

def test_outside_context_no_tracing() -> None:
    """Calling a Claim outside any TraceContext is a pass-through;
    no records accumulate anywhere."""
    @claim
    def inc(x: int) -> int:
        return x + 1

    result = inc(5)
    assert result == 6
    # No assertion on records — there's no context to check; the
    # property is "no error and result returned correctly."


def test_inside_context_records_calls() -> None:
    @claim
    def f(x: int) -> int:
        return x

    @claim
    def g(x: int) -> int:
        return x

    with TraceContext() as trace:
        _ = f(1)
        _ = g(2)
        _ = f(3)

    assert len(trace.records) == 3
    # Records are Claim instances themselves
    assert isinstance(trace.records[0], Claim)
    assert trace.records[0].name == 'f'
    assert trace.records[1].name == 'g'
    assert trace.records[2].name == 'f'


def test_records_are_claim_instances_after_isinstance_narrow() -> None:
    """After isinstance narrowing, the Claim's `.fn` is accessible
    typed (the test exercises pyright's narrowing path)."""
    @claim
    def h(x: int) -> str:
        return str(x)

    with TraceContext() as trace:
        _ = h(7)

    rec = trace.records[0]
    assert isinstance(rec, Claim)
    # rec is now Claim[Unknown, Unknown] under pyright; .fn is
    # accessible as a Callable, .name as str.
    assert rec.name == 'h'


def test_nested_contexts_isolate_records() -> None:
    """An inner TraceContext receives its own calls; the outer
    context only sees what it itself collected before entering
    the inner block."""
    @claim
    def f() -> None:
        return None

    with TraceContext() as outer:
        f()
        with TraceContext() as inner:
            f()
            f()
        f()

    assert len(outer.records) == 2  # 1 before + 1 after inner
    assert len(inner.records) == 2


def test_context_restores_on_exception() -> None:
    """If an exception escapes the with-block, the contextvar still
    resets so subsequent calls are pass-through."""
    @claim
    def f() -> None:
        return None

    try:
        with TraceContext():
            f()
            raise RuntimeError('boom')
    except RuntimeError:
        pass

    # After the exception, no context — call should not raise or
    # leak into a stale records list. The fact that this completes
    # without error is the assertion.
    f()


# ============ is_claim TypeIs narrowing ============

def test_is_claim_true_for_decorated() -> None:
    @claim
    def f() -> None:
        return None

    assert is_claim(f) is True


def test_is_claim_false_for_plain_function() -> None:
    def plain() -> None:
        return None

    assert is_claim(plain) is False


def test_is_claim_false_for_non_callable() -> None:
    assert is_claim(42) is False
    assert is_claim('string') is False
    assert is_claim(None) is False


def test_is_claim_narrows_to_claim_record() -> None:
    """is_claim is a TypeIs[ClaimRecord]; in the True branch, the
    object is narrowed to a structural ClaimRecord (typed `name`).
    This test exercises that ClaimRecord's `name` is accessible
    without further narrowing."""
    @claim
    def f() -> None:
        return None

    obj: object = f
    if is_claim(obj):
        # obj narrowed to ClaimRecord here
        assert isinstance(obj.name, str)
    else:
        # complement branch
        raise AssertionError('expected obj to be a Claim')


# ============ Equality + hashability (frozen dataclass invariants) ============

def test_claim_equality_by_field() -> None:
    def f() -> None:
        return None

    c1 = Claim(fn=f, name='f')
    c2 = Claim(fn=f, name='f')
    assert c1 == c2


def test_claim_inequality_on_different_fn() -> None:
    def f1() -> None:
        return None

    def f2() -> None:
        return None

    c1 = Claim(fn=f1, name='same')
    c2 = Claim(fn=f2, name='same')
    assert c1 != c2


# ============ ClaimRecord Protocol satisfaction ============

def test_claim_satisfies_claim_record_protocol() -> None:
    """A Claim instance flows into a list[ClaimRecord] without
    a cast — verifies the Protocol-based trace boundary."""
    @claim
    def f() -> None:
        return None

    records: list[ClaimRecord] = [f]
    assert records[0].name == 'f'
