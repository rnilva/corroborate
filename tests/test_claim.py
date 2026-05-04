"""Tests for `Claim` Protocol, `claim` decorator, `trace_context`,
`is_claim`, and `CallRecord`.

Verifies:
- `@claim` on a free function returns a Protocol-conforming wrapper.
- The manual-dataclass + `record_call` escape hatch satisfies
  the Claim Protocol structurally and participates in the trace.
- `CallRecord` captures (claim, args, kwargs, result) per call.
- `is_claim` narrows structurally (Claim Protocol).
- Trace contexts isolate properly + restore on exception."""
from __future__ import annotations

from dataclasses import dataclass

from corroborate.core.claim import (
    CallRecord,
    Claim,
    claim,
    is_claim,
    record_call,
    trace_context,
)


@claim
def _module_level_claim(a: int, b: int = 0) -> int:
    """Module-level `@claim` so pickle's module:qualname lookup
    can find both the wrapper and the wrapped fn. Emulates a
    substrate-authored claim without depending on a substrate."""
    return a + b


# ============ Free-function decoration ============

def test_claim_returns_callable_with_native_signature() -> None:
    @claim
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add.name == 'add'


def test_claim_preserves_name() -> None:
    @claim
    def some_specific_function() -> None:
        return None

    assert some_specific_function.name == 'some_specific_function'


def test_claim_satisfies_protocol() -> None:
    """Decorated function structurally satisfies the Claim Protocol —
    list[Claim[..., object]] accepts it without cast."""
    @claim
    def f() -> None:
        return None

    items: list[Claim[..., object]] = [f]
    assert items[0].name == 'f'


# ============ trace_context with output recording ============

def test_outside_context_no_tracing() -> None:
    @claim
    def inc(x: int) -> int:
        return x + 1

    # No active trace; call passes through cleanly.
    assert inc(5) == 6


def test_inside_context_records_call() -> None:
    @claim
    def f(x: int) -> int:
        return x * 2

    with trace_context() as records:
        result = f(7)

    assert result == 14
    assert len(records) == 1
    assert isinstance(records[0], CallRecord)
    assert records[0].claim.name == 'f'
    assert records[0].args == (7,)
    assert records[0].kwargs == {}
    assert records[0].result == 14


def test_records_capture_kwargs_and_outputs() -> None:
    """CallRecord captures (claim, args, kwargs, result) — v10's
    output-tracing contract."""
    @claim
    def add(a: int, *, b: int) -> int:
        return a + b

    with trace_context() as records:
        _ = add(1, b=2)
        _ = add(10, b=20)

    assert len(records) == 2
    assert records[0].args == (1,)
    assert records[0].kwargs == {'b': 2}
    assert records[0].result == 3
    assert records[1].args == (10,)
    assert records[1].kwargs == {'b': 20}
    assert records[1].result == 30


def test_nested_contexts_isolate_records() -> None:
    @claim
    def f() -> None:
        return None

    with trace_context() as outer:
        f()
        with trace_context() as inner:
            f()
            f()
        f()

    assert len(outer) == 2
    assert len(inner) == 2


def test_context_restores_on_exception() -> None:
    @claim
    def f() -> None:
        return None

    try:
        with trace_context():
            f()
            raise RuntimeError('boom')
    except RuntimeError:
        pass

    # Subsequent calls are pass-through; no error from a stale ctx.
    f()


# ============ Manual @dataclass + record_call escape hatch ============

def test_manual_dataclass_with_record_call() -> None:
    """Substrate authors who need a class-based Claim write a
    frozen dataclass exposing `name: str` and call `record_call`
    inside `__call__` to participate in the trace explicitly.
    This is the canonical escape hatch — no framework base
    class, just stdlib `@dataclass` plus the structural Claim
    Protocol."""
    @dataclass(frozen=True, slots=True)
    class Tripler:
        factor: int = 3

        @property
        def name(self) -> str:
            return f'Tripler(factor={self.factor})'

        def __call__(self, x: int) -> int:
            result = x * self.factor
            record_call(self, (x,), {}, result)
            return result

    t = Tripler()
    assert t(4) == 12

    with trace_context() as records:
        _ = t(5)

    assert len(records) == 1
    assert records[0].claim is t
    assert records[0].claim.name == 'Tripler(factor=3)'
    assert records[0].args == (5,)
    assert records[0].result == 15


def test_manual_dataclass_satisfies_claim_protocol() -> None:
    """The structural Claim Protocol accepts any object with
    `name: str` and `__call__`. No inheritance required."""
    @dataclass(frozen=True, slots=True)
    class Identity:
        @property
        def name(self) -> str:
            return 'Identity'

        def __call__(self, x: int) -> int:
            return x

    items: list[Claim[..., object]] = [Identity()]
    assert items[0].name == 'Identity'


def test_manual_dataclass_is_frozen() -> None:
    """Frozen-dataclass discipline: post-construction mutation
    fails. Substrate authors using the escape hatch get the
    same immutability guarantees as `@claim`-wrapped Free
    Claims."""
    @dataclass(frozen=True, slots=True)
    class Doubler:
        factor: int = 2

        @property
        def name(self) -> str:
            return 'Doubler'

        def __call__(self, x: int) -> int:
            return x * self.factor

    d = Doubler()
    assert d(5) == 10

    try:
        d.factor = 3  # pyright: ignore[reportAttributeAccessIssue]
        raise AssertionError('expected FrozenInstanceError')
    except Exception:
        pass


# ============ is_claim TypeIs narrowing ============

def test_is_claim_true_for_decorated_function() -> None:
    @claim
    def f() -> None:
        return None

    assert is_claim(f) is True


def test_is_claim_true_for_manual_dataclass_instance() -> None:
    @dataclass(frozen=True, slots=True)
    class M:
        @property
        def name(self) -> str:
            return 'M'

        def __call__(self) -> None:
            return None

    assert is_claim(M()) is True


def test_is_claim_false_for_plain_function() -> None:
    """Plain functions don't have a `name` attribute (only
    `__name__`); not Claim-shaped."""
    def plain() -> None:
        return None

    assert is_claim(plain) is False


def test_is_claim_false_for_non_callable() -> None:
    assert is_claim(42) is False
    assert is_claim('string') is False
    assert is_claim(None) is False


def test_claim_is_idempotent_on_already_decorated_function() -> None:
    """`claim(claim(f))` returns the inner wrapper unchanged —
    no double-wrap. Lets intervention dicts carry pre-wrapped
    values without producing nested `_FnClaim(fn=_FnClaim(...))`."""
    @claim
    def f(x: int) -> int:
        return x + 1

    again = claim(f)
    assert again is f


def test_claim_rejects_class_input() -> None:
    """`@claim` is for free functions only. Class-based Claims
    use the manual-dataclass + `record_call` escape hatch (no
    decorator on the class)."""
    class NotAClaim:
        def __call__(self) -> None: ...

    try:
        _ = claim(NotAClaim)
        raise AssertionError('expected TypeError')
    except TypeError as e:
        assert 'free functions only' in str(e)


# ============ Bake-in via functools.partial ============

def test_partial_over_function_claim_is_callable() -> None:
    """Bake-in pattern: `functools.partial(claim, **kwargs)` —
    no `bind` method, just stdlib partial."""
    from functools import partial

    @claim
    def add(a: int, b: int = 0) -> int:
        return a + b

    baked = partial(add, b=10)
    assert baked(5) == 15
    assert baked(2) == 12


def test_partial_signature_overlay_via_walker() -> None:
    """Walker recognises `partial(claim, **kwargs)` and overlays
    the baked values on the wrapped fn's signature so HP discovery
    sees the post-bake defaults."""
    from functools import partial

    from corroborate.core.signature import walk

    @claim
    def schedule(step: int, *, anneal: int = 10_000) -> float:
        return float(step) / anneal

    baked = partial(schedule, anneal=50_000)
    sig = walk(baked)
    # `anneal` field's default is overlaid by the bake.
    by_name = {kw.name: kw for kw in sig.kwargs}
    assert by_name['anneal'].default == 50_000


# ============ Pickle round-trip (framework-level) ============

def test_pickle_round_trip_function_claim() -> None:
    """`_FnClaim` is a single shared frozen-dataclass class living
    at module-scope in `corroborate.core.claim`; pickle finds it
    via standard module:qualname lookup. The wrapped `fn` must
    also be module-level — emulated here with a module-scope
    `_module_level_claim` so the test stays substrate-free.
    Substrate-side pickle round-trip on a real DQN claim lives in
    `src/corroborate_rl/tests/test_claim_substrate_integration.py`."""
    import pickle

    blob = pickle.dumps(_module_level_claim)
    restored = pickle.loads(blob)
    # Memoization gives us singleton identity.
    assert restored is _module_level_claim


def test_pickle_round_trip_partial_over_claim() -> None:
    """`functools.partial` over a claim pickles natively."""
    import pickle
    from functools import partial

    baked = partial(_module_level_claim, b=10)
    blob = pickle.dumps(baked)
    restored = pickle.loads(blob)
    assert restored.func is _module_level_claim
    assert restored.keywords == {'b': 10}


def test_is_claim_narrows_type() -> None:
    """is_claim is `TypeIs[Claim[..., object]]`; in the True branch,
    pyright narrows so `obj.name` is accessible without further cast."""
    @claim
    def f() -> None:
        return None

    obj: object = f
    if is_claim(obj):
        assert isinstance(obj.name, str)
    else:
        raise AssertionError('expected obj to be a Claim')
