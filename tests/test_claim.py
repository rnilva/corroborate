"""Tests for `Claim` Protocol, `claim` decorator, `trace_context`,
`is_claim`, and `CallRecord`.

Verifies:
- `@claim` on a free function returns a Protocol-conforming wrapper.
- `@claim` on a class applies dataclass(frozen, slots) + adds default
  `name` property + wraps `__call__` for trace participation.
- `@claim` on a method records the bound instance to the trace.
- `CallRecord` captures (claim, args, kwargs, result) per call.
- `is_claim` narrows structurally (Claim Protocol).
- Trace contexts isolate properly + restore on exception."""
from __future__ import annotations

from dataclasses import dataclass

from corroborate.claim import (
    CallRecord,
    Claim,
    ClaimBase,
    claim,
    is_claim,
    record_call,
    trace_context,
)


# ============ Free-function decoration ============

def test_claim_returns_callable_with_native_signature() -> None:
    @claim
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert add.name == 'add'
    # @claim on a function returns a singleton instance of an
    # auto-generated frozen-dataclass class. `invariants` ClassVar
    # exists, starts empty.
    assert add.invariants == ()


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


# ============ Module class via @claim ============

def test_module_via_claimbase_inheritance() -> None:
    """Module-claim authoring: inherit `ClaimBase` + apply
    `@dataclass(frozen=True, slots=True)`. Inherited `name`
    property + `invariants` ClassVar default; no decorator
    mutation."""
    @dataclass(frozen=True, slots=True)
    class Doubler(ClaimBase):
        factor: int = 2

        def __call__(self, x: int) -> int:
            return x * self.factor

    d = Doubler()
    assert d(5) == 10
    assert d.name == 'Doubler'
    assert d.invariants == ()

    # frozen — assignment fails.
    try:
        d.factor = 3  # pyright: ignore[reportAttributeAccessIssue]
        raise AssertionError('expected FrozenInstanceError')
    except Exception:
        pass


def test_module_records_self_to_trace_via_record_call() -> None:
    """Module authors call `record_call` inside `__call__` to
    participate in the trace — explicit, no decorator magic."""
    @dataclass(frozen=True, slots=True)
    class Doubler(ClaimBase):
        factor: int = 3

        def __call__(self, x: int) -> int:
            result = x * self.factor
            record_call(self, (x,), {}, result)
            return result

    d = Doubler()
    with trace_context() as records:
        _ = d(4)

    assert len(records) == 1
    assert records[0].claim is d
    assert records[0].claim.name == 'Doubler'
    assert records[0].args == (4,)
    assert records[0].result == 12


def test_module_satisfies_claim_protocol() -> None:
    @dataclass(frozen=True, slots=True)
    class Identity(ClaimBase):
        def __call__(self, x: int) -> int:
            return x

    items: list[Claim[..., object]] = [Identity()]
    assert items[0].name == 'Identity'


# ============ Manual @dataclass + record_call escape hatch ============

def test_manual_dataclass_with_record_call() -> None:
    """Authors who want explicit @dataclass control (e.g. custom
    options) opt out of `@claim` on the class and use `record_call`
    to participate in the trace explicitly."""
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


# ============ is_claim TypeIs narrowing ============

def test_is_claim_true_for_decorated_function() -> None:
    @claim
    def f() -> None:
        return None

    assert is_claim(f) is True


def test_is_claim_true_for_module_instance() -> None:
    @dataclass(frozen=True, slots=True)
    class M(ClaimBase):
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


def test_invariant_attaches_to_claim_class() -> None:
    """`@invariant(of=claim)` mutates `type(claim).invariants` so
    composition-level discovery (`signature.collect_invariants`) can
    surface it."""
    from collections.abc import Mapping

    from corroborate.bridge import BridgeResult
    from corroborate.invariant import invariant
    from corroborate.verdict import Verdict

    @claim
    def my_step(x: int) -> int:
        return x + 1

    assert my_step.invariants == ()

    @invariant(of=my_step, targets=('x',))
    def some_invariant(record: Mapping[str, int]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok', stats={},
            name='', targets=(),
        )

    # After @invariant, the bridge is appended to the claim's
    # class-level `invariants` ClassVar — visible via the instance.
    assert len(my_step.invariants) == 1
    assert 'some_invariant' in my_step.invariants[0].name


def test_invariant_attaches_to_module_class_via_instance() -> None:
    """`@invariant(of=...)` attaches at the class level. Authors
    pass a Module INSTANCE — its class carries the invariants
    ClassVar, so all instances see it."""
    from collections.abc import Mapping

    from corroborate.bridge import BridgeResult
    from corroborate.invariant import invariant
    from corroborate.verdict import Verdict

    @dataclass(frozen=True, slots=True)
    class Tripler(ClaimBase):
        factor: int = 3

        def __call__(self, x: int) -> int:
            return x * self.factor

    canonical = Tripler()  # default-instantiate to attach class-level invariant

    @invariant(of=canonical, targets=('x',))
    def tripler_invariant(record: Mapping[str, int]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok', stats={},
            name='', targets=(),
        )

    t = Tripler()
    assert len(t.invariants) == 1
    # Other instances of the same class see the SAME class-level invariants.
    other = Tripler(factor=5)
    assert other.invariants == t.invariants


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
    """`@claim` is for free functions only. Module-claims inherit
    `ClaimBase` directly + apply `@dataclass`."""
    class NotAClaim:
        def __call__(self) -> None: ...

    try:
        _ = claim(NotAClaim)  # pyright: ignore[reportCallIssue]
        raise AssertionError('expected TypeError')
    except TypeError as e:
        assert 'free functions only' in str(e)


def test_claim_idempotent_on_module_instance() -> None:
    """`claim(module_instance)` is idempotent — Module instances
    structurally satisfy Claim Protocol, so the function-claim
    branch is short-circuited."""
    @dataclass(frozen=True, slots=True)
    class M(ClaimBase):
        def __call__(self, x: int) -> int:
            return x

    instance = M()
    re_wrapped = claim(instance)  # pyright: ignore[reportCallIssue]
    assert re_wrapped is instance


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

    from corroborate.signature import walk

    @claim
    def schedule(step: int, *, anneal: int = 10_000) -> float:
        return float(step) / anneal

    baked = partial(schedule, anneal=50_000)
    sig = walk(baked)
    # `anneal` field's default is overlaid by the bake.
    by_name = {kw.name: kw for kw in sig.kwargs}
    assert by_name['anneal'].default == 50_000


# ============ Pickle round-trip ============

def test_pickle_round_trip_function_claim() -> None:
    """`_FnClaim` is a single shared frozen-dataclass class living
    at module-scope in `corroborate.claim`; pickle finds it via
    standard module:qualname lookup. The wrapped `fn` must also
    be module-level (which the substrate's `@claim`'d functions
    are)."""
    import pickle
    from corroborate.rl.dqn.claims.bootstrap import bootstrap as vanilla_bootstrap

    blob = pickle.dumps(vanilla_bootstrap)
    restored = pickle.loads(blob)
    # Memoization gives us singleton identity.
    assert restored is vanilla_bootstrap


def test_pickle_round_trip_partial_over_claim() -> None:
    """`functools.partial` over a claim pickles natively."""
    import pickle
    from functools import partial
    from corroborate.rl.dqn.claims.bootstrap import bootstrap as vanilla_bootstrap

    baked = partial(vanilla_bootstrap, gamma=0.95)
    blob = pickle.dumps(baked)
    restored = pickle.loads(blob)
    assert restored.func is vanilla_bootstrap
    assert restored.keywords == {'gamma': 0.95}


# ============ JIT silence (#5) ============

def test_trace_records_under_jit_for_structural_extraction() -> None:
    """When @claim'd functions run inside `jax.jit` / `lax.scan` /
    `vmap`, args are Tracer objects. Recording fires anyway —
    that's exactly the structural information `computation_graph.
    build_*` extracts (which Claim called which during the tracing
    pass). Earlier versions skipped tracer-arg calls; that
    silently dropped every claim inside a scan loop, making the
    full `dqn` (which uses scan) unprofileable."""
    import jax
    import jax.numpy as jnp

    @claim
    def double(x: jax.Array) -> jax.Array:
        return x * 2

    with trace_context() as records:
        # Inside jit the arg is a Tracer → still records (one call
        # per tracing pass).
        result = jax.jit(double)(jnp.float32(3.0))
        # Concrete call outside jit → records.
        _ = double(jnp.float32(4.0))

    assert float(result) == 6.0
    # Both calls record: one from the jit tracing pass + one from
    # the concrete eager call.
    assert len(records) == 2
    assert all(r.claim is double for r in records)


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
