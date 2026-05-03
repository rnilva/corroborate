"""Claim — typed wrapper around a callable scientific claim.

`@claim def f(...)` returns a typed `FnClaim[P, T]` instance.
`FnClaim` is a single shared frozen-dataclass class (NOT
generated per-function). Instances carry `fn` and `_name` as
typed fields; `invariants` is a property reading the side-table
`_FN_INVARIANTS`, keyed by the wrapped fn so all wrappers of the
same fn share invariants. Configuration uses `functools.partial`
directly — there is no `bind` method. The walker recognises
partials and unwraps them; `canonical_str` canonicalises partials
by recursing into `.func` and `.keywords`. v10's pattern: stdlib
idioms over invented framework primitives.

`Claim[**P, T]` is the structural Protocol — anything with a
`name: str` property and a `__call__` matches. `FnClaim` is the
sole built-in shape; substrate authors who need a class-based
Claim with stateful `__call__` write a frozen dataclass exposing
`name` and call `record_call(self, args, kwargs, result)` inside
`__call__` (escape hatch — see `tests/test_claim.py` for the
canonical example). Most substrate authoring goes through
`@claim` on free functions plus frozen-dataclass config bundles
that delegate to Free Claims; the escape hatch is genuinely
rare.

Under a `trace_context()`, each call appends a `CallRecord(claim,
args, kwargs, result)`. JIT/scan tracing fires each `@claim` call
once with abstract tracer values; that single pass IS the
structural information the graph extractor wants. Outside the
context, calls pass through with zero overhead — jit-safe."""
from __future__ import annotations

import contextvars
import inspect
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Protocol,
    TypeIs,
    overload,
    override,
    runtime_checkable,
)

from corroborate._introspection_boundary import get_attr_obj

# ============ Protocol — the structural contract ============

@runtime_checkable
class Claim[**P, T](Protocol):
    """A callable claim with a `name`. Structurally satisfied by
    `FnClaim[P, T]` instances and by any class-based Claim using
    the `record_call` escape hatch (a frozen dataclass with
    `name: str` and `__call__`)."""
    @property
    def name(self) -> str: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


# ============ Trace records ============

@dataclass(frozen=True, slots=True)
class CallRecord:
    """One Claim invocation captured by the active trace.

    `args`, `kwargs`, `result` are object-erased — heterogeneity
    is intrinsic at the trace boundary (per CLAUDE.md's
    polymorphism carve-out). Tracer args are NOT recorded — see
    `_is_tracing`."""
    claim: Claim[..., object]
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    result: object


_TRACE: contextvars.ContextVar[list[CallRecord] | None] = (
    contextvars.ContextVar('_corroborate_trace', default=None)
)


def record_call[**P, T](
    claim_obj: Claim[P, T],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    result: object,
) -> None:
    """Append a CallRecord to the active trace.

    Records **unconditionally** when a `trace_context()` is active —
    including under `jax.jit` / `lax.scan` / `vmap`'s tracing pass.
    The tracing pass fires each `@claim` call once with abstract
    tracer values; that single pass IS the structural information
    the graph extractor wants (`computation_graph.build_*`). After
    the trace pass exits XLA-compiled execution doesn't go through
    Python, so no further records accumulate.

    Free-function claims (`@claim` decorator) record automatically
    via `FnClaim.__call__`. Class-based Claims using the escape
    hatch call this inside `__call__` to participate in the trace
    explicitly."""
    ctx = _TRACE.get()
    if ctx is None:
        return
    ctx.append(CallRecord(
        claim=claim_obj, args=args, kwargs=dict(kwargs), result=result,
    ))


# ============ Free-function claim wrapper ============

# Memoize so `claim(f) is claim(f)` for the same `f`.
_FN_CACHE: dict[Callable[..., object], 'FnClaim[..., object]'] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FnClaim[**P, T]:
    """Typed wrapper for a free-function claim. Public — the
    concrete class consumers `isinstance`-check against.

    `fn` and `_name` are dataclass fields."""
    fn: Callable[P, T]
    _name: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self.fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        result = self.fn(*args, **kwargs)
        record_call(self, tuple(args), dict(kwargs), result)
        return result

    @override
    def __reduce__(self) -> tuple[Callable[..., object], tuple[str, str]]:
        """Pickle reconstruction. `@claim` rebinds the module
        attribute to this wrapper, so we look up the wrapper at
        the wrapped fn's `__module__:__qualname__`. Fails loudly
        for nested-function claims (`<locals>` in qualname) — they
        aren't reachable via module-attribute lookup, period."""
        qualname = self.fn.__qualname__
        if '<locals>' in qualname:
            raise PicklingError_for_nested(self._name)
        return (
            _unpickle_fn_claim,
            (self.fn.__module__, qualname),
        )


def PicklingError_for_nested(name: str) -> Exception:
    """Helper: produce a clear PicklingError when a nested-function
    claim is asked to pickle. Lazy import to avoid the typing
    cost of pickle.PicklingError on import."""
    import pickle
    return pickle.PicklingError(
        f"Cannot pickle FnClaim for {name!r} — defined inside a "
        f"function (qualname has '<locals>'). Move to module scope."
    )


def _unpickle_fn_claim(module_name: str, qualname: str) -> object:
    """Pickle reconstruction. Imports the module and looks up the
    wrapper at `qualname` (which `@claim`'s side-effect rebinding
    placed there)."""
    import importlib
    module = importlib.import_module(module_name)
    # `getattr` on a module returns `Any`; the runtime invariant is
    # that `qualname` resolves to the `@claim`-rebound `FnClaim`,
    # but the framework's typed surface is `object` here (pickle
    # reconstruction is intentionally erased).
    return get_attr_obj(module, qualname)


# ============ @claim decorator (function-only) ============

@overload
def claim[T, **P](target: Callable[P, T]) -> FnClaim[P, T]: ...
@overload
def claim(target: object) -> object: ...


def claim(target: object) -> object:
    """Wrap a free function as a `FnClaim[P, T]`. Memoized so
    `claim(f) is claim(f)`.

    `@claim` decorates free functions only — there is no class-
    decorator path. Substrate authors who need a class-based
    Claim (rare) write a frozen dataclass exposing `name: str`
    and call `record_call(self, args, kwargs, result)` inside
    `__call__`. Bake-in via `functools.partial`.

    The first overload is the typed primary path (`@claim` on a
    Callable produces a typed `FnClaim[P, T]`). The second is the
    catch-all for runtime defensive checks — passes `FnClaim`
    instances through idempotently, and raises `TypeError` on
    classes / non-callables."""
    if isinstance(target, FnClaim):
        # Idempotent: function-claim wrappers pass through unchanged.
        return target
    if isinstance(target, type):
        raise TypeError(
            '@claim is for free functions only. For class-based '
            'Claims, write a frozen dataclass with `name: str` '
            'and call `record_call(self, args, kwargs, result)` '
            'inside `__call__`.',
        )
    if callable(target):
        cached = _FN_CACHE.get(target)
        if cached is not None:
            return cached
        name = getattr(target, '__name__', '<anonymous>')
        if not isinstance(name, str):
            name = '<anonymous>'
        wrapper = FnClaim[..., object](fn=target, _name=name)
        _FN_CACHE[target] = wrapper
        return wrapper
    raise TypeError(
        f'@claim accepts a callable; got {type(target).__name__}',
    )


# ============ Predicates ============

def is_claim(obj: object) -> TypeIs[Claim[..., object]]:
    """`PEP 742 TypeIs` — narrows `obj` to `Claim[..., object]`
    in the True branch."""
    return isinstance(obj, Claim)


# ============ Trace context manager ============

@contextmanager
def trace_context() -> Generator[list[CallRecord]]:
    """Collect `CallRecord`s for a probe run.

    JIT/scan: Tracer args are detected and recording is skipped,
    so JIT boundaries are silent. Use `python_loop` instead of
    `scan_loop` for probe runs that need full trace coverage."""
    records: list[CallRecord] = []
    token = _TRACE.set(records)
    try:
        yield records
    finally:
        _TRACE.reset(token)
