"""Claim — typed wrapper around a callable scientific claim.

Two surface shapes, ONE Protocol:

- **Free function** — `@claim def f(...)` returns a typed
  `FnClaim[P, T]` instance. `FnClaim` is a single shared frozen-
  dataclass class (NOT generated per-function). Instances carry
  `fn` and `_name` as typed fields; `invariants` is a property
  reading the side-table `_FN_INVARIANTS`, keyed by the wrapped
  fn so all wrappers of the same fn share invariants.

- **Module class** — author writes
  `class M(ClaimBase): ...`, applies `@dataclass(frozen=True,
  slots=True)` themselves, and calls `record_call` inside
  `__call__` to participate in the trace. `ClaimBase` provides
  the typed `name` property + `invariants: ClassVar` default —
  pyright sees the inherited types; no decorator mutation needed.

Both structurally satisfy `Claim[**P, T]` Protocol.

Bake-in / configuration uses `functools.partial` directly — there
is no `bind` method. The walker recognises partials and unwraps
them; mechanism_key's `_canonical_str` canonicalises partials by
recursing into `.func` and `.keywords`. v10's pattern: stdlib
idioms over invented framework primitives.

Under a `trace_context()`, each call appends a `CallRecord(claim,
args, kwargs, result)` — UNLESS any arg is a `jax.core.Tracer`
(jit/scan tracing — recording would produce meaningless
first-call records). Outside the context, calls pass through with
zero overhead — jit-safe."""
from __future__ import annotations

import contextvars
import inspect
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Protocol,
    TypeIs,
    overload,
    runtime_checkable,
)

if TYPE_CHECKING:
    from corroborate.bridge import Bridge


# ============ Protocol — the structural contract ============

@runtime_checkable
class Claim[**P, T](Protocol):
    """A callable claim with `name` and an `invariants` set.
    Structurally satisfied by `FnClaim[P, T]` instances and by
    instances of `ClaimBase`-derived classes."""
    @property
    def name(self) -> str: ...
    @property
    def invariants(self) -> tuple[Bridge[Mapping[str, object]], ...]: ...
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


def record_call(
    claim_obj: Claim[..., object],
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

    Earlier versions skipped tracer-arg calls (a "jit-safety"
    guard), which silently dropped every claim that fired inside a
    scan loop — making the full `dqn` call unprofileable. The skip
    was over-conservative: trace_context is opt-in; if a user
    enters it, they want everything recorded.

    Module authors call this inside `__call__` to participate in
    the trace. Free-function claims (`@claim` decorator) record
    automatically via `FnClaim.__call__`."""
    ctx = _TRACE.get()
    if ctx is None:
        return
    ctx.append(CallRecord(
        claim=claim_obj, args=args, kwargs=dict(kwargs), result=result,
    ))


# ============ ClaimBase — Module base class ============

class ClaimBase:
    """Base class for Module claims. Subclasses inherit `name`
    and `invariants` properties — both members of the
    `Claim[P, T]` Protocol satisfied structurally without
    decorator mutation.

    Authors write:

        @dataclass(frozen=True, slots=True)
        class MLP(ClaimBase):
            hidden: tuple[int, ...] = (64, 64)

            def init(self, ...) -> Params: ...
            def __call__(self, params, obs):
                result = ...
                record_call(self, (params, obs), {}, result)
                return result

    `_class_invariants` is the per-class storage that
    `attach_invariant` mutates; the `invariants` property reads
    `type(self)._class_invariants`. Two-level split because the
    Protocol declares `invariants` as `@property` (matching
    `FnClaim`'s side-table-backed property) — `ClassVar` storage
    paired with a property accessor satisfies both shapes."""

    _class_invariants: ClassVar[tuple['Bridge[Mapping[str, object]]', ...]] = ()

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def invariants(self) -> tuple['Bridge[Mapping[str, object]]', ...]:
        return type(self)._class_invariants


# ============ Free-function claim wrapper ============

# Side-table for free-function claim invariants. Keyed by
# underlying `fn` so all `FnClaim` wrappers of the same function
# share invariants. Module-claim invariants live as `ClassVar` on
# the class via `ClaimBase`.
_FN_INVARIANTS: dict[
    Callable[..., object],
    tuple['Bridge[Mapping[str, object]]', ...],
] = {}


def get_fn_invariants(
    fn: Callable[..., object],
) -> tuple['Bridge[Mapping[str, object]]', ...]:
    """Read free-function-claim invariants attached to `fn`.
    Typed accessor for `_FN_INVARIANTS` so consumers don't reach
    into the dict directly."""
    return _FN_INVARIANTS.get(fn, ())


def _set_fn_invariants(
    fn: Callable[..., object],
    bridges: tuple['Bridge[Mapping[str, object]]', ...],
) -> None:
    """Replace the invariants tuple for a free-function claim.
    Internal — public attach/detach use this."""
    _FN_INVARIANTS[fn] = bridges


# Memoize so `claim(f) is claim(f)` for the same `f`.
_FN_CACHE: dict[Callable[..., object], 'FnClaim[..., object]'] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FnClaim[**P, T]:
    """Typed wrapper for a free-function claim. Public — the
    concrete class consumers `isinstance`-check against.

    `fn` and `_name` are dataclass fields. `invariants` is a
    property reading `_FN_INVARIANTS` keyed by `fn`."""
    fn: Callable[P, T]
    _name: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def invariants(self) -> tuple['Bridge[Mapping[str, object]]', ...]:
        return get_fn_invariants(self.fn)

    @property
    def __signature__(self) -> inspect.Signature:
        return inspect.signature(self.fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        result = self.fn(*args, **kwargs)
        record_call(self, tuple(args), dict(kwargs), result)
        return result

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
    return getattr(module, qualname)


# ============ @claim decorator (function-only) ============

@overload
def claim[T, **P](target: Callable[P, T]) -> FnClaim[P, T]: ...


def claim(target: object) -> object:
    """Wrap a free function as a `FnClaim[P, T]`. Memoized so
    `claim(f) is claim(f)`.

    For Module classes (callable dataclass-shaped components),
    inherit `ClaimBase` directly and apply `@dataclass(frozen=
    True, slots=True)` yourself — there is no class-decorator
    path. Bake-in via `functools.partial`."""
    if isinstance(target, (FnClaim, ClaimBase)):
        # Idempotent: function-claim wrappers AND Module-claim
        # instances pass through unchanged.
        return target
    if isinstance(target, type):
        raise TypeError(
            '@claim is for free functions only. For Module-shaped '
            'claims, inherit `ClaimBase` directly: '
            '`@dataclass(frozen=True, slots=True) class M(ClaimBase): ...`',
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


def iter_invariants(
    value: object,
) -> tuple['Bridge[Mapping[str, object]]', ...]:
    """Read `value.invariants` if `value` is a Claim, else `()`.
    Typed accessor consumers use to avoid `getattr(value,
    'invariants', ())` paths that lose element types.

    Used by the walker / collector to gather composition-discovered
    invariants."""
    if isinstance(value, Claim):
        return value.invariants
    return ()


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
