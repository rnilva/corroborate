"""Claim — typed generic wrapper around a function.

A scientific claim is an honest, executable program. The `@claim`
decorator wraps a function in a `Claim[**P, T]` dataclass instance
— itself `Callable[P, T]` via `__call__` — so callers see the
wrapper as a drop-in replacement for the underlying function.
Type information flows end-to-end through the ParamSpec `**P` and
return type `T`; no `getattr`/`setattr` paths, no marker
attributes, no type erasure.

Three contracts:

1. *Substitutability.* `Claim[P, T]` is `Callable[P, T]`. Where
   the original function went, the Claim goes. `partial(theory,
   slot=alternative_claim)` is just a value swap.
2. *Introspectability.* Each Claim instance carries typed `fn`
   and `name` attributes; statically resolved, no dynamic lookup.
   `isinstance(obj, Claim)` is the typed check (pyright narrows).
3. *Author intent.* The function's docstring is the paper-prose
   explanation visible to bridge authors and downstream readers.
   Per program-honesty discipline: every technique is an explicit
   Claim artifact; step functions read like paper-prose.

Under a `TraceContext`, each `Claim` invocation appends `self` to
a context-local list, so the functional-claim graph (PAPER_NOTES.
md §1.1, graph (a)) can be derived by enumerating which claims
fired during a probe run. Outside any context, calls pass through
with zero overhead — jit-safe."""
from __future__ import annotations

import contextvars
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, TypeIs


class ClaimRecord(Protocol):
    """Structural Protocol for what the trace stores at its
    container boundary. Concrete `Claim[**P, T]` instances
    satisfy this Protocol structurally — they all expose a
    read-only `name` accessor. The Protocol does NOT reference
    `P` or `T`, sidestepping invariance constraints that would
    otherwise block storing `Claim[(int,), str]` in a list typed
    for `Claim[..., object]`.

    `name` is declared as a read-only property so frozen-dataclass
    fields satisfy the Protocol (Protocol field defaults to
    writable; frozen fields aren't).

    Trace consumers that need the original `(P, T)` signature
    narrow via `isinstance(rec, Claim)`, then access `rec.fn`
    directly with its native `Callable[P, T]` type."""
    @property
    def name(self) -> str: ...


_TRACE: contextvars.ContextVar[list[ClaimRecord] | None] = (
    contextvars.ContextVar('_corroborate_trace', default=None)
)


@dataclass(frozen=True, slots=True)
class Claim[**P, T]:
    """Typed generic wrapper. Behaves as `Callable[P, T]`; carries
    `fn` and `name` as typed attributes.

    Structurally satisfies `ClaimRecord` — instances flow into the
    trace's `list[ClaimRecord]` without parameterization-variance
    issues. Pyright narrows `isinstance(rec, Claim)` back to
    `Claim[Unknown, Unknown]`; direct field access on `Claim[P, T]`
    instances at concrete-typed call sites preserves the native
    `(P, T)` signature.

    Equality is by-field (frozen dataclass default), so two Claims
    wrapping the same function with the same name compare equal —
    useful for trace-record deduplication."""
    fn: Callable[P, T]
    name: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        result = self.fn(*args, **kwargs)
        ctx = _TRACE.get()
        if ctx is not None:
            ctx.append(self)
        return result


def claim[**P, T](fn: Callable[P, T]) -> Claim[P, T]:
    """Wrap `fn` in a typed `Claim[P, T]`. The returned object IS
    `Callable[P, T]` — usable wherever the original function was.

    Idempotency is by construction: `claim(claim(fn))` produces
    `Claim[P, T](fn=Claim[P, T](fn=fn, name=...), name=...)`. The
    inner Claim is callable, so the outer wrapper still works, but
    callers should not double-wrap. Use `is_claim(obj)` to detect
    already-wrapped claims if the situation calls for it — it
    narrows correctly for pyright via PEP 742 `TypeIs`."""
    return Claim(fn=fn, name=fn.__name__)


def is_claim(obj: object) -> TypeIs[ClaimRecord]:
    """`PEP 742 TypeIs`-narrowing predicate. Returns True iff `obj`
    is a `Claim[..., ...]` instance; pyright then narrows `obj` to
    `ClaimRecord` (read-only `name`) in the True branch and the
    complement in the False branch — without `cast`.

    Concrete users who need the full `Claim[P, T]` shape after
    narrowing should `isinstance(obj, Claim)` directly; pyright
    narrows that to `Claim[Unknown, Unknown]`, and field access on
    `.fn` recovers the native `Callable[P, T]` at the call site."""
    return isinstance(obj, Claim)


@contextmanager
def trace_context() -> Generator[list[ClaimRecord]]:
    """Context manager collecting `Claim` invocations for a probe
    run. Yields a `list[ClaimRecord]` that's appended to as
    `@claim`'d functions run inside the with-block. After exit,
    the contextvar resets so subsequent calls don't append.

    Not usable inside `jax.jit`-compiled code: the contextvar is
    read at concrete-call time, which jit elides.

    Usage:

        with trace_context() as records:
            f()
            g()
        # records contains [Claim<f>, Claim<g>] in completion order
    """
    records: list[ClaimRecord] = []
    token = _TRACE.set(records)
    try:
        yield records
    finally:
        _TRACE.reset(token)
