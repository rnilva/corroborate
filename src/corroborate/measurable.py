"""Measurable — typed scalar derivation from a Record.

A `Measurable[T]` is a named function `(Record) -> T` that
produces a summary quantity (late-window mean, growth ratio,
threshold-margin, etc.) bridges consume. The framework's
post-hoc analytical layer.

Measurables differ from Claims and Bridges:
- Claims are STEPS in the algorithm.
- Bridges are TESTS over the algorithm's record (verdict-producing).
- Measurables are SCALAR DERIVATIONS from the record (analytical
  quantities bridges or consumers read).

For v0 the framework provides only the typed wrapper. Composition
patterns (factories, dependent measurables) are author-side
concerns expressed as ordinary Python functions returning
`Measurable[T]`. A registry / dependency-resolver lands when
graph-derivation use cases force it; v0 does not need them."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from corroborate.bridge import Record


@dataclass(frozen=True, slots=True)
class Measurable[T]:
    """Typed generic wrapper. Behaves as `Callable[[Record], T]`;
    carries `name` as a typed attribute. The return type `T` is
    preserved through `__call__` so consumers see the native
    scalar type without narrowing."""
    fn: Callable[[Record], T]
    name: str

    def __call__(self, record: Record) -> T:
        return self.fn(record)


def measurable[T](
    *,
    name: str | None = None,
) -> Callable[[Callable[[Record], T]], Measurable[T]]:
    """Decorator factory wrapping a `(Record) -> T` function in a
    typed `Measurable[T]`. `name` defaults to `fn.__name__` when
    omitted.

    Usage:

        @measurable()
        def max_q_late_mean(record: Record) -> float:
            ...

    Or as a factory closure:

        def late_window_mean(target: str) -> Measurable[float]:
            @measurable(name=f'{target}_late_mean')
            def fn(record: Record) -> float:
                ...
            return fn"""
    def decorator(fn: Callable[[Record], T]) -> Measurable[T]:
        resolved_name = name if name is not None else fn.__name__
        return Measurable(fn=fn, name=resolved_name)
    return decorator
