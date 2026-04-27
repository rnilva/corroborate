"""Measurable — typed scalar derivation from a record.

A `Measurable[R, T]` is a named function `(R) -> T` that produces
a summary quantity (late-window mean, growth ratio, threshold
margin, etc.) bridges or consumers read. Two type parameters:

- `R: Mapping[str, object]` — the record schema. Author can use
  plain Mapping, TypedDict, or a custom Mapping subclass.
- `T` — the scalar return type (float, int, bool, str, jax.Array,
  etc.). Preserved through `__call__` so consumers see the native
  scalar type without narrowing.

The framework's post-hoc analytical layer. Distinct from Claims
(steps in the algorithm) and Bridges (verdict-producing tests).

For v0 the framework provides only the typed wrapper. Composition
patterns (factories, dependent measurables) are author-side
concerns expressed as ordinary Python functions returning
`Measurable[R, T]`. A registry / dependency-resolver lands when
graph-derivation use cases force it; v0 doesn't need them
(framework-subtraction discipline)."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True, slots=True)
class Measurable[R: Mapping[str, object], T]:
    """Typed generic wrapper. Behaves as `Callable[[R], T]`;
    carries `name` as a typed attribute. Both `R` (record type)
    and `T` (scalar return type) preserved through `__call__`."""
    fn: Callable[[R], T]
    name: str

    def __call__(self, record: R) -> T:
        return self.fn(record)


@overload
def measurable[R: Mapping[str, object], T](
    fn: Callable[[R], T], /,
) -> Measurable[R, T]: ...


@overload
def measurable[R: Mapping[str, object], T](
    *,
    name: str | None = None,
) -> Callable[[Callable[[R], T]], Measurable[R, T]]: ...


def measurable[R: Mapping[str, object], T](
    fn: Callable[[R], T] | None = None,
    /,
    *,
    name: str | None = None,
) -> Measurable[R, T] | Callable[[Callable[[R], T]], Measurable[R, T]]:
    """Wrap an `(R) -> T` function in a typed `Measurable[R, T]`.

    Both decorator forms supported:

        @measurable
        def constant_one(record: Mapping[str, object]) -> float:
            return 1.0

        @measurable(name='custom')
        def variable(record: DQNRecord) -> float:
            return float(record['q_max'])

    The first form (no parens) is canonical when no parameters
    are needed; the second form is for `name` overrides. Pyright
    infers `R` from the function's parameter annotation and `T`
    from its return type."""
    if fn is not None:
        return Measurable(fn=fn, name=fn.__name__)

    def decorator(inner: Callable[[R], T]) -> Measurable[R, T]:
        resolved_name = name if name is not None else inner.__name__
        return Measurable(fn=inner, name=resolved_name)
    return decorator
