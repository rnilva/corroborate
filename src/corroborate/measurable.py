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

`reads` is the leaf record keys this measurable ultimately
depends on — the dual of `Bridge.targets`. For a measurable
defined directly over `record['q_max']`, reads = `('q_max',)`.
For a reduction `max_abs(from_key('q_max'))`, reads propagates
to `('q_max',)` (the leaf, not the intermediate name). The
framework's reads-set discipline (axiom 19's redundancy
primitive) consumes this for fingerprinting.

Composition patterns (factories, dependent measurables) live in
`reductions.py` (for time-axis reductions) and as ordinary
Python factories returning `Measurable[R, T]`. The framework
explicitly does NOT ship a name-keyed registry + signature
resolver — value-based composition is the typed equivalent
without the stringly-keyed indirection."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import overload


@dataclass(frozen=True, slots=True)
class Measurable[R: Mapping[str, object], T]:
    """Typed generic wrapper. Behaves as `Callable[[R], T]`;
    carries `name` and `reads` as typed attributes. Both `R`
    (record type) and `T` (scalar return type) preserved through
    `__call__`.

    `reads` defaults to `()` for measurables whose leaf-key set
    is unknown (e.g. ad-hoc test fixtures). Reductions and the
    `from_key` primitive populate it explicitly so downstream
    invariant factories can derive `Bridge.targets` from it."""
    fn: Callable[[R], T]
    name: str
    reads: tuple[str, ...] = field(default=())

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
    reads: tuple[str, ...] = (),
) -> Callable[[Callable[[R], T]], Measurable[R, T]]: ...


def measurable[R: Mapping[str, object], T](
    fn: Callable[[R], T] | None = None,
    /,
    *,
    name: str | None = None,
    reads: tuple[str, ...] = (),
) -> Measurable[R, T] | Callable[[Callable[[R], T]], Measurable[R, T]]:
    """Wrap an `(R) -> T` function in a typed `Measurable[R, T]`.

    Both decorator forms supported:

        @measurable
        def constant_one(record: Mapping[str, object]) -> float:
            return 1.0

        @measurable(name='custom', reads=('q_max',))
        def q_max_value(record: DQNRecord) -> float:
            return float(record['q_max'])

    The first form (no parens) is canonical when no parameters
    are needed; the second form is for `name` / `reads` overrides.
    Pyright infers `R` from the function's parameter annotation
    and `T` from its return type."""
    if fn is not None:
        return Measurable(fn=fn, name=fn.__name__, reads=())

    def decorator(inner: Callable[[R], T]) -> Measurable[R, T]:
        resolved_name = name if name is not None else inner.__name__
        return Measurable(fn=inner, name=resolved_name, reads=reads)
    return decorator
