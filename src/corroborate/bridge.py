"""Bridge — named paper-level assertion over a record.

A `Bridge[R]` is a function `(R) -> BridgeResult` where `R` is
the record's type. The bound `R: Mapping[str, object]` accepts
plain `Mapping[str, object]` (framework-neutral), uniform-value
mappings like `Mapping[str, jax.Array]`, TypedDicts (per-key
typed records), and any custom Mapping subclass. Authors who
declare a TypedDict for their record schema get fully typed
field access inside the bridge body — no `isinstance` narrows.

`targets` — the record fields the bridge reads — are declared
at decoration time so the framework can derive measurement-graph
annotations and reads-set fingerprints (axiom 19's redundancy
primitive consumes them in later modules).

Bridges and Claims are distinct:
- Claims are STEPS in the algorithm — units of intervention.
- Bridges are TESTS over the algorithm's record — paper-level
  assertions about what the run produces.

Bridge factories — functions returning `Bridge[R]` instances —
are the framework's pattern for parameterized bridges. No special
framework support; just compose `@bridge` with closures."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from corroborate.verdict import RefutationClass, Verdict


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """The outcome of applying a `Bridge` to a record.

    `stats` are typed as a `Mapping` over scalar primitives —
    sufficient for audit values (ρ, ATE, sample sizes) and
    discrete labels (estimand expressions, tier markers). Lists,
    arrays, or nested structures do not belong here; they belong
    on the underlying record. `refutation_class` is the optional
    sub-classification (see `verdict.py`); `None` when the
    verdict is HELD or INVARIANT_VIOLATION (no refutation to
    classify)."""
    verdict: Verdict
    reason: str
    stats: Mapping[str, float | int | bool | str]
    name: str
    targets: tuple[str, ...]
    refutation_class: RefutationClass | None = None


@dataclass(frozen=True, slots=True)
class Bridge[R: Mapping[str, object]]:
    """A named paper-level assertion. Behaves as
    `Callable[[R], BridgeResult]` via `__call__`; carries `name`
    and `targets` as typed attributes for graph derivation and
    reads-set fingerprinting.

    `R` is the record schema. Bound at `Mapping[str, object]` —
    accepts plain Mappings, uniform-value mappings, TypedDicts,
    and Mapping subclasses. Authors using TypedDict get typed
    field access inside the bridge body."""
    fn: Callable[[R], BridgeResult]
    name: str
    targets: tuple[str, ...]

    def __call__(self, record: R) -> BridgeResult:
        return self.fn(record)


def bridge[R: Mapping[str, object]](
    *,
    targets: tuple[str, ...],
    name: str | None = None,
) -> Callable[[Callable[[R], BridgeResult]], Bridge[R]]:
    """Decorator factory wrapping an `(R) -> BridgeResult`
    function in a typed `Bridge[R]`. `R` is inferred from the
    function's parameter annotation. `targets` is explicit;
    `name` defaults to `fn.__name__` when omitted.

    Usage with Mapping[str, object] (framework-neutral):

        @bridge(targets=('max_q_late',))
        def max_q_decreases(record: Mapping[str, object]) -> BridgeResult:
            v = record['max_q_late']
            if isinstance(v, float): ...

    Usage with TypedDict (typed-body access):

        class DQNRecord(TypedDict):
            max_q_late: float
            epsilon: float

        @bridge(targets=('max_q_late',))
        def max_q_decreases(record: DQNRecord) -> BridgeResult:
            v = record['max_q_late']  # typed as float — no narrow
            ...

    Factory closure:

        def monotonic_of(target: str) -> Bridge[Mapping[str, object]]:
            @bridge(targets=(target,), name=f'monotonic({target})')
            def fn(record: Mapping[str, object]) -> BridgeResult:
                ...
            return fn"""
    def decorator(fn: Callable[[R], BridgeResult]) -> Bridge[R]:
        resolved_name = name if name is not None else fn.__name__
        return Bridge(fn=fn, name=resolved_name, targets=targets)
    return decorator
