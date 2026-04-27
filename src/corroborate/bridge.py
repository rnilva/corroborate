"""Bridge — named paper-level assertion over a record.

A Bridge is a function that consumes a `Record` and produces a
`BridgeResult` containing a `Verdict`, an explanation, and stats
sufficient to audit the verdict. *Targets* — the record fields
the bridge reads — are declared at decoration time so the
framework can derive measurement-graph annotations and reads-set
fingerprints (used by axiom 19's redundancy primitive in later
modules).

`Record` is `Mapping[str, object]` — heterogeneous by design
(records hold per-step arrays, scalar metadata, env names,
seeds). The `object` value type is GENUINE polymorphism: the
framework is domain-neutral, so it cannot constrain the value
space. Bridge authors narrow at use site via `isinstance` /
PEP 742 `TypeIs[T]` predicates, never `cast`.

Bridges and Claims are distinct:
- Claims are STEPS in the algorithm — units of intervention.
- Bridges are TESTS over the algorithm's record — paper-level
  assertions about what the run produces.

Bridge factories — functions returning `Bridge` instances —
are the framework's pattern for parameterized bridges. No special
framework support; just compose `@bridge` with closures."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from corroborate.verdict import RefutationClass, Verdict


# Records hold heterogeneous values: arrays, scalars, env names,
# seeds, episode summaries. The `object` here is GENUINE
# polymorphism — the framework is domain-neutral, so it cannot
# constrain the value space. Bridge authors narrow at use site.
type Record = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """The outcome of applying a `Bridge` to a `Record`.

    `stats` are typed as a `Mapping` over scalar primitives —
    sufficient for audit values (ρ, ATE, sample sizes) and
    discrete labels (estimand expressions, tier markers). Lists,
    arrays, or nested structures do not belong here; they belong
    on the underlying `Record`. `refutation_class` is the optional
    PEP-742-narrowed sub-classification (see `verdict.py`); `None`
    when the verdict is HELD or INVARIANT_VIOLATION (no
    refutation to classify)."""
    verdict: Verdict
    reason: str
    stats: Mapping[str, float | int | bool | str]
    name: str
    targets: tuple[str, ...]
    refutation_class: RefutationClass | None = None


@dataclass(frozen=True, slots=True)
class Bridge:
    """A named paper-level assertion. Behaves as
    `Callable[[Record], BridgeResult]` via `__call__`; carries
    `name` and `targets` as typed attributes for graph derivation
    and reads-set fingerprinting."""
    fn: Callable[[Record], BridgeResult]
    name: str
    targets: tuple[str, ...]

    def __call__(self, record: Record) -> BridgeResult:
        return self.fn(record)


def bridge(
    *,
    targets: tuple[str, ...],
    name: str | None = None,
) -> Callable[[Callable[[Record], BridgeResult]], Bridge]:
    """Decorator factory wrapping a `(Record) -> BridgeResult`
    function in a typed `Bridge`. `targets` is explicit (the
    framework does not auto-introspect signatures); `name`
    defaults to `fn.__name__` when omitted.

    Usage:

        @bridge(targets=('max_q_late',))
        def max_q_decreases(record: Record) -> BridgeResult:
            ...

    Or as a factory closure:

        def monotonic_of(target: str) -> Bridge:
            @bridge(targets=(target,), name=f'monotonic({target})')
            def fn(record: Record) -> BridgeResult:
                ...
            return fn"""
    def decorator(fn: Callable[[Record], BridgeResult]) -> Bridge:
        resolved_name = name if name is not None else fn.__name__
        return Bridge(fn=fn, name=resolved_name, targets=targets)
    return decorator
