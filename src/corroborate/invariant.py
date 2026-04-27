"""Invariant — theorem-direct bridge attached to a Claim.

An `Invariant` is a `Bridge` with two distinguishing properties:

1. *Attached to a Claim.* The invariant tests a property of the
   theorem behind that claim — e.g. Q-boundedness for a tabular
   contraction claim, or Bellman residual decay for a DQN claim.
   The relationship is recorded in `stats['of_claim']` so
   downstream consumers can group invariants by claim.

2. *Tautological tag.* `stats['kind'] = 'tautological'` flags the
   result so `aggregate_verdict` (later module) treats a REJECT
   as `INVARIANT_VIOLATION` (mechanism didn't operate; claim is
   out of scope under this run) rather than `NO_EFFECT` (claim
   was tested and refuted). This is axiom 18: invariants are
   theorem-direct, not proxy-via-assumption.

The author writes the invariant body as an ordinary
`(Record) -> BridgeResult` function; `@invariant` injects the
`kind` and `of_claim` tags into the returned `BridgeResult`'s
stats automatically.

For v0 the framework provides only the decorator. Aggregation
that maps `kind=='tautological'` REJECT to
`INVARIANT_VIOLATION` lands in the verdict-aggregation module
(later step). Until then, invariant results carry the tag but
consumers are free to ignore it."""
from __future__ import annotations

from collections.abc import Callable

from corroborate.bridge import Bridge, BridgeResult, Record
from corroborate.claim import ClaimRecord


def invariant(
    *,
    of: ClaimRecord,
    targets: tuple[str, ...],
    name: str | None = None,
) -> Callable[[Callable[[Record], BridgeResult]], Bridge]:
    """Decorator factory: wraps a `(Record) -> BridgeResult`
    function in a Bridge whose results carry `stats['kind'] =
    'tautological'` and `stats['of_claim'] = of.name`.

    Tag injection is automatic — the inner function does not need
    to set `kind` or `of_claim` itself. Existing `stats` on the
    returned BridgeResult are preserved; the invariant tags
    overlay/extend.

    Usage:

        @claim
        def vanilla_greedify(q: Array) -> int: ...

        @invariant(of=vanilla_greedify, targets=('max_q_late',))
        def q_bounded(record: Record) -> BridgeResult:
            ...
    """
    def decorator(fn: Callable[[Record], BridgeResult]) -> Bridge:
        resolved_name = name if name is not None else f'invariant_{fn.__name__}_of_{of.name}'

        def wrapper(record: Record) -> BridgeResult:
            result = fn(record)
            tagged_stats: dict[str, float | int | bool | str] = {
                **result.stats,
                'kind': 'tautological',
                'of_claim': of.name,
            }
            return BridgeResult(
                verdict=result.verdict,
                reason=result.reason,
                stats=tagged_stats,
                name=result.name if result.name else resolved_name,
                targets=result.targets if result.targets else targets,
                refutation_class=result.refutation_class,
            )

        return Bridge(fn=wrapper, name=resolved_name, targets=targets)
    return decorator
