"""Invariant — theorem-direct bridge attached to a Claim.

An `Invariant` is a `Bridge[R]` with two distinguishing
properties:

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

Generic in `R: Mapping[str, object]` (the record schema), same
as `Bridge[R]`. The author writes the body as an ordinary
`(R) -> BridgeResult` function; `@invariant` injects the `kind`
and `of_claim` tags into the returned `BridgeResult`'s stats
automatically.

`bounded(of=Measurable, threshold, ...)` is the canonical
factory for theorem-condition invariants: lifts a `Measurable[R,
float]` into an `INVARIANT_VIOLATION`-on-overflow bridge with
the theorem reference recorded in `stats['theorem']`."""
from __future__ import annotations

from collections.abc import Callable, Mapping

from corroborate.bridge import Bridge, BridgeResult
from corroborate.claim import ClaimRecord
from corroborate.measurable import Measurable
from corroborate.verdict import Verdict


def invariant[R: Mapping[str, object]](
    *,
    of: ClaimRecord,
    targets: tuple[str, ...],
    name: str | None = None,
) -> Callable[[Callable[[R], BridgeResult]], Bridge[R]]:
    """Decorator factory: wraps an `(R) -> BridgeResult` function
    in a `Bridge[R]` whose results carry `stats['kind'] =
    'tautological'` and `stats['of_claim'] = of.name`.

    Tag injection is automatic — the inner function does not need
    to set `kind` or `of_claim` itself. Existing `stats` on the
    returned BridgeResult are preserved; the invariant tags
    overlay/extend.

    Usage:

        @claim
        def vanilla_greedify(q: Array) -> int: ...

        @invariant(of=vanilla_greedify, targets=('max_q_late',))
        def q_bounded(record: Mapping[str, object]) -> BridgeResult:
            ...
    """
    def decorator(fn: Callable[[R], BridgeResult]) -> Bridge[R]:
        resolved_name = name if name is not None else f'invariant_{fn.__name__}_of_{of.name}'

        def wrapper(record: R) -> BridgeResult:
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
            )

        return Bridge(fn=wrapper, name=resolved_name, targets=targets)
    return decorator


# ============ Theorem-condition invariant factory ============

def bounded[R: Mapping[str, object]](
    of: Measurable[R, float],
    threshold: float,
    *,
    theorem: str,
    of_claim: ClaimRecord,
    name: str | None = None,
) -> Bridge[R]:
    """Tautological invariant: `|of(record)| < threshold`.

    Composes a `Measurable[R, float]` value into a `Bridge[R]`
    whose verdict is `HELD` when the bound holds and
    `INVARIANT_VIOLATION` when it doesn't. `theorem` records the
    reference (e.g. `'Banach contraction on T*'`) in
    `stats['theorem']` so the verdict layer can cite the
    out-of-scope reason — INVARIANT_VIOLATION ⇒ this run sat
    outside the theorem's domain of applicability, NOT that the
    paper-claim was empirically refuted.

    `targets` derives from `of.reads`. The reads-set propagation
    means a reduction (`max_abs(from_key('max_q'))`) attached as
    an invariant to `mlp_q` correctly fingerprints the bridge's
    record-key dependencies for redundancy + corpus-graph
    derivation."""
    inv_name = name if name is not None else f'bounded[{of.name}<{threshold:g}]'

    @invariant(of=of_claim, targets=of.reads, name=inv_name)
    def fn(record: R) -> BridgeResult:
        val = of(record)
        ok = abs(val) < threshold
        return BridgeResult(
            verdict=Verdict.HELD if ok else Verdict.INVARIANT_VIOLATION,
            reason=f'|{of.name}| = {val:.4g} vs threshold {threshold:g}',
            stats={
                'theorem': theorem,
                'value': val,
                'threshold': float(threshold),
                'measurable': of.name,
            },
            name=inv_name,
            targets=of.reads,
        )
    return fn
