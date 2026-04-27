"""Invariant — theorem-direct bridge attached to a Claim.

An invariant in `corroborate` is a `Bridge[R]` carrying
`stats['kind']='tautological'` and `stats['of_claim']=...`. The
tautological tag flags the result so `aggregate_verdict` treats a
rejection as `INVARIANT_VIOLATION` (theorem out of scope; mechanism
didn't operate) rather than `NO_EFFECT` (mechanism tested and
refuted). Axiom 18: theorem-direct, not proxy-via-assumption.

**Theorem-gap framing.** The primary primitive at the theorem
layer is *gap magnitude*, not threshold-bounded boolean tests.
For non-trivial domains (deep RL under deadly triad), the
literal theorem conditions categorically don't hold — Banach
contraction fails under FA + bootstrap, Watkins's tabular
convergence doesn't apply, linear ε violates strict GLIE by
construction, etc. Asking "is this run inside the theorem's
domain?" is dishonest because the answer is no for *every* run.
The principled question is "how far from the domain is it, and
does the intervention reduce that distance?"

So invariants in this framework are continuous gap-Measurables.
The threshold-wrap (`at_most(gap, threshold)`) is one consumer
of the gap, used when an author commits a scope claim. Three
roles consume the same gap primitive differently:

- **Intervention study (role 1):** read the scalar gap directly
  into `RunRow.stats`; per-comparison Δ-gap on
  `ComparisonRow.stats`. The verdict on the *Δ-gap difference*
  comes from existing power/MDE machinery (Hedges' g, SE,
  POWER_INSUFFICIENT), not from a threshold on the gap itself.
  Scope-conditioning via `at_most`-wraps gates which cells feed
  the comparison aggregation.
- **Falsification (role 2):** `at_most(gap, threshold)` wraps
  the gap as a `Bridge[R]` returning HELD or INVARIANT_VIOLATION.
  The threshold is the author's commitment, written into
  `Hypothesis.bridges`. INVARIANT_VIOLATION preempts outcome
  verdicts in `aggregate_verdict`.
- **Causal analysis (role 3):** same `at_most`-wraps used as
  scope predicate; gates which runs feed Pearson / PC / mediation
  discovery. Without scope conditioning, discovery includes
  off-mechanism runs and biases the inferred graph.

Same primitive, three downstream pipelines. The author commits
scope by writing `at_most(...)` calls into their hypothesis's
`bridges` tuple — no new field on Hypothesis.

**Numerical guardrails are NOT invariants.** Static checks like
`assert 0 <= eps_final <= eps_init <= 1` (Kolmogorov axiom) live
inline in claim bodies. INVARIANT_VIOLATION is reserved for
*theorem-domain* departures; numerical sanity has its own
mechanism."""
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

    For theorem-condition invariants composed from gap
    measurables, prefer the `at_most(...)` factory — this raw
    decorator is the lower-level primitive."""
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


# ============ Theorem-gap factory: scope-commitment wrap ============

def at_most[R: Mapping[str, object]](
    gap: Measurable[R, float],
    threshold: float,
    *,
    of_claim: ClaimRecord,
    name: str | None = None,
) -> Bridge[R]:
    """Wrap a theorem-gap `Measurable[R, float]` in a tautological
    `Bridge[R]` returning `HELD` when `gap(record) <= threshold`
    and `INVARIANT_VIOLATION` otherwise.

    The wrap is the place where the *author commits scope*: the
    paper's claim "method X's mechanism operates when gap_Y <
    threshold" is written by including `at_most(gap_Y, threshold,
    of_claim=...)` in the Hypothesis's `bridges` tuple. The
    theorem reference is read off `gap.stats` (set when the gap
    Measurable was constructed) — `at_most` is consumer-side, the
    theorem identity lives with the gap.

    Three roles consume `at_most`-wrapped bridges:

    - Falsification: INVARIANT_VIOLATION preempts outcome verdict
      in `aggregate_verdict`.
    - Causal-analysis scope predicate: gates discovery-input runs.
    - Intervention sample-filter: gates cells before Δ-gap
      comparison aggregation. (The intervention's *outcome*
      verdict is on the Δ-gap, computed by existing stats; this
      bridge gates which cells feed that comparison.)

    `targets` derives from `gap.reads`. The reads-set propagates
    leaf-record-keys → measurable → bridge for redundancy +
    corpus-graph derivation."""
    bridge_name = name if name is not None else f'at_most[{gap.name}<={threshold:g}]'

    @invariant(of=of_claim, targets=gap.reads, name=bridge_name)
    def fn(record: R) -> BridgeResult:
        val = gap(record)
        ok = val <= threshold
        return BridgeResult(
            verdict=Verdict.HELD if ok else Verdict.INVARIANT_VIOLATION,
            reason=f'{gap.name} = {val:.4g} vs threshold {threshold:g}',
            stats={
                'gap_value': float(val),
                'threshold': float(threshold),
                'measurable': gap.name,
            },
            name=bridge_name,
            targets=gap.reads,
        )
    return fn
