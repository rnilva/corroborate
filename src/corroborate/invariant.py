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

import math
from collections.abc import Callable, Mapping

from corroborate.bridge import Bridge, BridgeResult
from corroborate.claim import (
    Claim,
    ClaimBase,
    FnClaim,
    get_fn_invariants,
    _set_fn_invariants,  # pyright: ignore[reportPrivateUsage]
)
from corroborate.measurable import Measurable
from corroborate.verdict import Verdict


def attach_invariant(
    bridge: Bridge[Mapping[str, object]],
    *,
    to: Claim[..., object],
) -> None:
    """Attach a tautological Bridge to a claim so
    composition-discovery surfaces it when the claim is in a
    theory tree.

    Storage location depends on claim shape:

    - Free-function claim (`FnClaim`): keyed by underlying `fn`
      in the `_FN_INVARIANTS` side-table. All wrappers of the
      same fn share invariants.
    - Module claim (`ClaimBase` subclass instance): mutates
      `type(to).invariants` ClassVar — all instances of the
      class see the bridge."""
    if isinstance(to, FnClaim):
        existing = get_fn_invariants(to.fn)
        _set_fn_invariants(to.fn, existing + (bridge,))
        return
    if isinstance(to, ClaimBase):
        cls = type(to)
        existing_cls = cls.invariants
        cls.invariants = existing_cls + (bridge,)
        return
    raise TypeError(
        f'attach_invariant expects FnClaim or ClaimBase instance; '
        f'got {type(to).__name__}',
    )


def detach_invariant(
    bridge: Bridge[Mapping[str, object]],
    *,
    from_claim: Claim[..., object],
) -> None:
    """Reverse of `attach_invariant`. Removes `bridge` from
    `from_claim`'s invariant set (by identity). Used by tests +
    teardown code that mutate global state during an experiment.

    No-op if the bridge isn't attached."""
    if isinstance(from_claim, FnClaim):
        existing = get_fn_invariants(from_claim.fn)
        _set_fn_invariants(
            from_claim.fn,
            tuple(b for b in existing if b is not bridge),
        )
        return
    if isinstance(from_claim, ClaimBase):
        cls = type(from_claim)
        cls.invariants = tuple(b for b in cls.invariants if b is not bridge)
        return
    raise TypeError(
        f'detach_invariant expects FnClaim or ClaimBase instance; '
        f'got {type(from_claim).__name__}',
    )


def _build_tagged_bridge[R: Mapping[str, object]](
    fn: Callable[[R], BridgeResult],
    *,
    of: Claim[..., object],
    targets: tuple[str, ...],
    name: str | None = None,
) -> Bridge[R]:
    """Internal helper: build a tautological-tagged Bridge from an
    (R) → BridgeResult function. Used by `invariant` (build+attach)
    and `at_most` (build only)."""
    of_name = of.name
    resolved_name = name if name is not None else f'invariant_{fn.__name__}_of_{of_name}'

    def wrapper(record: R) -> BridgeResult:
        result = fn(record)
        tagged_stats: dict[str, float | int | bool | str] = {
            **result.stats,
            'kind': 'tautological',
            'of_claim': of_name,
        }
        return BridgeResult(
            verdict=result.verdict,
            reason=result.reason,
            stats=tagged_stats,
            name=result.name if result.name else resolved_name,
            targets=result.targets if result.targets else targets,
        )

    return Bridge(fn=wrapper, name=resolved_name, targets=targets)


def invariant[R: Mapping[str, object]](
    *,
    of: Claim[..., object],
    targets: tuple[str, ...],
    name: str | None = None,
) -> Callable[[Callable[[R], BridgeResult]], Bridge[R]]:
    """Decorator factory: build a tautological-tagged Bridge AND
    attach it to the claim's `invariants` ClassVar so
    composition-discovery surfaces it.

    Use this when defining substrate-level invariants the framework
    should auto-fire whenever the corresponding claim is in a
    theory tree. For one-off invariants meant to live on a single
    Hypothesis (without affecting the claim's class), use
    `at_most(...)` (which constructs without attaching) and pass
    explicitly via `Hypothesis.bridges`."""
    def decorator(fn: Callable[[R], BridgeResult]) -> Bridge[R]:
        bridge = _build_tagged_bridge(fn, of=of, targets=targets, name=name)
        # `attach_invariant` is typed at `Bridge[Mapping[str, object]]`;
        # `bridge` is `Bridge[R]` where R: Mapping[str, object]. The
        # variance is honest at runtime (record types vary across
        # invariants) but pyright needs the upper-bound cast.
        attach_invariant(bridge, to=of)  # pyright: ignore[reportArgumentType]
        return bridge
    return decorator


# ============ Theorem-gap factory: scope-commitment wrap ============

def at_most[R: Mapping[str, object]](
    gap: Measurable[R, float],
    threshold: float,
    *,
    of_claim: Claim[..., object],
    name: str | None = None,
) -> Bridge[R]:
    """Wrap a theorem-gap `Measurable[R, float]` in a tautological
    `Bridge[R]` whose verdict is:

    - `HELD` when `gap(record) <= threshold` (data confirms scope).
    - `INVARIANT_VIOLATION` when `gap(record) > threshold` (data
      shows the theorem's domain was exceeded).
    - `POWER_INSUFFICIENT` when `gap(record)` is NaN (no data was
      available to compute the gap — e.g. replay buffer never
      filled, fewer than two sync windows, eval pass produced no
      episodes). The author committed scope but the run can't
      tell whether scope held; treating NaN as HELD would be a
      silent false-confirmation.

    The wrap is the place where the *author commits scope*: the
    paper's claim "method X's mechanism operates when gap_Y <
    threshold" is written by including `at_most(gap_Y, threshold,
    of_claim=...)` in the Hypothesis's `bridges` tuple. The
    theorem reference is read off `gap.stats` (set when the gap
    Measurable was constructed) — `at_most` is consumer-side, the
    theorem identity lives with the gap.

    Three roles consume `at_most`-wrapped bridges:

    - Falsification: INVARIANT_VIOLATION preempts outcome verdict
      in `aggregate_verdict`. POWER_INSUFFICIENT means rerun at
      higher n / more eval bursts.
    - Causal-analysis scope predicate: gates discovery-input runs.
      NaN-bearing facts can be filtered out of the in-scope set.
    - Intervention sample-filter: gates cells before Δ-gap
      comparison aggregation. (The intervention's *outcome*
      verdict is on the Δ-gap, computed by existing stats; this
      bridge gates which cells feed that comparison.)

    `targets` derives from `gap.reads`. The reads-set propagates
    leaf-record-keys → measurable → bridge for redundancy +
    corpus-graph derivation."""
    bridge_name = name if name is not None else f'at_most[{gap.name}<={threshold:g}]'

    def fn(record: R) -> BridgeResult:
        val = gap(record)
        if math.isnan(val):
            return BridgeResult(
                verdict=Verdict.POWER_INSUFFICIENT,
                reason=(
                    f'{gap.name} = NaN (no data); cannot evaluate '
                    f'against threshold {threshold:g}'
                ),
                stats={
                    'gap_value': val,
                    'threshold': float(threshold),
                    'measurable': gap.name,
                },
                name=bridge_name,
                targets=gap.reads,
            )
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
    # Constructor only — does NOT auto-attach to of_claim.invariants.
    # Substrate authors who want auto-discovery via the composition
    # tree should pass the returned Bridge to `attach_invariant(..., to=of_claim)`.
    return _build_tagged_bridge(
        fn, of=of_claim, targets=gap.reads, name=bridge_name,
    )
