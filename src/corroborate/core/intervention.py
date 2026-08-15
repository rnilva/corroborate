"""Intervention — typed structural delta to a theory.

A Hypothesis describes CHANGES to the theory; `Intervention`s are
the typed primitives describing each individual slot swap. A
`DoEffect` carries N arms, each a tuple of `Intervention`s — Pearl's
`do(·)` operator for that arm. Binary contrast is the special case
N=2, often with one empty-tuple "no intervention" arm.

Pearl-coherence: each arm in `DoEffect.arms` is a joint
intervention `do(slot_1 = v_1, slot_2 = v_2, ...)`. Multi-level
sweeps (dose-response), factorial designs, and N-way contrasts
are first-class — the framework dispatches one cell per arm; the
analysis-side primitives (per-pair `paired_g`, dose-response,
`factorial_2x2`) pick the reference structure separately.

Intervention vs HP variation. HPs (γ, lr, batch_size, total_steps)
MUST NOT appear as Interventions — they are cell covariates that
meta-regression cleaves on (cf. v9's aggregation reframing). Slot
swaps (`bootstrap`, `replay.sample`, `action_select`) ARE
Interventions — they describe the structural mechanism delta.

Two hypotheses with the same arm contents but different HP grid
points share an arm key and pair as same-arm cells; the HP
difference becomes a covariate, not an arm distinguisher.

Identity is via static fingerprint (`canonical_str`). The runtime
graph-signature delta check (per-arm `signature(g_arm_i) -
signature(g_arm_j)`) — the principled HPO-smuggle gate — is
deferred (`FUTURE_WORKS.md` line 41). A static fingerprint cannot
distinguish "scalar-only diff to a partial's keywords" (HP smuggle
that ought to be rejected) from "structural replacement swap"
(real intervention). Both produce different `arm_key()` values;
the runtime check is what would reject the former when the
dialectic loop forces it."""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeIs

from corroborate._internals.canonical import canonical_str


class ArmRole(StrEnum):
    """Typed sentinel for "which side of a BINARY DoEffect
    contrast" — bridge authors use this in place of literal
    arm_key strings, so the bridge body stays decoupled from the
    persistence-side canonical fingerprint of the unmodified or
    treatment arm.

    Binary-only by design. When `DoEffect` carries N > 2 arms,
    bridges use `arm_keys[i]` indexing (or explicit string keys)
    instead — the `BASELINE` / `TREATMENT` semantics don't extend
    naturally to multi-level interventions. The bridge dispatcher
    raises if an `ArmRole` sentinel appears in a non-binary
    bridge's params.

    Resolution lives in `bridge.py`'s evaluate path: when a
    bridge's source is a binary DoEffect, `ArmRole.TREATMENT`
    resolves to `arm_keys[1]` and `ArmRole.BASELINE` to
    `arm_keys[0]`."""
    BASELINE = 'baseline'
    TREATMENT = 'treatment'

type Replacement = object
"""Runtime universe of slot replacements: any value the substrate
admits at a `slot_path`. Two shapes are common:

1. **Callable slot swaps** — `FnClaim` wrappers, `functools.partial`
   bindings, plain callables (covers class-based Claims via the
   `record_call` escape hatch). The canonical case the framework
   was originally designed for; `bootstrap`, `action_select`,
   `replay.sample` all live here.
2. **Config-bundle slot swaps** — frozen-dataclass instances
   (`Replay(capacity=5000)`, `MLP(hidden=[64])`, etc.). The slot
   value carries construction-time HPs + slot-Claim refs, all in
   one immutable record.

`canonical_str` handles both — two structurally-equal
replacements produce the same fingerprint, so arm identity is
preserved across processes. The `Replacement` alias is `object`
because the framework's substrate-agnostic shape can't enforce
"this slot accepts a callable" without knowing the substrate's
type contract; the substrate's own type-checker catches slot/
replacement mismatches at the call site."""


def is_replacement(v: object) -> TypeIs[Replacement]:
    """Narrow `v` to `Replacement`. With `Replacement = object`
    this is trivially True — kept as a typed gate so YAML
    parsers and other ingest sites have a single named predicate
    to call rather than dropping the check."""
    del v
    return True


@dataclass(frozen=True, slots=True)
class Intervention:
    """A typed structural delta: at `slot_path`, swap the parent
    claim for `replacement`.

    `slot_path` is a dotted topology path (`bootstrap`,
    `replay.sample`, `action_select`) addressing a kwarg slot of
    the bound theory. `replacement` is the substitute Claim,
    callable, or `functools.partial`.

    Two Interventions with the same `slot_path` and a
    canonically-equal `replacement` produce the same `arm_key()`.
    HP variation does NOT appear at this layer."""
    slot_path: str
    replacement: Replacement

    def arm_key(self) -> str:
        return f'{self.slot_path}={canonical_str(self.replacement)}'

    def apply[T](
        self, base: Callable[..., T],
    ) -> functools.partial[T]:
        """Apply this do() to a base composition.

        Pearl: `do(slot_path = replacement)` on the SCM `base`.
        Returns the post-intervention SCM with `slot_path` pinned
        to `replacement` and every other mechanism preserved.
        Implemented as `functools.partial(base, slot_path=
        replacement)` — single-slot surgical replacement, all
        other slots fall through to base's defaults / outer
        partial bindings.

        Nested `slot_path` (containing `.`) is NOT supported —
        substituting at depth would require knowing the parent
        claim's structure to reconstruct it. Authors with nested
        intent construct the parent claim explicitly with the
        substituted child:

            # Wrong (not supported):
            Intervention(slot_path='replay.sample', replacement=X)
            # Right:
            Intervention(slot_path='replay',
                         replacement=Replay(sample=X))

        The framework's config bundles are frozen dataclasses;
        constructing the substituted parent is the implementation
        author's idiom."""
        if '.' in self.slot_path:
            raise ValueError(
                f'nested slot_path {self.slot_path!r} not supported '
                f'by Intervention.apply(); construct the parent '
                f'claim with the nested override explicitly.',
            )
        return functools.partial(
            base, **{self.slot_path: self.replacement},
        )


def apply_interventions[T](
    base: Callable[..., T],
    interventions: Sequence[Intervention],
) -> Callable[..., T]:
    """Sequentially apply a tuple of `do()`s to a base composition.

    Pearl: `do()`s on disjoint variables commute; on overlapping
    variables, the later one shadows the earlier (last-wins via
    `functools.partial`'s kwarg merge). Order is preserved so the
    overlapping case is deterministic, but callers should design
    intervention tuples with disjoint `slot_path`s — overlapping
    interventions on the same slot are a code smell."""
    composed: Callable[..., T] = base
    for iv in interventions:
        composed = iv.apply(composed)
    return composed


def combined_arm_key(interventions: tuple[Intervention, ...]) -> str:
    """Stable arm fingerprint for a tuple of Interventions.

    Sorted by slot_path so order-of-construction does not change
    identity. Empty tuple → `'baseline'`; non-empty → `'+'`-joined
    arm keys."""
    if not interventions:
        return 'baseline'
    return '+'.join(sorted(i.arm_key() for i in interventions))


@dataclass(frozen=True, slots=True)
class DoEffect:
    """Multi-arm contrast on the claim graph. Each arm is a tuple
    of slot-replacement `Intervention`s; arm identity is the
    `combined_arm_key` of that tuple.

    Pearl-rung-2 edges in the causal graph have an *intervention*
    as the source node, NOT a measurable. `DoEffect` carries N
    arms — each a `do(joint_intervention)` operation in Pearl's
    calculus. The framework dispatches one cell per (grid_point,
    arm) and tags each `RunRow.arm_key` with the canonical
    fingerprint of its arm.

    The empty-tuple arm (when present) is the "no intervention"
    control — the substrate's vanilla composition. Non-empty arms
    are interventional structural deltas. Binary contrast is the
    special case `arms=(control, treatment)` with one tuple
    typically empty. Multi-level (dose-response) is
    `arms=((), (level_1,), (level_2,))`. Factorial 2×2 is
    `arms=((), (a,), (b,), (a, b))`.

    Pearl-coherence: each arm is a joint `do(·)` operation;
    `DoEffect.arms` declares the structural intervention
    conditions. Analyses (per-pair `paired_g` /
    `stratified_arm_diff_pooled`, dose-response, factorial) pick
    the contrast structure separately from the arm declaration.
    The framework's job is to dispatch arms; analyses choose
    references."""
    arms: tuple[tuple[Intervention, ...], ...]

    def arm_keys(self) -> tuple[str, ...]:
        """Canonical fingerprint per arm. Empty tuple → `'baseline'`;
        non-empty → `'+'`-joined slot=replacement keys. The string
        flows to `RunRow.arm_key` per cell, and analyses filter
        cells by string match on these keys."""
        return tuple(combined_arm_key(a) for a in self.arms)

    def node_key(self) -> str:
        """Canonical string identity for the do-node in
        `CausalGraph`. Renders as `do(arm0|arm1|...|armN)` — the
        multi-arm contrast made explicit at the intervention
        node."""
        return f"do({'|'.join(self.arm_keys())})"
